"""Write a rig as a MOC3 deformer tree (Cubism v3.03), plus model3/cdi3.

The rig's apply() stays the source of truth; it is factored the way Cubism
models are built (rules checked against Core):

  Body      root warp keyed on the body parameters. Its keyforms are the
            body field on a grid (turn, lean, breath) followed by the
            body affine (tilt, breath scale) sampled from the rig.
  <group>   warp under Body, keyed on ParamAngleX x Y x Z (27 keyforms).
            Parts that move as one piece (eyes, brows, mouth, nose) are
            affine -> 1x1. Face, hair and bun get a grid fitted by least
            squares to the rig's positions at every own and head keyform.
  mesh      keyforms over its own parameters only, in its parent's [0,1].

A mesh the fitted grid misses by more than TOL px goes directly under Body,
keyed on its own and head parameters, so it stays exact. Opacity that
depends on a head parameter keeps that parameter in the mesh. Body layers
that only switch on and off share parent warps that carry the opacity.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent/'vendor/py-moc3/src'))
from moc3 import Moc3, SECTION_LAYOUT  # noqa: E402
from moc3._core import ElemType, ADDITIONAL_V303  # noqa: E402

# Two field pairs of this serializer version are swapped; recheck when
# upgrading. Checked by name so a second import never swaps them back.
if SECTION_LAYOUT[41].name == 'art_mesh.position_index_counts':
    SECTION_LAYOUT[41], SECTION_LAYOUT[44] = SECTION_LAYOUT[44], SECTION_LAYOUT[41]
    SECTION_LAYOUT[82], SECTION_LAYOUT[83] = SECTION_LAYOUT[83], SECTION_LAYOUT[82]

HEAD = ['ParamAngleX', 'ParamAngleY', 'ParamAngleZ']
BODY = ['ParamBodyAngleZ', 'ParamBreath']
PPU = 1000.
TOL = .35                                  # px, max fitted-grid miss per vertex
TOL_SWUNG = TOL
SWUNG_WEIGHT = .1                          # fit priority of samples with hair swung
CELL_PX = {'Face': 12., 'Hair': 12., 'Bun': 12., 'Gap_L': 12., 'Gap_R': 12.}   # starting cell size
REFINE = 5                                 # rounds of splitting missed cells
MIN_CELL = 1.5                             # px
QUAD = True
MARGIN = 24.                               # px around a group's points
OPACITY_WARPS = True


# ---- warp maths (Core: per-cell bilinear when quad, else two triangles) ----

def warp_weights(uv, rows, cols, quad):
    x = uv[:, 0]*cols; y = uv[:, 1]*rows
    c0 = np.clip(np.floor(x).astype(int), 0, cols-1); r0 = np.clip(np.floor(y).astype(int), 0, rows-1)
    fx = x-c0; fy = y-r0
    i00 = r0*(cols+1)+c0
    idx = np.stack([i00, i00+1, i00+cols+1, i00+cols+2], 1)
    if quad:
        w = np.stack([(1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy], 1)
    else:
        lo = fx+fy <= 1
        w = np.where(lo[:, None], np.stack([1-fx-fy, fx, fy, 0*fx], 1),
                     np.stack([0*fx, 1-fy, 1-fx, fx+fy-1], 1))
    return idx, w


def warp_apply(grid, rows, cols, uv, quad):
    idx, w = warp_weights(uv, rows, cols, quad)
    return (grid[idx]*w[:, :, None]).sum(1)


def uv_of(p, F):
    """px -> [0,1] of a warp whose rest grid lines are F['xs'], F['ys']."""
    p = np.asarray(p, float)
    return np.stack([np.interp(p[:, 0], F['xs'], np.linspace(0, 1, len(F['xs']))),
                     np.interp(p[:, 1], F['ys'], np.linspace(0, 1, len(F['ys'])))], 1)


def fit_grid(uv, target, rows, cols, quad, rest, smooth=1e-3, iters=3000, sw=None):
    """Least-squares grid so warp(grid, uv) ~ target; smooth where no data
    (conjugate gradients on data term + second differences + tiny ridge)."""
    n = (rows+1)*(cols+1)
    idx, w = warp_weights(uv, rows, cols, quad)
    fi = idx.ravel()
    resid = target-(rest[idx]*w[:, :, None]).sum(1)
    sw = np.ones(len(uv)) if sw is None else np.asarray(sw, float)

    def Wt(y):
        return np.stack([np.bincount(fi, (w*(sw*y[:, k])[:, None]).ravel(), n) for k in (0, 1)], 1)

    def sd(D):
        G = D.reshape(rows+1, cols+1, 2)
        return G[:, :-2]-2*G[:, 1:-1]+G[:, 2:], G[:-2]-2*G[1:-1]+G[2:]

    def sdT(ex, ey):
        out = np.zeros((rows+1, cols+1, 2))
        out[:, :-2] += ex; out[:, 1:-1] -= 2*ex; out[:, 2:] += ex
        out[:-2] += ey; out[1:-1] -= 2*ey; out[2:] += ey
        return out.reshape(n, 2)
    diag = np.bincount(fi, (w*w*sw[:, None]).ravel(), n)
    lam = smooth*max(diag.mean(), 1e-6); mu = 1e-6*max(diag.mean(), 1e-6)

    def A(D):
        return Wt((D[idx]*w[:, :, None]).sum(1))+lam*sdT(*sd(D))+mu*D
    b = Wt(resid)
    M = 1/(diag+12*lam+mu)
    D = np.zeros((n, 2)); r = b-A(D); z = M[:, None]*r; p = z.copy()
    rz = (r*z).sum(); b2 = (b*b).sum()
    for _ in range(iters):
        Ap = A(p); a = rz/(p*Ap).sum()
        D += a*p; r -= a*Ap
        if (r*r).sum() < 1e-20*b2:
            break
        z = M[:, None]*r; rz2 = (r*z).sum(); p = z+(rz2/rz)*p; rz = rz2
    return rest+D


def fit_group(g, members, samples, head_combos):
    """Rest grid lines over the members' points; refine the lines through
    cells where any keyform misses a point by more than TOL."""
    S0, _, (who, moved) = samples(members, {a: 0. for a in HEAD})
    tol = np.where(moved, TOL_SWUNG, TOL)
    x0, y0 = S0.min(0)-MARGIN; x1, y1 = S0.max(0)+MARGIN
    if g in CELL_PX:
        xs = np.linspace(x0, x1, int(np.ceil((x1-x0)/CELL_PX[g]))+1)
        ys = np.linspace(y0, y1, int(np.ceil((y1-y0)/CELL_PX[g]))+1)
    else:
        xs, ys = np.array([x0, x1]), np.array([y0, y1])
    for it in range(REFINE+1):
        F = dict(xs=xs, ys=ys, rows=len(ys)-1, cols=len(xs)-1)
        rows, cols = F['rows'], F['cols']
        gx, gy = np.meshgrid(xs, ys)
        restg = np.stack([gx, gy], -1).reshape(-1, 2)
        uv = uv_of(S0, F)
        forms, err = [], np.zeros(len(S0))
        for h in head_combos:
            _, T, _ = samples(members, h)
            if all(h[a] == 0 for a in HEAD):
                G = restg.copy()
            elif (rows, cols) == (1, 1):
                X = np.c_[S0, np.ones(len(S0))]
                M, *_ = np.linalg.lstsq(X, T, rcond=None)
                G = np.c_[restg, np.ones(4)]@M
            else:
                G = fit_grid(uv, T, rows, cols, QUAD, restg, sw=np.where(moved, SWUNG_WEIGHT, 1.))
            err = np.maximum(err, np.linalg.norm(warp_apply(G, rows, cols, uv, QUAD)-T, axis=1))
            forms.append(G)
        F['forms'] = forms
        bad = err > tol
        if not bad.any() or (rows, cols) == (1, 1) or it == REFINE:
            break

        def split(lines, coord):
            k = np.clip(np.searchsorted(lines, coord, side='right')-1, 0, len(lines)-2)
            k = np.unique(k)
            k = k[(lines[k+1]-lines[k]) > 2*MIN_CELL]
            return np.sort(np.r_[lines, (lines[k]+lines[k+1])/2])
        nx, ny = split(xs, S0[bad, 0]), split(ys, S0[bad, 1])
        if len(nx) == len(xs) and len(ny) == len(ys):
            break
        xs, ys = nx, ny
    worst = {nm: (float(err[(who == nm) & ~moved].max()),
                  float(err[(who == nm) & moved].max()) if ((who == nm) & moved).any() else 0.)
             for nm in members}
    return F, worst


def write(rig, meshes, out: Path, name: str, canvas, log=print):
    """meshes: prepared drawables (name, vertices, uvs, triangles, draw_order, visible)."""
    out.mkdir(parents=True, exist_ok=True)
    W, H = canvas
    origin = (W/2., H/2.)
    PARAMS = rig.params
    names = [m['name'] for m in meshes]
    AXES = rig.axes
    default = rig.default
    combos = rig.combos
    rest = {a: default[a] for a in HEAD+BODY}
    swing_loose = rig.swing_loose
    body_extra = rig.body_extra
    body_lines = rig.body_grid
    groups_of = rig.groups

    def run(mm, v, p):
        pts, op = rig.apply(mm['name'], v, {**default, **p}, mm.get('visible', True))
        return np.asarray(pts, float), float(op)

    # ---- sample the rig: every own x head keyform, body at rest ----
    info = {}
    for mm in meshes:
        nm = mm['name']
        axes = AXES.get(nm, [])
        assert all(a in axes for a in BODY), nm
        own = [a for a in axes if a not in HEAD+BODY]
        head = [a for a in HEAD if a in axes]
        v = np.asarray(mm['vertices'], float)
        pos, ops = {}, {}
        for o in combos(own):
            key = tuple(o.items())
            for h in combos(head):
                pts, op = run(mm, v, {**rest, **o, **h})
                pos[key, tuple(h.items())] = pts
                ops[key, tuple(h.items())] = op
        op_needs_head = any(len({ops[k, hh] for hh in [tuple(h.items()) for h in combos(head)]}) > 1
                            for k in {k for k, _ in ops})
        L = rig.layers[nm]
        info[nm] = dict(own=own, head=head, v=v, pos=pos, ops=ops, op_head=op_needs_head,
                        group=L.warp if L.group == 'head' else None)

    # ---- body: affine per body keyform, checked on every vertex ----
    src = np.vstack([info[n]['pos'][(tuple((a, default[a]) for a in info[n]['own']),
                                     tuple((a, 0.) for a in info[n]['head']))] for n in names])
    body_err, affine = 0., {}
    for b in combos(BODY):
        tgt = []
        for mm in meshes:
            i = info[mm['name']]
            o = {a: default[a] for a in i['own']}
            tgt.append(run(mm, i['v'], {**rest, **o, **b})[0])
        tgt = np.vstack(tgt)
        X = np.c_[src, np.ones(len(src))]
        M, *_ = np.linalg.lstsq(X, tgt, rcond=None)
        body_err = max(body_err, np.abs(X@M-tgt).max())
        affine[tuple(b.items())] = M
    log(f'body: affine fit max err {body_err:.2e} px over {len(src)} vertices')
    gx, gy = np.meshgrid(body_lines['xs'], body_lines['ys'])
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    body_axes = body_extra+BODY
    body_rows, body_cols = len(body_lines['ys'])-1, len(body_lines['xs'])-1
    body_forms = []
    for p in combos(body_axes):
        pts = rig.body_field(grid, p)
        body_forms.append(np.c_[pts, np.ones(len(pts))]@affine[tuple((a, p[a]) for a in BODY)])
    log(f'body: {body_rows}x{body_cols} grid, {len(body_forms)} keyforms over {body_axes}')

    # ---- head groups: fit grids, drop meshes the grid misses ----
    groups = {}
    for nm in names:
        g = info[nm]['group']
        if g:
            groups.setdefault(g, []).append(nm)
    head_combos = combos(HEAD)

    def samples(members, h):
        S, T, who, moved = [], [], [], []
        for nm in members:
            i = info[nm]
            for o in combos(i['own']):
                k = tuple(o.items())
                S.append(i['pos'][k, tuple((a, 0.) for a in i['head'])])
                T.append(i['pos'][k, tuple((a, h[a]) for a in i['head'])])
                who += [nm]*len(i['v'])
                moved += [nm in swing_loose and any(o[a] != default[a] for a in i['own'])]*len(i['v'])
        return np.vstack(S), np.vstack(T), (np.array(who), np.array(moved))

    fitted, fit_report = {}, {}
    for g, members in groups.items():
        while members:
            F, worst = fit_group(g, members, samples, head_combos)
            bad = [nm for nm in members if worst[nm][0] > TOL or worst[nm][1] > TOL_SWUNG]
            if not bad:
                break
            log(f'  {g}: {", ".join(f"{n} {worst[n][0]:.2f}/{worst[n][1]:.2f}px" for n in bad)} -> baked under Body')
            for nm in bad:
                info[nm]['group'] = None
            members = [nm for nm in members if nm not in bad]
        if not members:
            continue
        groups[g] = members
        fitted[g] = F
        log(f"{g:7s} {F['rows']}x{F['cols']}: " + ', '.join(
            f'{n} {worst[n][0]:.2f}' + (f'/{worst[n][1]:.2f}' if n in swing_loose else '') for n in members) + ' px')
        fit_report[g] = {n: worst[n] for n in members}

    # ---- opacity-only meshes under Body -> shared opacity warps ----
    shows = {}
    if OPACITY_WARPS:
        for nm in names:
            i = info[nm]
            if i['group'] or i['head'] or not i['own']:
                continue
            ref = i['pos'][tuple((a, default[a]) for a in i['own']), ()]
            if max(np.abs(P-ref).max() for P in i['pos'].values()) > 1e-6:
                continue
            sig = (tuple(i['own']), tuple(i['ops'][tuple(o.items()), ()] for o in combos(i['own'])))
            shows.setdefault(sig, []).append(nm)
        for k, (sig, members) in enumerate(shows.items()):
            for nm in members:
                info[nm]['show'] = k
        log(f'opacity warps: {len(shows)} for {sum(len(m) for m in shows.values())} meshes')

    # ---- write ----
    to_model = lambda a: (np.asarray(a, float)-origin)/PPU
    pos_all = []

    def align():
        while len(pos_all) % 16:
            pos_all.append(0.)
        return len(pos_all)
    pids = [a for a, _, _ in PARAMS]
    band_axes = [[]]

    def band(axes):
        a = [pids.index(x) for x in axes]
        if a not in band_axes:
            band_axes.append(a)
        return band_axes.index(a)

    gnames = list(fitted)
    defs = [dict(id='Body', parent=-1, axes=body_axes, rows=body_rows, cols=body_cols,
                 quad=True, forms=[to_model(f) for f in body_forms])]
    for g in gnames:
        F = fitted[g]
        defs.append(dict(id='Warp_'+g, parent=0, axes=HEAD, rows=F['rows'], cols=F['cols'],
                         quad=QUAD and F['rows']*F['cols'] > 1,
                         forms=[uv_of(f, body_lines) for f in F['forms']]))
    # identity on the Body warp's own lattice: a coarser child warp would map
    # its meshes linearly between its corners and drop the Body grid's bend
    gu, gv = np.meshgrid(np.linspace(0, 1, body_cols+1), np.linspace(0, 1, body_rows+1))
    ident = np.stack([gu.ravel(), gv.ravel()], 1)
    show_base = len(defs)
    for k, ((axes, ops), members) in enumerate(shows.items()):
        defs.append(dict(id=f'Show_{k}', parent=0, axes=list(axes), rows=body_rows, cols=body_cols,
                         quad=defs[0]['quad'], forms=[ident]*len(ops), ops=list(ops)))
    nd = len(defs)
    for d in defs:
        d['band'] = band(d['axes'])
    wkop, wkpos = [], []
    for d in defs:
        d['begin'] = len(wkop)
        for k, f in enumerate(d['forms']):
            wkpos.append(align()); pos_all.extend(np.asarray(f).reshape(-1).tolist())
            wkop.append(float(d['ops'][k]) if 'ops' in d else 1.)

    uv_all, idx_all, uv_begin, idx_begin, vcounts, icounts = [], [], [], [], [], []
    bands, kf_begin, kf_counts, kf_op, kf_do, kf_pos, parents = [], [], [], [], [], [], []
    for mm in meshes:
        nm = mm['name']; i = info[nm]
        uv = np.asarray(mm['uvs'], float); tri = np.asarray(mm['triangles'], int)
        uv_begin.append(len(uv_all)); idx_begin.append(len(idx_all))
        uv_all.extend(uv.reshape(-1).tolist()); idx_all.extend(tri.reshape(-1).tolist())
        vcounts.append(len(i['v'])); icounts.append(tri.size)
        g = i['group']
        if 'show' in i:
            axes = []
            parents.append(show_base+i['show']); F = body_lines
        elif g:
            axes = i['own']+(i['head'] if i['op_head'] else [])
            parents.append(1+gnames.index(g)); F = fitted[g]
        else:
            axes = i['own']+i['head']
            parents.append(0); F = body_lines
        axes = [a for a in AXES[nm] if a in axes]      # keep the mesh's axis order
        bands.append(band(axes)); kf_begin.append(len(kf_op))
        forms = combos(axes)
        for p in forms:
            o = tuple((a, p.get(a, default[a])) for a in i['own'])
            h = tuple((a, p.get(a, 0.) if not g else 0.) for a in i['head'])
            hop = tuple((a, p.get(a, 0.)) for a in i['head'])
            kf_pos.append(align()); pos_all.extend(uv_of(i['pos'][o, h], F).reshape(-1).tolist())
            kf_op.append(1. if 'show' in i else i['ops'][o, hop]); kf_do.append(float(mm['draw_order']))
        kf_counts.append(len(forms))

    n = len(meshes); p = len(PARAMS)
    slot, bb = [], []
    for ax in band_axes:
        bb.append(len(slot)); slot.extend(ax)
    mask_idx, mask_begin, mask_count = [], [], []
    for nm in names:
        mask_begin.append(len(mask_idx))
        srcs = [names.index(s) for s in rig.masks.get(nm, []) if s in names]
        mask_idx.extend(srcs); mask_count.append(len(srcs))
    keys, key_off = [], []
    for _, vals, _ in PARAMS:
        key_off.append(len(keys)); keys.extend(vals)

    m = Moc3(); m.header.version = 2                   # V3_03
    m.canvas.pixels_per_unit = PPU
    m.canvas.origin_x, m.canvas.origin_y = origin
    m.canvas.canvas_width, m.canvas.canvas_height = float(W), float(H)
    c = m.counts
    for k, val in enumerate([0, nd, nd, 0, n, p, 0, len(wkop), 0, len(kf_op), len(pos_all), len(slot),
                             len(band_axes), p, len(keys), len(uv_all), len(idx_all), len(mask_idx), 1, n, 0, 0, 0]):
        c[k] = val
    for e in SECTION_LAYOUT:
        if e.elem_type != ElemType.RUNTIME:
            m[e.name] = [("" if e.elem_type == ElemType.STR64 else 0)]*c[e.count_idx]
    S = m
    S['deformer.ids'] = [d['id'] for d in defs]
    S['deformer.keyform_binding_band_indices'] = [d['band'] for d in defs]
    S['deformer.visibles'] = [True]*nd; S['deformer.enables'] = [True]*nd
    S['deformer.parent_part_indices'] = [-1]*nd
    S['deformer.parent_deformer_indices'] = [d['parent'] for d in defs]
    S['deformer.types'] = [0]*nd; S['deformer.specific_indices'] = list(range(nd))
    S['warp_deformer.keyform_binding_band_indices'] = [d['band'] for d in defs]
    S['warp_deformer.keyform_begin_indices'] = [d['begin'] for d in defs]
    S['warp_deformer.keyform_counts'] = [len(d['forms']) for d in defs]
    S['warp_deformer.vertex_counts'] = [(d['rows']+1)*(d['cols']+1) for d in defs]
    S['warp_deformer.rows'] = [d['rows'] for d in defs]; S['warp_deformer.cols'] = [d['cols'] for d in defs]
    S['warp_deformer_keyform.opacities'] = wkop
    S['warp_deformer_keyform.keyform_position_begin_indices'] = wkpos
    S['art_mesh.ids'] = names
    S['art_mesh.keyform_binding_band_indices'] = bands
    S['art_mesh.keyform_begin_indices'] = kf_begin; S['art_mesh.keyform_counts'] = kf_counts
    S['art_mesh.visibles'] = [True]*n; S['art_mesh.enables'] = [True]*n
    S['art_mesh.parent_part_indices'] = [-1]*n
    S['art_mesh.parent_deformer_indices'] = parents
    S['art_mesh.texture_indices'] = [0]*n; S['art_mesh.drawable_flags'] = [4]*n
    S['art_mesh.vertex_counts'] = vcounts; S['art_mesh.position_index_counts'] = icounts
    S['art_mesh.uv_begin_indices'] = uv_begin; S['art_mesh.position_index_begin_indices'] = idx_begin
    S['art_mesh.mask_begin_indices'] = mask_begin; S['art_mesh.mask_counts'] = mask_count
    S['drawable_mask.art_mesh_indices'] = mask_idx
    S['art_mesh_keyform.opacities'] = kf_op; S['art_mesh_keyform.draw_orders'] = kf_do
    S['art_mesh_keyform.keyform_position_begin_indices'] = kf_pos
    S['keyform_position.xys'] = pos_all; S['uv.xys'] = uv_all; S['position_index.indices'] = idx_all
    S['parameter.ids'] = pids
    S['parameter.min_values'] = [v[0] for _, v, _ in PARAMS]
    S['parameter.max_values'] = [v[-1] for _, v, _ in PARAMS]
    S['parameter.default_values'] = [d for _, _, d in PARAMS]
    S['parameter.repeats'] = [False]*p; S['parameter.decimal_places'] = [3]*p
    S['parameter.keyform_binding_begin_indices'] = list(range(p))
    S['parameter.keyform_binding_counts'] = [1]*p
    S['keyform_binding.keys_begin_indices'] = key_off
    S['keyform_binding.keys_counts'] = [len(v) for _, v, _ in PARAMS]
    S['keys.values'] = keys
    S['keyform_binding_index.indices'] = slot
    S['keyform_binding_band.begin_indices'] = bb
    S['keyform_binding_band.counts'] = [len(a) for a in band_axes]
    S['draw_order_group.object_begin_indices'] = [0]; S['draw_order_group.object_counts'] = [n]
    S['draw_order_group.object_total_counts'] = [n]; S['draw_order_group.min_draw_orders'] = [0]
    S['draw_order_group.max_draw_orders'] = [n-1]
    S['draw_order_group_object.types'] = [0]*n; S['draw_order_group_object.indices'] = list(range(n))
    S['draw_order_group_object.group_indices'] = [-1]*n
    m[ADDITIONAL_V303.name] = [bool(d['quad']) for d in defs]
    # every index Core follows must stay inside its table: an index past the
    # end reads runtime pointers and crashes only for some memory layouts
    nb = len(band_axes)
    assert len(bb) == nb and max(bands+[d['band'] for d in defs]) < nb
    kc = {a: len(v) for a, v, _ in PARAMS}
    for ax, cnt in zip([band_axes[b] for b in bands]+[band_axes[d['band']] for d in defs],
                       kf_counts+[len(d['forms']) for d in defs]):
        assert int(np.prod([kc[pids[a]] for a in ax])) == cnt
    ends = [kf_pos[b+k]+2*vcounts[j] for j, (b, cnt) in enumerate(zip(kf_begin, kf_counts)) for k in range(cnt)]
    ends += [wkpos[d['begin']+k]+2*(d['rows']+1)*(d['cols']+1) for d in defs for k in range(len(d['forms']))]
    assert max(ends) <= len(pos_all)
    assert max(kf_do) <= n-1 and min(kf_do) >= 0
    moc = out/f'{name}.moc3'
    m.to_file(moc)
    manifests(out, name, pids)
    (out/'deformers.json').write_text(json.dumps(dict(
        groups={g: groups[g] for g in gnames},
        baked_under_body=[nm for nm in names if not info[nm]['group'] and groups_of.get(nm) == 'head'],
        quad=QUAD, tol_px=TOL, tol_swung_px=TOL_SWUNG,
        fit_px={g: {n: [round(a, 3), round(b, 3)] for n, (a, b) in r.items()} for g, r in fit_report.items()}),
        ensure_ascii=False, indent=2)+'\n')
    log(f'wrote {moc} ({moc.stat().st_size/1e6:.1f} MB): {nd} warps, {len(kf_op)} mesh keyforms, '
        f'{len(pos_all)} floats')
    return moc


def manifests(out: Path, name: str, param_ids):
    (out/f'{name}.cdi3.json').write_text(json.dumps({
        "Version": 3,
        "Parameters": [{"Id": i, "GroupId": "", "Name": i} for i in param_ids],
        "ParameterGroups": [], "Parts": [],
    }, indent=2))
    (out/f'{name}.model3.json').write_text(json.dumps({
        "Version": 3,
        "FileReferences": {
            "Moc": f"{name}.moc3",
            "Textures": ["texture_00.png"],
            "DisplayInfo": f"{name}.cdi3.json",
        },
        "Groups": [
            {"Target": "Parameter", "Name": group, "Ids": ids}
            for group, ids in [("EyeBlink", ["ParamEyeLOpen", "ParamEyeROpen"]),
                               ("LipSync", ["ParamMouthOpenY"])]
            if all(i in param_ids for i in ids)
        ],
    }, indent=2))
