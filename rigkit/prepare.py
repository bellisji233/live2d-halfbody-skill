"""New character: art folder -> prepared drawables (meshes.json + texture atlas).

  python -m rigkit.prepare CHARACTER_JSON

Reads layout.json and parts/ from the art folder (every part a full-canvas
RGBA PNG, drawn bottom to top in layout order) and writes the prepared
folder that rigkit.build reads. Pixels are never changed: only crops,
meshes and atlas placement are decided here.

Meshes by role:
  head layers that bend with the surface (face, hair, ears, neck, blush)
      cells of the hair lattice, clipped to the layer: neighbouring layers
      share vertices along their seams, so they bend alike;
  eye lashes and the eye mask
      rows on the lid contours, so the lid closes along real rows;
  open mouth
      rows on the lip contours;
  small parts that move as one piece
      a regular grid;
  body and arm poses
      a grid every BODY_CELL px, so the body warp can bend them.

A brow the front hair covers (at rest or at any brow form) gets a copy
just above the highest hair layer that covers it, clipped by those hair
layers: it shows only where the hair is, moving with the brow (see
through_layers). The copy shares the brow's place in the atlas.
"""
from __future__ import annotations
from pathlib import Path
import json, sys
import numpy as np
import cv2
from PIL import Image

from .config import Character, through_name
from .context import Context
from .rig import Rig

ATLAS = 4096
PAD = 4            # px around each piece in the atlas (bilinear and mipmap bleed)
MARGIN = 3         # px of transparent margin kept around a layer's pixels
BODY_CELL = 24.    # px
THROUGH_MIN = 20   # px of a brow under a hair layer, at any brow form, for that layer to clip a copy


def dilate_color(rgba):
    """Spread the RGB of opaque pixels into transparent ones (alpha kept), so
    bilinear sampling never pulls stale colours into an edge."""
    a = rgba[:, :, 3]
    peak = int(a.max())
    if peak == 0:
        return rgba
    thr = 250 if peak >= 250 else max(1, int(peak*.75))
    solid = a >= thr
    if solid.all():
        return rgba
    _, labels = cv2.distanceTransformWithLabels((~solid).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(solid)
    out = rgba.copy()
    out[:, :, :3] = rgba[ys[labels-1], xs[labels-1], :3]
    return out


def pack(entries):
    """Shelf packing by height."""
    atlas = np.zeros((ATLAS, ATLAS, 4), np.uint8)
    x = y = shelf = 0
    placed = {}
    for e in sorted([e for e in entries if not e.get('copy_of')], key=lambda e: -e['rgba'].shape[0]):
        h, w = e['rgba'].shape[:2]
        if x+w+2*PAD > ATLAS:
            x = 0; y += shelf+2*PAD; shelf = 0
        if y+h+2*PAD > ATLAS:
            raise RuntimeError('atlas full: crop tighter or split the texture')
        atlas[y:y+h+2*PAD, x:x+w+2*PAD] = np.pad(e['rgba'], ((PAD, PAD), (PAD, PAD), (0, 0)), mode='edge')
        placed[e['name']] = (x+PAD, y+PAD)
        x += w+2*PAD
        shelf = max(shelf, h)
    return atlas, placed


def grid(w, h, nx, ny):
    gx, gy = np.meshgrid(np.linspace(0, w-1, nx), np.linspace(0, h-1, ny))
    tris = []
    for r in range(ny-1):
        for c in range(nx-1):
            a = r*nx+c
            tris.extend([[a, a+1, a+nx], [a+1, a+nx+1, a+nx]])
    return np.stack([gx.ravel(), gy.ravel()], 1), np.array(tris)


def lattice_mesh(lattice, crop):
    """Lattice cells clipped to the crop; vertices relative to the crop."""
    x0, y0, x1, y1 = crop
    lo = np.array([x0+.5, y0+.5]); hi = np.array([x1-.5, y1-.5])
    P = lattice.points
    vertices, triangles, lookup = [], [], {}
    for cell in P[lattice.cells]:
        if np.any(cell.max(0) < lo) or np.any(cell.min(0) > hi):
            continue
        polygon = list(cell)
        for axis, edge, keep in [(0, lo[0], 1), (0, hi[0], -1), (1, lo[1], 1), (1, hi[1], -1)]:
            result = []
            for a, b in zip(polygon, polygon[1:]+polygon[:1]):
                ina = (a[axis]-edge)*keep >= 0; inb = (b[axis]-edge)*keep >= 0
                if ina:
                    result.append(a)
                if ina != inb:
                    result.append(a+(b-a)*(edge-a[axis])/(b[axis]-a[axis]))
            polygon = result
            if not polygon:
                break
        if len(polygon) < 3:
            continue
        index = []
        for p in polygon:
            key = tuple(np.round(p, 7))
            if key not in lookup:
                lookup[key] = len(vertices); vertices.append(p-lo)
            index.append(lookup[key])
        for j in range(1, len(index)-1):
            tri = [index[0], index[j], index[j+1]]
            a, b, c = np.array(vertices)[tri]
            area = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
            if abs(area) > 1e-7:
                triangles.append(tri if area > 0 else tri[::-1])
    return np.asarray(vertices), np.asarray(triangles)


def through_layers(char, layers, log):
    """layers with a copy of each brow the front hair covers, just above the
    highest hair layer covering it; the copy's masks are those hair layers."""
    rig = Rig(char, [l['name'] for l in layers])
    if rig.through_opacity() <= 0:
        return layers
    W, H = char.canvas
    alpha = lambda l: np.asarray(Image.open(char.art/l['texture']).convert('RGBA'))[:, :, 3]
    out = list(layers)
    for i, b in enumerate(layers):
        L = rig.layers[b['name']]
        if L.role != 'brow' or not b.get('visible', True):
            continue
        ys, xs = np.nonzero(alpha(b) > 128)
        if not len(xs):
            continue
        pixels = np.stack([xs, ys], 1)+.5
        at = [np.floor(rig.apply(b['name'], pixels, {'ParamBrowForm': f})[0]).astype(int)
              for f in rig.keys['ParamBrowForm']]
        masks = []
        for h in layers[i+1:]:
            if rig.layers[h['name']].role != 'hair':
                continue
            hair = alpha(h) > 128
            under = max(int(hair[q[ok, 1], q[ok, 0]].sum()) for q in at
                        for ok in [(q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)])
            if under >= THROUGH_MIN:
                masks.append(h['name'])
        if masks:
            top = max(k for k, l in enumerate(out) if l['name'] in masks)
            out.insert(top+1, dict(b, name=through_name(b['name']), masks=masks, copy_of=b['name']))
            log(f'{b["name"]}: shows through {", ".join(masks)} at {rig.through_opacity():.0%}')
    return out


def prepare(char_path, out=None, log=print):
    char = Character(char_path)
    out = Path(out) if out else char.prepared
    out.mkdir(parents=True, exist_ok=True)
    W, H = char.canvas
    ctx = Context(char)
    layers = through_layers(char, list(char.layout['layers']), log)
    names = [l['name'] for l in layers]
    for side in 'LR':
        if f'Eye_White_{side}' in names:
            # clips the iris and eye white to the eye opening
            layers.append(dict(name=f'Eye_Mask_{side}', texture=str(char.part(f'Eye_White_{side}')),
                               visible=False, group='head', mask=True))
    table = char.layers([l['name'] for l in layers])
    entries = []
    for i, l in enumerate(layers):
        L = table[l['name']]
        rgba = np.asarray(Image.open(char.art/l['texture']).convert('RGBA')).copy()
        if l.get('mask'):
            rgba[:, :, 3] = np.where(rgba[:, :, 3] > 128, 255, 0); rgba[:, :, :3] = 255
        ys, xs = np.where(rgba[:, :, 3] > 0)
        if not len(xs):
            log(f'skip {l["name"]}: empty'); continue
        x0 = max(0, int(xs.min())-MARGIN); y0 = max(0, int(ys.min())-MARGIN)
        x1 = min(W, int(xs.max())+MARGIN+1); y1 = min(H, int(ys.max())+MARGIN+1)
        if L.role in ('eye_lash', 'eye_mask'):
            # lash and mask share one window: the lid rows of both line up
            x0, y0, x1, y1 = ctx.eye_window(L.side)
        crop = rgba[y0:y1, x0:x1].copy()
        padded = dilate_color(crop)
        crop[crop[:, :, 3] == 0, :3] = padded[crop[:, :, 3] == 0, :3]
        h, w = crop.shape[:2]
        if L.role in ('eye_lash', 'eye_mask'):
            verts, tris = _lid_mesh(ctx, L, (x0, y0, x1, y1))
        elif L.role == 'mouth' and L.variant == 'Open':
            verts, tris = _mouth_mesh(ctx, char, (x0, y0, x1, y1))
        elif L.group in ('body', 'pose'):
            verts, tris = grid(w, h, max(2, int(np.ceil((w-1)/BODY_CELL))+1), max(2, int(np.ceil((h-1)/BODY_CELL))+1))
        elif L.role in ('face', 'hair', 'neck', 'cheek'):
            verts, tris = lattice_mesh(ctx.lattice, (x0, y0, x1, y1))
        else:
            nx, ny = {'brow': (9, 4), 'eye_white': (13, 11), 'iris': (13, 11), 'eye_lower': (13, 11)}.get(L.role, (9, 7))
            verts, tris = grid(w, h, nx, ny)
        entries.append(dict(name=l['name'], index=i, rgba=crop, verts=verts, tris=np.asarray(tris),
                            crop=(x0, y0, x1, y1), visible=l.get('visible', True),
                            copy_of=l.get('copy_of'), masks=l.get('masks')))
    atlas, placed = pack(entries)
    meshes = []
    for k, e in enumerate(entries):
        ax, ay = placed[e['copy_of'] or e['name']]; x0, y0, x1, y1 = e['crop']; v = e['verts']
        meshes.append(dict(name=e['name'], draw_order=k, visible=e['visible'],
                           vertices=(v+[x0+.5, y0+.5]).tolist(), uvs=((v+[ax+.5, ay+.5])/ATLAS).tolist(),
                           triangles=e['tris'].tolist(), crop=list(e['crop'])))
        if e['masks']:
            meshes[-1]['masks'] = e['masks']
    (out/'meshes.json').write_text(json.dumps(dict(canvas=[W, H], atlas=ATLAS, meshes=meshes)))
    Image.fromarray(atlas).save(out/'texture_00.png')
    log(f'{len(meshes)} drawables -> {out}')
    return out


def _lid_mesh(ctx, L, crop):
    """Mesh rows ON the lid: a rectangular grid straddling the lid hinge
    leaves a false opening at the closed key."""
    x0, y0, x1, y1 = crop
    nx, ny = (33 if L.role == 'eye_lash' else 13), 9
    xx = np.linspace(x0+.5, x1-.5, nx)
    up, lo, _ = ctx.eye_contours(xx, L.side)
    rows = [np.full(nx, y0+.5), up-34, up-30, up, up+(lo-up)/3, up+(lo-up)*2/3, lo, lo+5, np.full(nx, y1-.5)]
    rows = np.clip(np.stack(rows), y0+.5, y1-.5)
    gx = np.tile(xx-x0-.5, (ny, 1)); gy = rows-y0-.5
    _, tris = grid(2, 2, nx, ny)
    return np.stack([gx.ravel(), gy.ravel()], 1), tris


def _mouth_mesh(ctx, char, crop):
    x0, y0, x1, y1 = crop
    nx, ny = 17, 9
    xx = np.linspace(x0+.5, x1-.5, nx)
    up, lo, _ = ctx.mouth_contours(xx)
    bottom = char['mouth']['open_band'][1]
    rows = [np.full(nx, y0+.5), up-2, up, up+(lo-up)/3, up+(lo-up)*2/3, lo, lo+2, np.full(nx, bottom), np.full(nx, y1-.5)]
    rows = np.clip(np.stack(rows), y0+.5, y1-.5)
    gx = np.tile(xx-x0-.5, (ny, 1)); gy = rows-y0-.5
    _, tris = grid(2, 2, nx, ny)
    return np.stack([gx.ravel(), gy.ravel()], 1), tris


if __name__ == '__main__':
    prepare(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
