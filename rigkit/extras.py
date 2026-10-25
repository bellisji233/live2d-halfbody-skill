"""Gesture and blush layers, made from edits of the finished model at rest.

  python -m rigkit.extras ART_FOLDER LAYERS

LAYERS is the finished split (rigkit.seethrough split, or rigkit.split alone).
The first run writes ART_FOLDER/R0-rest.png: the layers at rest, on the art's
canvas. rigkit.variants then makes the edits of it (EDITS below). Each run
after that turns the edits present into layers and adds them to
LAYERS/layout.json, hidden:

  Pose_Wave, Pose_Explain, Pose_Point   the whole body in that pose; the model
      shows it in place of the body group while the gesture is on. It is the
      body where the edit left it alone and the edit where it changed (the arm,
      its sleeve, the page where the arm hung), so only the arm is redrawn.
  Cheek_Flush_L, Cheek_Flush_R          a deeper blush laid over each cheek.

The edits are made from the model at rest, not from the art, so they are drawn
the way the layers look (See-through's softer rendering when its layers are
used), and switching a gesture on does not change the style of the body.
A check sheet (手势和脸红.png) shows each edit beside the model with it on.
When the layers have changed since R0-rest was written (the clothes
reordered), a run measures what each edit changed against R0-rest there, and
takes the rest of the pose from the layers as they are now: no new edits.
"""
from pathlib import Path
import argparse, json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .split import WHITE, load, align, match_colours, figure_alpha, dilate, erode, keep_large, fill_holes, nearest_fill, components

REST = 'R0-rest'
POSES = {'P1-wave': 'Pose_Wave', 'P2-explain': 'Pose_Explain', 'P3-point': 'Pose_Point'}
BLUSH = 'F1-blush'
EDITS = [BLUSH, *POSES]


def read(folder):
    doc = json.loads((folder/'layout.json').read_text())
    L = {}
    for l in doc['layers']:
        a = np.asarray(Image.open(folder/l['texture']).convert('RGBA')).astype(np.float32)
        L[l['name']] = (a[..., :3], a[..., 3]/255.)
    return doc, L


def over(names, L, under=None):
    """Colour and coverage of the named layers stacked bottom to top."""
    rgb, a = next(iter(L.values()))
    out = np.full(rgb.shape, WHITE, np.float32) if under is None else under.copy()
    cover = np.zeros(a.shape, np.float32)
    for n in names:
        rgb, a = L[n]
        out = a[..., None]*rgb+(1-a[..., None])*out
        cover = cover+a*(1-cover)
    return out, cover


def visible(doc, skip=()):
    return [l['name'] for l in doc['layers'] if l.get('visible', True) and l['name'] not in skip]


def flat(names, L):
    """Several layers as one: premultiplied colour over nothing."""
    rgb, a = next(iter(L.values()))
    P, A = np.zeros(rgb.shape, np.float32), np.zeros(a.shape, np.float32)
    for n in names:
        rgb, a = L[n]
        P = a[..., None]*rgb+(1-a[..., None])*P
        A = a+(1-a)*A
    return P/np.maximum(A, 1e-4)[..., None], A


def edit(path, rest, pad_top):
    """An edit of R0-rest on the layers' canvas, shifted and colour-matched onto it."""
    v = load(path, pad_top, (rest.shape[1], rest.shape[0]-pad_top))
    v, shift, _ = align(v, rest)
    v, _ = match_colours(v, rest)
    return v, float(np.hypot(*shift))


def pose_layer(v, rest, figure, body, allowed, reach=None, hair_holes=None):
    """The body where the edit did not change it, the edit where it did: a
    colour change, or the figure moving over the page (a pale palm is close
    to the page's white). reach: where else the arm may go (over long hair
    behind the body); a change there counts only where it carries on from
    the change inside allowed."""
    d = cv2.GaussianBlur(np.abs(v-rest).max(2), (0, 0), 3)        # line jitter averages out, a redrawn arm does not
    # page white the fill cannot reach (closed in by a raised arm and the
    # hair) is still page where it was page at rest and the edit did not draw
    # over it (a white sleeve's flat inside is within what it redrew)
    paper = (v.min(2) > 246) & (np.ptp(v, 2) < 6)
    drawn = fill_holes(keep_large(d > 25, 3000), 20000)
    page = paper & (figure < .5) & ~drawn
    if hair_holes is not None:
        # the curls of the hair behind the body, uncovered as an arm moves off them
        # where an arm moved off: small gaps of page white the edit left between
        # strands of hair (a third or more of what is round them dark hair; a white cuff has its
        # sleeve round it, kept)
        holes, behind = hair_holes
        page |= paper & holes
        dark = v.mean(2) < 120
        for c in components(paper & (body[1] > .5), 1):
            ring = dilate(c, 3) & ~c
            if c.sum() < 800 and dark[ring].mean() > .3:
                page |= c
    fa = figure_alpha(v, fixed=True, page=page)   # fixed: a pale garment is not page
    moved = cv2.GaussianBlur(np.abs(fa-figure), (0, 0), 3)
    raw = (d > 25) | (moved > .5)
    changed = fill_holes(keep_large(raw & allowed, 3000), 20000) & allowed
    if reach is not None:
        wide = fill_holes(keep_large(raw & reach, 3000), 20000) & reach
        _, lab = cv2.connectedComponents(wide.astype(np.uint8))
        hit = np.unique(lab[changed & wide])
        changed |= np.isin(lab, hit[hit > 0])
        allowed = reach
    w = cv2.GaussianBlur(dilate(changed, 4).astype(np.float32), (0, 0), 2)*allowed
    fv = np.clip((v-(1-fa)[..., None]*WHITE)/np.maximum(fa, .05)[..., None], 0, 255)
    fv = nearest_fill(fv, fa > .98, (fa > 0) & (fa < .5))           # a faint edge takes the colour just inside
    b_rgb, b_a = body
    A = w*fa+(1-w)*b_a
    P = (w*fa)[..., None]*fv+((1-w)*b_a)[..., None]*b_rgb
    rgb = P/np.maximum(A, 1e-4)[..., None]
    return rgb, A.astype(np.float32), changed


def flush_layers(v, rest, face, face_shows, centre_x):
    """A flat deep-blush colour over each cheek, its opacity an elliptical
    falloff fitted to how much pinker the edit is where the cheek shows: soft
    all round, and it carries on under the hair and the lower lashes instead
    of stopping at their edge."""
    d = v-rest
    m = keep_large((cv2.GaussianBlur(np.abs(d).max(2), (0, 0), 2) > 10) & face_shows, 300)
    if not m.any():
        return None
    strength = np.abs(d).max(2)*m
    C = np.median(v[strength >= np.percentile(strength[m], 90)], axis=0)
    den = ((C-rest)**2).sum(2)
    solved = np.clip((d*(C-rest)).sum(2)/np.maximum(den, 400.), 0, 1)*dilate(m, 6)*face_shows
    H, W = solved.shape
    ys, xs = np.mgrid[:H, :W].astype(np.float32)
    out = {}
    for side, half in (('L', xs < centre_x), ('R', xs >= centre_x)):
        w = solved*half
        if w.sum() < 50:
            continue
        mx, my = (w*xs).sum()/w.sum(), (w*ys).sum()/w.sum()
        dx, dy = xs-mx, ys-my
        cov = np.array([[(w*dx*dx).sum(), (w*dx*dy).sum()], [(w*dx*dy).sum(), (w*dy*dy).sum()]])/w.sum()
        inv = np.linalg.inv(cov)
        g = np.exp(-.5*(inv[0, 0]*dx*dx+2*inv[0, 1]*dx*dy+inv[1, 1]*dy*dy))
        seen = face_shows & half
        peak = (solved*g)[seen].sum()/max((g*g)[seen].sum(), 1e-6)
        a = np.clip(peak*g, 0, 1)*face
        out[f'Cheek_Flush_{side}'] = (np.broadcast_to(C, v.shape).astype(np.float32), np.where(a < .02, 0., a).astype(np.float32))
    return out, C


def _centre_x(L, names):
    for n in names:
        if n in L and (L[n][1] > .5).any():
            return float(np.nonzero(L[n][1] > .5)[1].mean())
    return None


def save(folder, doc, name, group, rgb, a, after):
    rgba = np.dstack([np.clip(rgb, 0, 255), np.clip(a, 0, 1)*255]).round().astype(np.uint8)
    Image.fromarray(rgba).save(folder/'parts'/f'{name}.png')
    layers = [l for l in doc['layers'] if l['name'] != name]
    i = max(k for k, l in enumerate(layers) if l['name'] in after)+1
    layers.insert(i, dict(name=name, texture=f'parts/{name}.png', group=group, visible=False))
    doc['layers'] = layers


def run(art, folder, log=print):
    art, folder = Path(art), Path(folder)
    doc, L = read(folder)
    W, H = doc['width'], doc['height']
    pad_top = H-Image.open(art/'A0-base.png').size[1]
    base = [n for n in visible(doc) if not n.startswith(('Pose_', 'Cheek_Flush_'))]
    rest, figure = over(base, L)
    rest_png = art/f'{REST}.png'
    if not rest_png.exists():
        Image.fromarray(np.clip(rest[pad_top:], 0, 255).round().astype(np.uint8)).save(rest_png)
        log(f'已写出 {rest_png.name}（图层静止合成）。下一步：python -m rigkit.variants {art} {" ".join(EDITS)}')
        return
    old = load(rest_png, pad_top)
    off = np.abs(old-rest).max(2)[pad_top:] > 8
    # the layers changed since the edits were made (the clothes reordered): what
    # an edit changed is measured against R0-rest, which it was made from; the
    # rest of the pose is the body as it is now. Against the new rest, the
    # parts that changed would count as redrawn by every edit, and a gesture
    # would bring the old drawing of them back
    ref = rest
    if off.sum() > 500:
        changed_since = np.zeros(rest.shape[:2], bool)
        changed_since[pad_top:] = off
        changed_since = dilate(changed_since, 4)
        ref = np.where(changed_since[..., None], old, rest)
        log(f'  图层和 {REST}.png 有 {int(off.sum())} px 不同：那里变体改了哪里按 {REST}.png 算，画面用现在的图层')

    group = {l['name']: l['group'] for l in doc['layers']}
    # a dangle stays on top of every pose with its own swing
    body = [n for n in base if group[n] == 'body' and not n.startswith('Dangle_')]
    first, last = base.index(body[0]), base.index(body[-1])
    _, below = over(base[:first], L)
    _, above = over(base[last+1:], L)
    body_flat = flat(body, L)
    allowed = (above < .5) & ((body_flat[1] > .5) | (below < .5))      # at rest the body or the page shows here
    reach = None
    hanging = [n for n in L if n.startswith(('Hair_Back', 'Hair_Tail_'))]
    if hanging and 'Neck' in L:
        rows = lambda n: np.nonzero((L[n][1] > .5).any(1))[0]
        if max(rows(n).max() for n in hanging) > rows('Neck').max():
            reach = above < .5                                          # long hair: a raised arm goes over it
    hair_holes = (fill_holes(below > .5, 20000) & (below < .1), below > .5)   # page closed in by, and hair behind the body
    made, checks = [], []
    for edit_name, layer in POSES.items():
        if not (art/f'{edit_name}.png').exists():
            continue
        v, drift = edit(art/f'{edit_name}.png', rest, pad_top)
        rgb, a, changed = pose_layer(v, ref, figure, body_flat, allowed, reach, hair_holes)
        save(folder, doc, layer, 'pose', rgb, a, after=body+[p for p in POSES.values()])
        L[layer] = (rgb, a)
        model, _ = over(base[:first]+[layer]+base[last+1:], L)
        checks.append((edit_name, v, model))
        made.append(layer)
        log(f'  {layer}：对位 {drift:.1f} px，重画 {changed.mean()*100:.1f}% 画面')
    if (art/f'{BLUSH}.png').exists() and 'Face_Skin' in L:
        v, drift = edit(art/f'{BLUSH}.png', rest, pad_top)
        i = base.index('Face_Skin')
        _, on_face = over(base[i+1:], L)
        face_shows = erode((L['Face_Skin'][1] > .5) & (on_face < .02), 2)
        cx = _centre_x(L, ['Nose', 'Mouth_Closed', 'Face_Skin'])
        res = flush_layers(v, ref, L['Face_Skin'][1], face_shows, cx)
        if res is None:
            log(f'  {BLUSH}：脸上没有找到变化，没有生成脸红层')
        else:
            layers, C = res
            for n, (rgb, a) in layers.items():
                save(folder, doc, n, 'head', rgb, a, after=['Face_Skin', 'Cheek_Flush_L'])
                L[n] = (rgb, a)
                made.append(n)
            model, _ = over(base[:i+1]+list(layers)+base[i+1:], L)
            checks.append((BLUSH, v, model))
            log(f'  脸红：对位 {drift:.1f} px，颜色 {tuple(int(c) for c in C)}')
    (folder/'layout.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2)+'\n')
    missing = [e for e in EDITS if not (art/f'{e}.png').exists()]
    if missing:
        log(f'  还没有：{" ".join(missing)}（python -m rigkit.variants {art} {" ".join(missing)}）')
    if checks:
        check_sheet(checks, rest, folder/'手势和脸红.png')
    log(f'已加入 {folder/"layout.json"}：{"、".join(made) or "无"}')
    return made


def check_sheet(checks, rest, path, cell=520):
    """Per edit: the model at rest | the edit | the model with it on."""
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', 22)
    except OSError:
        font = None
    h = int(cell*rest.shape[0]/rest.shape[1])
    out = Image.new('RGB', (3*cell, len(checks)*(h+34)), 'white')
    dr = ImageDraw.Draw(out)
    for r, (name, v, model) in enumerate(checks):
        y = r*(h+34)
        for k, (img, cap) in enumerate([(rest, '静止'), (v, f'{name}（变体）'), (model, '模型里')]):
            im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((cell, h), Image.LANCZOS)
            out.paste(im, (k*cell, y+34)); dr.text((k*cell+8, y+6), cap, fill=(80, 40, 40), font=font)
    out.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('art'); ap.add_argument('layers')
    a = ap.parse_args()
    run(a.art, a.layers)


if __name__ == '__main__':
    main()
