"""Layers from See-through (shitagaki-lab/see-through), joined to our own split.

  python -m rigkit.seethrough fetch IMAGE ART_FOLDER/seethrough [--site auto|hf|modelscope] [--res N]
  python -m rigkit.seethrough split ART_FOLDER BASE_LAYERS OUT [--pad-top 90]

fetch sends IMAGE to a public See-through demo and writes one full-canvas PNG
per layer of the PSD it returns. It needs gradio_client and psd-tools. Ask
the owner of the art before sending it. Two demos run the same model; --site
auto (the default) tries Hugging Face first and ModelScope when that fails:
  Hugging Face ZeroGPU, a few minutes a run. Without an account the day's
               GPU time is short of one run; a free account's Read token (in
               HF_TOKEN or saved by `hf auth login`) gives about 1-2 runs a day
  ModelScope   free; needs a ModelScope account's access token (in
               MODELSCOPE_API_TOKEN or ~/.cache/modelscope/token); runs queue,
               so a run can take half an hour
Each runs at the highest resolution it offers (Hugging Face 1280, ModelScope
1536) unless --res asks for less: 1536 draws the hair outline and the eyes
closer to the art.
Given a .psd instead of an image (made on a demo's web page, in ComfyUI or
on one's own GPU), it only unpacks that PSD.

split: ART_FOLDER holds A0-base.png and seethrough/. BASE_LAYERS is our own
split (rigkit.split): See-through has no expressions, cuts the front hair as
one piece and paints poorly under the fringe and inside the back hair, so
those come from it. Every part See-through draws is used as it draws it,
colour included: no part then carries a pixel of its neighbour, which is what
makes the model clean in motion; at rest it is See-through's rendering of the
art, a little softer than the art. Where it goes wrong, ours: a part it paints
over another one (a hat band across the face) is cut back, an item it files
under the wrong part (hair clips with the hat) goes to ours, one hidden under
the other half of the figure and apart from that part (a beret's pompom with
the blouse) is dropped, a part it leaves out or gets wrong over a wide area
(trousers) and back hair it leaves bare under the locks come from our split.
"""
from pathlib import Path
import argparse, json
import numpy as np
import cv2
from PIL import Image

from .split import (Stack, load, components, _u8, dilate, erode, keep_large, fill_holes, hue_distance, rest_image, sheet,
                    figure_alpha)

ST_PARTS = {                      # our layer -> See-through layers, bottom first
    'Hair_Back': ['back hair'],
    'Neck': ['neck'],
    'Body': ['bottomwear', 'topwear', 'handwear-r', 'handwear-l', 'neckwear', 'objects'],
    'Face_Skin': ['ears-r', 'ears-l', 'face', 'earwear'],  # ears under the face: the cheek line stays whole
    'Hat': ['headwear'],
}
LOCKS = ['Hair_Side_R', 'Hair_Side_L', 'Hair_Bang']   # split out of See-through's one front hair
# eyes and brows: See-through's -r is the character's right, our _L (screen left)
FEATURES = {'Brow_Neutral_L': 'eyebrow-r', 'Brow_Neutral_R': 'eyebrow-l',
            'Eye_White_L': 'eyewhite-r', 'Eye_White_R': 'eyewhite-l',
            'Iris_L': 'irides-r', 'Iris_R': 'irides-l'}
LASHES = {'L': 'eyelash-r', 'R': 'eyelash-l'}        # cut into our lash, crease and lower lash
SPILL, SPILL_AREA = 8, 200    # px beyond our part (or what hides it) that See-through may run on; least area cut
MISS, MISS_WIDTH, MISS_AREA = 60, 8, 300   # off from the art by this much, this wide (px radius) and large: ours


def st_layer(folder, name, size, pad_top):
    im = Image.open(Path(folder)/f'{name}.png').convert('RGBA').resize(size, Image.LANCZOS)
    a = np.asarray(im).astype(np.float32)
    rgb, al = a[..., :3], np.clip(a[..., 3]/255., 0, 1)
    if pad_top:
        rgb = np.concatenate([np.zeros((pad_top,)+rgb.shape[1:], np.float32), rgb], 0)
        al = np.concatenate([np.zeros((pad_top, al.shape[1]), np.float32), al], 0)
    al = np.where(al < .02, 0., al)
    al[:, :2] = al[:, -2:] = 0; al[-2:] = 0                          # resampling fringe on the canvas border
    al = np.where(keep_large(al > .02, 30), al, 0.)               # specks
    return rgb, al.astype(np.float32)


def over(layers):
    """Composite (rgb, alpha) pairs, bottom first."""
    rgb = np.zeros(layers[0][0].shape, np.float32); a = np.zeros(layers[0][1].shape, np.float32)
    for r, al in layers:
        rgb = al[..., None]*r+(1-al[..., None])*rgb
        a = al+(1-al)*a
    return np.where(a[..., None] > 1e-3, rgb/np.maximum(a, 1e-3)[..., None]*1., rgb), a


def offset_field(diff, known, sigma=(6., 24., 80.)):
    """Smooth version of diff, known only on `known`, spread everywhere: the
    finest scale where there is enough support, coarser ones further out."""
    w = known.astype(np.float32)
    out = np.zeros(diff.shape, np.float32); got = np.zeros(w.shape, np.float32)
    for s in sigma:
        ws = cv2.GaussianBlur(w, (0, 0), s)
        num = np.dstack([cv2.GaussianBlur(diff[..., c]*w, (0, 0), s) for c in range(3)])
        est = num/np.maximum(ws, 1e-6)[..., None]
        conf = np.clip(ws/.2, 0, 1)*(1-got)
        out += conf[..., None]*est; got += conf
    if known.any():
        out += (1-got)[..., None]*np.median(diff[known], 0)
    return out


def st_shapes(S, R, H, W, over_locks=None):
    """Our layers cut from See-through's: the big parts, the front hair split into
    our locks, and the eyes and brows (shape and See-through's colour).
    over_locks: what our split draws above all the locks (the hat)."""
    shapes = {k: over([S[n] for n in v if n in S]) for k, v in ST_PARTS.items()}
    if 'Neck' in R and 'Body' in R:
        # See-through's clothes run on over the neck seen in the collar opening
        # (it puts the neck under them); our body has the opening, as our split had it
        opening = (R['Neck'][1] > .5) & (R['Body'][1] < .5)
        hole = cv2.GaussianBlur(opening.astype(np.float32), (0, 0), .8)
        rgb, a = shapes['Body']
        shapes['Body'] = (rgb, (a*(1-hole)).astype(np.float32))
    front_rgb, front_a = S['front hair']

    # the front hair is cut into the locks along our split; pixels none of our
    # locks claims go to the nearest one
    lab = np.full((H, W), -1, np.int32)
    for i, n in enumerate(LOCKS):                                  # later locks lie on top
        lab[R[n][1] > .5] = i
    if (lab >= 0).any():
        _, idx = cv2.distanceTransformWithLabels((lab < 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(lab >= 0)
        near = lab[ys[idx-1], xs[idx-1]]
        lab = np.where(lab >= 0, lab, near)
    owns = []
    for i, n in enumerate(LOCKS):
        own = (lab == i).astype(np.float32)
        own = np.maximum(cv2.GaussianBlur(own, (0, 0), .6), own)   # lock-to-lock seams: one soft pixel over the neighbour
        own = np.maximum(own, dilate(lab == i, 3) & (lab > i))     # and the lower lock runs on under the upper one
        owns.append(front_a*own)
    for i, n in enumerate(LOCKS):
        upper = np.zeros((H, W), np.float32)
        for j in range(i+1, len(LOCKS)):
            upper = np.maximum(upper, owns[j])
        if over_locks is not None:
            upper = np.maximum(upper, (over_locks > .5).astype(np.float32))
        hidden = R[n][1]*np.clip(upper, 0, 1)                      # under a lock above (as cut here) or the hat: our own split
        a = np.maximum(owns[i], hidden)
        shapes[n] = (front_rgb, a.astype(np.float32))


    # eyes and brows as See-through draws them: each a whole shape (the brow
    # runs on under the hair, the iris and eye white under the lash), so nothing
    # of the face or hair is cut into them
    fshapes = {n: S[s] for n, s in FEATURES.items() if n in R and s in S}
    for side, s in LASHES.items():
        parts = [p for p in (f'Eye_Open_{side}', f'Eye_Crease_{side}', f'Eye_Lower_{side}') if p in R]
        if s not in S or f'Eye_Open_{side}' not in parts:
            continue
        rgb, a = S[s]
        # crease and lash along our split; loose bits below the eye are the lower lash
        lab = np.zeros((H, W), np.int32)
        for i, p in enumerate(parts):
            if not p.startswith('Eye_Lower'):
                lab[R[p][1] > .3] = i+1
        _, idx = cv2.distanceTransformWithLabels((lab == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(lab > 0)
        hard = np.where(lab > 0, lab, lab[ys[idx-1], xs[idx-1]])
        ink = a > .2
        n_c, comp, stats, cent = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
        white = fshapes.get(f'Eye_White_{side}', (None, R[f'Eye_White_{side}'][1]))[1] > .5
        mid = np.nonzero(white)[0].mean()
        if f'Eye_Lower_{side}' in parts and n_c > 2:
            main = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            low = [k for k in range(1, n_c) if k != main and cent[k][1] > mid]
            # a lower lash stroke the art has; a dot only See-through drew goes
            drawn = dilate(R[f'Eye_Lower_{side}'][1] > .3, 6)
            stray = [k for k in low if not drawn[comp == k].any()]
            if stray:
                a = a*(1-dilate(np.isin(comp, stray), 2))
                low = [k for k in low if k not in stray]
            hard = np.where(np.isin(comp, low), parts.index(f'Eye_Lower_{side}')+1, hard)
        # every pixel of the lash layer (its soft fringe too) goes with the nearest ink
        seed = np.where(ink, hard, 0)
        _, idx = cv2.distanceTransformWithLabels((seed == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(seed > 0)
        owner = np.where(seed > 0, seed, seed[ys[idx-1], xs[idx-1]])
        for i, p in enumerate(parts):
            own = (owner == i+1).astype(np.float32)
            own = np.maximum(cv2.GaussianBlur(own, (0, 0), .6), own)
            fshapes[p] = (rgb, (a*own).astype(np.float32))
    return shapes, fshapes


def plan_as_drawn(art, base_layers, out, pad_top=0, log=print):
    """See-through's layers as it draws them, colour included: nothing of the
    art is fitted into them, so no part carries a pixel of its neighbour. Ours
    only where See-through has nothing (expressions; the closed mouth and nose
    too, which it draws poorly) and where its hidden fill is poor (the forehead under the fringe: V5;
    the inside of the back hair: V8), shifted to meet its colours."""
    art, base_layers, out = Path(art), Path(base_layers), Path(out)
    st = art/'seethrough'
    B = load(art/'A0-base.png', pad_top)
    H, W = B.shape[:2]
    S = {p.stem: st_layer(st, p.stem, (W, H-pad_top), pad_top) for p in st.glob('*.png')}
    used = {s for v in ST_PARTS.values() for s in v} | set(FEATURES.values()) | set(LASHES.values()) \
        | {'front hair', 'nose', 'mouth'}
    if set(S)-used:
        log(f'  没有用到的 See-through 图层：{"、".join(sorted(set(S)-used))}')
    layout = json.loads((base_layers/'layout.json').read_text())['layers']
    R = {}
    for l in layout:
        a = np.asarray(Image.open(base_layers/l['texture']).convert('RGBA')).astype(np.float32)
        R[l['name']] = (a[..., :3], a[..., 3]/255.)
    names = [l['name'] for l in layout]
    top_lock = max(names.index(n) for n in LOCKS if n in names)
    # the hat as both splits have it: See-through's alone runs on over the face
    hat = [np.minimum(R[n][1], over([S[x] for x in ST_PARTS[n] if x in S])[1]) for n in names[top_lock+1:]
           if n in ST_PARTS and any(x in S for x in ST_PARTS[n])]
    shapes, fshapes = st_shapes(S, R, H, W, np.maximum.reduce(hat) if hat else None)
    # the closed mouth and the nose are ours, as drawn in the art: See-through
    # draws them thinner and lopsided, with a faint haze over the whole canvas
    # (so a mesh too coarse for the mouth form), and the other mouths are ours.
    # The lash runs under the hair as a whole: a tip kept above it shows cut
    # off from the eye when the head turns towards that side
    for side in LASHES:
        tips = f'Eye_Open_Over_{side}'
        if tips in R:
            fshapes[tips] = (R[tips][0], np.zeros((H, W), np.float32))
    shapes.update(fshapes)
    vis = {l['name']: l['visible'] for l in layout}
    order = [l['name'] for l in layout][::-1]
    group = {l['name']: l['group'] for l in layout}
    shown = np.full((H, W), -1, np.int32)               # our layer that shows at rest
    for i, n in enumerate(order[::-1]):
        if vis[n]:
            shown[R[n][1] > .5] = len(order)-1-i
    under_locks = np.maximum.reduce([R[n][1] for n in LOCKS if n in R]) > .5
    cover = np.zeros((H, W), np.float32); covers = {}; moved = []
    gcover = {}                                          # cover by each group's layers
    for n in order:
        covers[n] = cover.copy()
        if n in ST_PARTS and n in R:
            # a piece apart from what our split has of the part, hidden at rest
            # under the other half of the figure, is an item filed with the
            # wrong part (the beret's pompom with the blouse): left in, it
            # stays behind when the head moves off it
            other = 1-np.prod([1-c for g, c in gcover.items() if g != group[n]] or [np.ones((H, W))], 0)
            own = dilate(R[n][1] > .1, SPILL)
            rgb, a = shapes[n]
            loose = np.zeros((H, W), bool)
            for c in components(a > .5, SPILL_AREA):
                if not (c & own).any() and (other[c] > .5).mean() > .9:
                    loose |= c
            if loose.any():
                shapes[n] = (rgb, (a*(1-cv2.GaussianBlur(dilate(loose, 3).astype(np.float32), (0, 0), 1.))).astype(np.float32))
                log(f'  {n}：去掉藏在别的部件下、和本部件不相连的 {int(loose.sum())} px')
            # See-through also paints what it guesses is hidden. Kept where our
            # split has the part or a layer above hides it; elsewhere it would
            # show at rest over what the art shows there (a hat band painted
            # across the face). What it paints there as the art has it is an
            # item filed under the wrong part (hair clips with the hat): it goes
            # to our layer that shows there. The locks and features are cut
            # along our split already.
            keep = dilate((R[n][1] > .1) | (cover > .5), SPILL)
            rgb, a = shapes[n]
            stray = (a > .5) & ~keep
            if stray.sum() > SPILL_AREA:
                same = stray & (np.abs(rgb-B).max(2) < 50) & (shown >= 0)
                # each item whole to the layer most of it lies on: a clip split
                # between two locks would come apart when they swing
                to = {}
                for c in components(same, 200):               # whole items, not specks that match by chance
                    ks, counts = np.unique(shown[c], return_counts=True)
                    k = int(ks[np.argmax(counts)])
                    if group[order[k]] != group[n]:
                        # our split puts it with the other half of the figure
                        # (hair over a sleeve as clothes): See-through's reading
                        # stays, or it would not move with the rest of its part
                        keep |= c
                        continue
                    to[k] = to.get(k, np.zeros((H, W), bool)) | c
                stray &= ~keep
                for k, mk in sorted(to.items()):
                    moved.append((order[k], rgb, a*cv2.GaussianBlur(mk.astype(np.float32), (0, 0), 1.)))
                    log(f'  {n}：画在 {order[k]} 位置上的 {int(mk.sum())} px 归到 {order[k]}')
                if stray.sum() > SPILL_AREA:
                    cut = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.5)
                    shapes[n] = (rgb, (a*np.maximum(cut, keep)).astype(np.float32))
                    log(f'  {n}：去掉 See-through 画在原画别的部件上的 {int(stray.sum())} px')
            # the back hair under the locks, hidden at rest, painted by our split
            # and not by See-through: ours, or the page shows through there
            # when they swing (nothing lies below the back hair)
            gap = keep_large((R[n][1] > .5) & (shapes[n][1] < .5) & (cover > .98) & under_locks, 20)
            if n == 'Hair_Back' and gap.any():
                moved.append((n, R[n][0], R[n][1]*gap))
                log(f'  {n}：被盖住、See-through 没画的 {int(gap.sum())} px 用自拆')
        if vis[n]:
            a = shapes[n][1] if n in shapes else R[n][1]
            cover = cover+a*(1-cover)
            g = gcover.get(group[n], np.zeros((H, W), np.float32))
            gcover[group[n]] = g+a*(1-g)
    colour = {n: rgb.copy() for n, (rgb, a) in shapes.items()}
    for n, rgb, a in moved:
        if n in shapes:
            colour[n] = a[..., None]*rgb+(1-a[..., None])*colour[n]
            shapes[n] = (shapes[n][0], np.maximum(shapes[n][1], a).astype(np.float32))

    def fill(n, src, take, ref):
        """src where take, shifted to meet ref where the layer shows."""
        a = shapes[n][1]
        known = erode((covers[n] < .02) & (a > .98), 2) & (np.abs(ref-src).max(2) < 60)
        src = src+offset_field(np.clip(ref-src, -90, 90), known)
        w = cv2.GaussianBlur(take.astype(np.float32), (0, 0), 2.)[..., None]
        colour[n] = w*src+(1-w)*colour[n]

    if (art/'V5-blank-face.png').exists():
        hidden = (covers['Face_Skin'] > .5) & (S['face'][1] > .02)
        fill('Face_Skin', load(art/'V5-blank-face.png', pad_top), erode(hidden, 2), S['face'][0])
    if (art/'V8-head-off.png').exists():
        v8 = load(art/'V8-head-off.png', pad_top)
        hair_seen = (covers['Hair_Back'] < .02) & (shapes['Hair_Back'][1] > .98)
        body_seen = (covers['Body'] < .02) & (shapes['Body'][1] > .98)
        hairlike = hue_distance(v8, hair_seen) < hue_distance(v8, body_seen)
        hairlike = fill_holes(keep_large(hairlike & (shapes['Hair_Back'][1] > .02), 400), 400)
        hairlike &= erode(figure_alpha(v8) > .5, 2) & (covers['Hair_Back'] > .5)
        # under the clothes V8 is the art where the edit left it: a sleeve's
        # outline there would go into the hair behind the sleeve. Redrawn means
        # changed over an area (the head); every outline differs a little
        redrawn = cv2.morphologyEx(_u8(np.abs(v8-B).max(2) > 40), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)) > 0
        hairlike &= ~((shapes['Body'][1] > .5) & ~dilate(redrawn, 8))
        fill('Hair_Back', v8, hairlike, S['back hair'][0])
    for n in LOCKS:                                     # a lock under another: our split's drawing
        fill(n, R[n][0], (covers[n] > .5) & (R[n][1] > .5), S['front hair'][0])
    for n in [n for n in order if n.startswith('Dangle_')]:
        # See-through draws the dangle on the clothes: under ours, the clothes
        # behind it from our split (V9), so it does not show twice when it swings
        g = 'Body' if group[n] == 'body' else 'Face_Skin'
        fill(g, R[g][0], dilate(R[n][1] > .1, 3), shapes[g][0])

    # A part See-through leaves out or gets wrong over a wide area (no trousers,
    # a neck that stops short inside the collar): there the layer our split
    # shows at rest is taken from our split, and See-through's layers above it
    # give way. Narrow differences are its drawing (a hem a little lower) and stay.
    rest = np.full((H, W, 3), 255., np.float32)
    for n in order[::-1]:
        if vis[n]:
            a = shapes[n][1] if n in shapes else R[n][1]
            rest = a[..., None]*(colour[n] if n in shapes else R[n][0])+(1-a[..., None])*rest
    off = (np.abs(rest-B).max(2) > MISS) & (figure_alpha(B) > .5)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*MISS_WIDTH+1, 2*MISS_WIDTH+1))
    wrong = keep_large(cv2.morphologyEx(off.astype(np.uint8), cv2.MORPH_OPEN, disk) > 0, MISS_AREA)
    if wrong.any():
        near = dilate(off, 2)
        for _ in range(20):                       # and the narrow ragged edge that runs on from it
            wrong = dilate(wrong, 2) & near
        wrong = dilate(wrong, 4)
        # a layer is taken from our split where See-through left it out (mostly
        # bare there) or drew it wrong over a wide area (trousers as a dark
        # block); along an edge, where it drew it its own way, its drawing
        # stays: our back hair along a sleeve carries the sleeve's outline
        take = np.zeros((H, W), bool)
        for c in components(wrong, 1):
            for k in np.unique(shown[c]):
                px = c & (shown == k)
                if k < 0 or order[k] not in shapes:
                    continue
                bare = (shapes[order[k]][1][px] < .5).mean() > .3
                wide = (cv2.morphologyEx(_u8(px & off), cv2.MORPH_OPEN, disk) > 0).sum() >= MISS_AREA
                if bare or wide:
                    take |= px
        wrong = take
        w = cv2.GaussianBlur(wrong.astype(np.float32), (0, 0), 2.)
        for k in np.unique(shown[wrong]):
            n = order[k]
            if k < 0 or n not in shapes:
                continue
            m = w*(shown == k)
            a = shapes[n][1]
            # holes left inside the part, where ours has it and it shows at rest
            # (the page under them is a near colour, so they were not off enough)
            holes = fill_holes(np.maximum(a, m) > .5) & ~(np.maximum(a, m) > .5) & (R[n][1] > .5) & (shown == k)
            m = np.maximum(m, holes.astype(np.float32))
            # where See-through has nothing, our colour outright (its colour there is no colour)
            mc = np.where(a < .1, (m > 0).astype(np.float32), m)
            colour[n] = mc[..., None]*R[n][0]+(1-mc[..., None])*colour[n]
            shapes[n] = (shapes[n][0], (m*R[n][1]+(1-m)*a).astype(np.float32))
            for u in order[:k]:
                if vis[u] and u in shapes:
                    shapes[u] = (shapes[u][0], (shapes[u][1]*(1-m)).astype(np.float32))
            log(f'  {n}：See-through 缺画或画错的 {int((wrong & (shown == k)).sum())} px'
                +(f'、内部的洞 {int(holes.sum())} px' if holes.any() else '')+' 用自拆')

    stack = Stack(B)
    for n in order:
        if n in shapes:
            stack.add(n, group[n], np.clip(colour[n], 0, 255), shapes[n][1], vis[n])
            stack.given.add(n)
        else:
            stack.add(n, group[n], R[n][0], R[n][1], vis[n])      # expressions: our split
    layers = stack.save(out)
    rest = rest_image(layers)
    fig = figure_alpha(B) > .5
    err = np.abs(rest-B).max(2)*fig
    log(f'  See-through 原样：静止与原画（人物范围）平均差 {err.mean():.2f}（0–255），按设计不追原画')
    sheet(layers, B, out/'拆层总览.png')
    meta = dict(pad_top=pad_top, size=[W, H], layers=[l[0] for l in layers][::-1],
                rest_max_diff=round(float(err.max()), 2), seethrough='as_drawn')
    (out/'split.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    return layers


MODELSCOPE = 'https://studio-ljsabc-see-through.api-inference.modelscope.net'
HUGGINGFACE = '24yearsold/see-through-demo'
MAX_RES = {'Hugging Face': 1280, 'ModelScope': 1536}


def modelscope_token():
    import os
    path = Path('~/.cache/modelscope/token').expanduser()
    return os.environ.get('MODELSCOPE_API_TOKEN') or (path.read_text().strip() if path.exists() else None)


def fetch(image, out, res=None, seed=42, site='auto', log=print):
    """Run a See-through demo (Hugging Face, ModelScope when that fails) and write its layers as PNGs."""
    import os
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if Path(image).suffix.lower() == '.psd':
        return unpack(image, out, log)
    try:
        from gradio_client import Client, handle_file
        import psd_tools  # noqa: F401  (unpack needs it; fail before the GPU run, not after)
    except ImportError:
        raise SystemExit('需要两个库：pip install gradio_client psd-tools')
    import httpx
    # httpx waits 5 s by default; through a slow proxy the upload or the queue
    # reply can take longer, and a job dropped after it was queued still uses
    # up the day's GPU time. So: long waits, and only a connection that never
    # got through is tried again.
    wait = dict(httpx_kwargs={'timeout': httpx.Timeout(300., connect=30.)})
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        try:                                                   # saved by `hf auth login` (~/.cache/huggingface/token)
            from huggingface_hub import get_token
            hf_token = get_token()
        except ImportError:
            pass
    ms_token = modelscope_token()

    def hugging_face():
        try:
            return Client(HUGGINGFACE, hf_token=hf_token, **wait)
        except TypeError:
            return Client(HUGGINGFACE, token=hf_token, **wait)

    # token=False: gradio_client would otherwise send the saved Hugging Face
    # token along, and ModelScope refuses the call with it
    modelscope = lambda: Client(MODELSCOPE, token=False, headers={'Authorization': f'Bearer {ms_token}'}, **wait)
    sites = {'hf': [('Hugging Face', hugging_face)], 'modelscope': [('ModelScope', modelscope)]}.get(site)
    if sites is None:                                           # auto: Hugging Face first, ModelScope when it fails
        sites = [('Hugging Face', hugging_face)]+([('ModelScope', modelscope)] if ms_token else [])
    if any(n == 'ModelScope' for n, _ in sites) and not ms_token:
        raise SystemExit('魔搭需要访问令牌：放进 MODELSCOPE_API_TOKEN 或 ~/.cache/modelscope/token')
    psd, errors = None, []
    for name, connect in sites:
        r = min(res or MAX_RES[name], MAX_RES[name])
        log(f'  See-through（{name}），分辨率 {r}')
        for attempt in range(3):
            try:
                psd, _ = connect().predict(handle_file(str(image)), r, seed, True, api_name='/inference')
                break
            except Exception as e:                             # quota, queue, network
                if attempt < 2 and isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
                    log(f'  连接失败（{e}），重试')
                    continue
                errors.append(f'{name}：{e}')
                log(f'  {name} 没有跑成：{e}')
                break
        if psd:
            break
    if not psd:
        hint = ''
        if not hf_token:
            hint += '\nHugging Face 不登录的额度不够跑一次：用免费账号建一个 Read token，放进 HF_TOKEN 或 ~/.cache/huggingface/token'
        if not ms_token:
            hint += '\n备用：魔搭（modelscope.cn）账号的访问令牌，放进 MODELSCOPE_API_TOKEN 或 ~/.cache/modelscope/token'
        raise SystemExit('See-through 没有跑成。\n'+'\n'.join(errors)+hint)
    return unpack(psd, out, log)


def unpack(psd, out, log=print):
    """One full-canvas PNG per layer of a See-through PSD."""
    import shutil
    try:
        from psd_tools import PSDImage
    except ImportError:
        raise SystemExit('需要 psd-tools：pip install psd-tools')
    if Path(psd).resolve() != (out.parent/'seethrough.psd').resolve():
        shutil.copy(psd, out.parent/'seethrough.psd')
    doc = PSDImage.open(psd)
    for layer in doc.descendants():
        if layer.is_group():
            continue
        im = layer.topil()
        if im is None:
            continue
        full = Image.new('RGBA', doc.size, (0, 0, 0, 0))
        full.paste(im.convert('RGBA'), (layer.left, layer.top))
        full.save(out/f'{layer.name.replace("/", "_")}.png')
        log(f'  {layer.name}')
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fetch'); f.add_argument('image'); f.add_argument('out')
    f.add_argument('--res', type=int, default=None); f.add_argument('--seed', type=int, default=42)
    f.add_argument('--site', choices=['auto', 'hf', 'modelscope'], default='auto')
    p = sub.add_parser('split'); p.add_argument('art'); p.add_argument('base'); p.add_argument('out')
    p.add_argument('--pad-top', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'fetch':
        fetch(a.image, a.out, a.res, a.seed, a.site)
    else:
        plan_as_drawn(a.art, a.base, a.out, a.pad_top)


if __name__ == '__main__':
    main()
