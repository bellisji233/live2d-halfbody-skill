"""One illustration plus its edit variants -> layered art (layout.json + parts/).

  python -m rigkit.split ART_DIR OUT_DIR [--pad-top PX] [--hat NAME]

Every variant is the base with ONE thing changed (references/出图.md). Two
kinds are used:

  label maps   S1 (the base) and V4 (the head without its front hair), each
               region filled with one flat colour (PALETTE): which layer a
               pixel belongs to;
  peels        V1 hat off -> V2 side hair off -> V3 bangs off (each only when
               the character has the part), V5 blank face,
               V6 no iris, D* expressions, V9 dangles off: what is behind a
               part, and the part itself as the difference

    with = alpha * part + (1 - alpha) * without

Pixels a viewer sees at rest always come from the base illustration
(Stack.finish unmixes them so the rest image is the base exactly);
variant pixels are used only where a layer is hidden at rest.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
import cv2
from PIL import Image

WHITE = 255.
BASE = 'A0-base'
PALETTE = {'hat': (255, 0, 0), 'bang': (0, 0, 255), 'side': (0, 255, 0), 'hair': (255, 255, 0),
           'face': (255, 0, 255), 'neck': (0, 255, 255), 'body': (255, 128, 0), 'tail': (128, 0, 255)}
# label map regions, top of the drawing order first; tails only on S1 maps that have them
LABEL_MAPS = ('S1-segment', 'V4-skin-mask')
LABEL_ORDER = ['hat', 'bang', 'side', 'face', 'body', 'neck', 'tail', 'hair']
# which image each variant edits; anything not listed edits the base
BASES = {'V2-side-off': 'V1-hat-off', 'V3-bangs-off': 'V2-side-off', 'V4-skin-mask': 'V3-bangs-off',
         'V5-blank-face': 'V3-bangs-off', 'D7-brows-angry': 'V3-bangs-off', 'D8-brows-troubled': 'V3-bangs-off',
         'V7-ears-clear': 'V3-bangs-off', 'V8-head-off': 'V3-bangs-off',
         'F1-blush': 'R0-rest', 'P1-wave': 'R0-rest', 'P2-explain': 'R0-rest', 'P3-point': 'R0-rest'}
# steps a character without a hat, side locks or bangs leaves out; the variants
# that would edit one edit the step before it (no bangs: V4 edits V2)
SKIPPABLE = ('V1-hat-off', 'V2-side-off', 'V3-bangs-off')
ORDER = ['S1-segment', 'V1-hat-off', 'V2-side-off', 'V3-bangs-off', 'V4-skin-mask', 'V5-blank-face', 'V6-no-iris',
         'D1-mouth-open', 'D2-mouth-o', 'D3-mouth-line', 'D4-mouth-frown', 'D5-eyes-closed', 'D6-eyes-smile',
         'D7-brows-angry', 'D8-brows-troubled', 'B0-black-bg', 'V7-ears-clear', 'V8-head-off', 'V9-dangles-off']
REQUIRED = ['S1-segment', 'V4-skin-mask', 'V5-blank-face', 'V6-no-iris',
            'D1-mouth-open', 'D5-eyes-closed']
MOUTHS = {'Open': 'D1-mouth-open', 'O': 'D2-mouth-o', 'Line': 'D3-mouth-line', 'Frown': 'D4-mouth-frown'}
BROWS = {'Angry': 'D7-brows-angry', 'Troubled': 'D8-brows-troubled'}


def base_of(name, present):
    """The image a variant edits, past the steps this character leaves out.
    present(n): whether the character has variant n."""
    b = BASES.get(name, BASE)
    while b in SKIPPABLE and not present(b):
        b = BASES.get(b, BASE)
    return b


# ------------------------------------------------------------------ images
def load(path, pad_top=0, size=None):
    """RGB float32, resized to size (w, h) when given. pad_top white rows
    are added above (room for the head to move up); the canvas grows."""
    im = Image.open(path).convert('RGB')
    if size and im.size != tuple(size):
        im = im.resize(tuple(size), Image.LANCZOS)
    rgb = np.asarray(im).astype(np.float32)
    if pad_top:
        rgb = np.concatenate([np.full((pad_top,)+rgb.shape[1:], WHITE, np.float32), rgb], 0)
    return rgb


def figure_alpha(rgb, tol=8, keep=None, fixed=False, page=None):
    """Character vs the flat white background: flood fill from every white
    border pixel, plus pockets of flat paper white near the edge (gaps
    between loose strands, between an arm and the body) that antialiasing
    cuts off from the page. Edge pixels are matted against white. keep:
    pixels known to be the character (from the label map); white areas
    there (a pom-pom, a collar) stay opaque even where they touch the page.
    fixed: every filled pixel within tol of the white it started from, not
    of its neighbour, so the fill does not creep into a pale garment along
    a faint outline (no label map to hold it). page: pixels known to be the
    page though the fill cannot reach them (white closed in by the figure)."""
    h, w = rgb.shape[:2]
    work = np.clip(rgb, 0, 255).astype(np.uint8).copy()
    mask = np.zeros((h+2, w+2), np.uint8)
    border = [(x, 0) for x in range(w)]+[(x, h-1) for x in range(w)]+[(0, y) for y in range(h)]+[(w-1, y) for y in range(h)]
    for x, y in border:
        if mask[y+1, x+1] == 0 and work[y, x].min() > 240:
            cv2.floodFill(work, mask, (x, y), (0, 0, 0), loDiff=(tol,)*3, upDiff=(tol,)*3,
                          flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8) | (cv2.FLOODFILL_FIXED_RANGE if fixed else 0))
    outside = mask[1:-1, 1:-1] > 0
    paper = (rgb.min(2) > 246) & (rgb.max(2)-rgb.min(2) < 6) & ~outside
    reach = cv2.dilate(outside.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(paper.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 40000:
            c = lab == i
            if reach[c].any() and np.median(rgb[c].min(1)) >= 252:
                outside |= c
    ink = WHITE-rgb
    if keep is not None:
        light = (ink.max(2) <= 70).astype(np.uint8)
        thick = cv2.distanceTransform(light, cv2.DIST_L2, 5) > 3.5
        outside &= ~(keep & (cv2.dilate(thick.astype(np.uint8), disk(4)) > 0) & (light > 0))
    if page is not None:
        outside |= page
    alpha = np.where(outside, 0., 1.)
    # within reach of the page, a pixel is its nearest solid colour blended
    # with white: with = a*F + (1-a)*white (thin gaps between strands too)
    band = (cv2.dilate(outside.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) & ~outside
    solid = ~outside & (ink.max(2) > 70)
    if solid.any() and band.any():
        _, lab = cv2.distanceTransformWithLabels((~solid).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(solid)
        F = ink[ys[lab-1], xs[lab-1]]                      # nearest solid colour, as distance from white
        den = (F*F).sum(2)
        a = np.clip((ink*F).sum(2)/np.maximum(den, 1.), 0, 1)
        # light areas more than a few px thick are white things (a pom-pom,
        # a collar), not gaps: they keep a plain soft edge
        light = (~outside & ~solid).astype(np.uint8)
        thick = cv2.distanceTransform(light, cv2.DIST_L2, 5) > 3.5
        white_thing = (cv2.dilate(thick.astype(np.uint8), disk(4)) > 0) & (light > 0)
        plain = np.clip((253.-rgb.min(axis=2))/9., 0, 1)
        a = np.where(white_thing | (den < 40.**2), plain, a)
        alpha[band] = np.minimum(alpha[band], a[band])
    return alpha.astype(np.float32)


def two_background_alpha(white, black, alpha, reach=6):
    """The same picture on white and on black: at an edge pixel
    white - black = (1-a)*white, so its transparency is measured instead of
    guessed from colours (pale hair against white paper). Used within reach
    of the page found on the white picture only: inside the figure the edit
    re-renders a few spots (an eye corner), and image-model noise is about
    +-10, so values close to 0 or 1 snap."""
    a2 = 1-np.median(white-black, axis=2)/WHITE
    a2 = np.where(a2 > .85, 1., np.where(a2 < .06, 0., a2))
    near = dilate(alpha < .5, reach) & dilate(alpha >= .5, reach)      # the figure's edge only
    out = alpha.copy()
    out[near] = np.clip(a2[near], 0, 1)
    return out.astype(np.float32)


def align(img, ref):
    """Shift img onto ref (global translation, phase correlation).
    Returns the shifted image and the shift that was removed (px)."""
    a = cv2.cvtColor(np.clip(ref, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    out = cv2.warpAffine(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(WHITE,)*3)
    return out, (float(dx), float(dy)), float(resp)


def colour_fit(img, ref, tol=25.):
    """Per-channel (gain, offset) that bring a variant's colours onto the
    base, fitted on pixels that did not change (image models drift); None
    when too few did not."""
    same = np.abs(img-ref).max(2) < tol
    same &= (ref.min(2) < 250)                         # the page says nothing about drift
    if same.sum() < 1000:
        return None
    return [tuple(np.polyfit(img[..., c][same], ref[..., c][same], 1)) for c in range(3)]


def apply_colour_fit(img, fit):
    if fit is None:
        return img, '无'
    out = img.copy()
    for c, (a, b) in enumerate(fit):
        out[..., c] = img[..., c]*a+b
    return np.clip(out, 0, 255), ' '.join(f'{a:.3f}×{b:+.1f}' for a, b in fit)


def match_colours(img, ref, tol=25.):
    """The variant's colours brought onto the base (colour_fit)."""
    return apply_colour_fit(img, colour_fit(img, ref, tol))


def lum(rgb):
    return rgb@np.float32([.299, .587, .114])


# ------------------------------------------------------------------ masks
def _u8(m):
    return m.astype(np.uint8)


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))


def dilate(m, r):
    return cv2.dilate(_u8(m), disk(r)) > 0 if r > 0 else m.astype(bool)


def erode(m, r):
    return cv2.erode(_u8(m), disk(r)) > 0 if r > 0 else m.astype(bool)


def fill_holes(mask, max_area=None):
    """Fill enclosed holes (all of them, or those up to max_area px)."""
    m = _u8(mask)
    n, lab, st, _ = cv2.connectedComponentsWithStats(1-m, connectivity=4)
    h, w = m.shape
    out = m.astype(bool).copy()
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if x == 0 or y == 0 or x+bw == w or y+bh == h:
            continue                                   # touches the border: not a hole
        if max_area is None or area <= max_area:
            out[lab == i] = True
    return out


def components(mask, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(_u8(mask))
    return [lab == i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]


def twin_tails(s1, min_area=2000):
    """The two tails S1 paints violet, left first; [] when it has none."""
    m = np.abs(s1-np.float32(PALETTE['tail'])).max(2) < 60
    tails = sorted(components(m, min_area), key=lambda c: np.nonzero(c)[1].mean())
    return tails if len(tails) == 2 else []


def keep_large(mask, min_area):
    out = np.zeros(mask.shape, bool)
    for c in components(mask, min_area):
        out |= c
    return out


def change_mask(with_part, without, strong=40., close=15, min_area=400):
    """Where the part is: strong differences, thin line jitter removed,
    gaps closed, holes filled (a part over a similar colour still counts)."""
    d = np.abs(with_part-without).max(2)
    core = _u8(d > strong)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, disk(close//2))
    return fill_holes(keep_large(core, min_area)), d


def feather(mask, sigma=.7):
    """Binary region -> alpha with a ~1 px soft rim, solid inside."""
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    return np.maximum(a, erode(mask, 1).astype(np.float32))


def nearest_fill(rgb, src, dst):
    """Colour every dst pixel with the colour of its nearest src pixel."""
    if not src.any():
        return rgb
    _, lab = cv2.distanceTransformWithLabels(_u8(~src), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(src)
    out = rgb.copy()
    sel = dst & ~src
    idx = lab[sel]-1
    out[sel] = rgb[ys[idx], xs[idx]]
    return out


def label_map(img, figure, keys, min_area=150, complete=True):
    """Flat-colour variant -> one mask per palette key. Each pixel goes to
    the nearest palette colour; specks and pin holes are cleaned; figure
    pixels left unlabelled (outlines between fills) go to the upper of the
    neighbouring regions within 3 px, then (complete=True, every region
    filled) to the nearest region."""
    cols = np.array([PALETTE[k] for k in keys], np.float32)
    dist = np.stack([np.abs(img-c).max(2) for c in cols], 2)
    best = dist.argmin(2)
    ok = (dist.min(2) < 110) & figure
    masks = {}
    for i, k in enumerate(keys):
        m = cv2.morphologyEx(_u8((best == i) & ok), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        masks[k] = fill_holes(keep_large(m, min_area), 400)
    taken = np.zeros(figure.shape, bool)
    for k in [k for k in LABEL_ORDER if k in masks]:
        masks[k] &= ~taken
        taken |= masks[k]
    for k in [k for k in LABEL_ORDER if k in masks]:          # gaps: upper layer first
        grab = dilate(masks[k], 3) & figure & ~taken
        masks[k] |= grab
        taken |= grab
    left = figure & ~taken
    if complete and left.any() and taken.any():
        lab = np.full(figure.shape, -1, int)
        for i, k in enumerate(keys):
            lab[masks[k]] = i
        _, near = cv2.distanceTransformWithLabels(_u8(~taken), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(taken)
        idx = near[left]-1
        src = lab[ys[idx], xs[idx]]
        for i, k in enumerate(keys):
            sel = np.zeros(figure.shape, bool)
            sel[left] = src == i
            masks[k] |= sel
    return masks


def matte(with_part, without, region, hi=60., lo=10., grow=4):
    """A part over a known background: with = a*F + (1-a)*without.
    F is the colour of the nearest pixel that is clearly the part
    (difference above hi); a projects the difference onto F - without."""
    diff = with_part-without
    d = np.abs(diff).max(2)
    m = region & (d > lo)
    a = np.zeros(d.shape, np.float32)
    rgb = with_part.copy()
    if not m.any():
        return rgb, a
    core = m & (d > hi)
    if core.sum() < 4:
        core = m & (d >= np.percentile(d[m], 80))
    if grow is not None:
        m &= dilate(core, grow)
    _, lab = cv2.distanceTransformWithLabels(_u8(~core), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(core)
    F = with_part[ys[lab-1], xs[lab-1]]
    den = ((F-without)**2).sum(2)
    a = np.where(m, np.clip((diff*(F-without)).sum(2)/np.maximum(den, 1.), 0, 1), 0).astype(np.float32)
    a[core] = 1.
    rgb = np.where((a >= .9)[..., None], with_part, F)
    return rgb, a


def tidy(a, reach=.6, min_area=12, hair=None):
    """Drop specks of a matte that lie away from its main piece (redraw
    noise): pieces farther than reach x the main piece's size are cut.
    hair: front hair mask; small loose pieces against it are strand ends
    the label map called face, and are cut too."""
    solid = a > .25
    parts = components(solid, min_area)
    if not parts:
        return a*0
    main = max(parts, key=lambda c: c.sum())
    x0, y0, x1, y1 = _box(main)
    d = cv2.distanceTransform(_u8(~main), cv2.DIST_L2, 5)
    near = cv2.distanceTransform(_u8(~hair), cv2.DIST_L2, 5) if hair is not None else None
    strand = lambda c: near is not None and c is not main and c.sum() < main.sum()/4 and np.median(near[c]) < 12
    keep = np.logical_or.reduce([c for c in parts if d[c].min() <= reach*max(x1-x0, y1-y0) and not strand(c)])
    return a*dilate(keep, 2)


# ------------------------------------------------------------------ layers
class Stack:
    """Layers listed top to bottom. Visible pixels come from the base."""

    def __init__(self, base):
        self.base = base
        self.layers = []           # (name, group, rgb, alpha, visible), top first
        self.keep = {}             # name -> pixels whose own colour is kept even where seen
        self.given = set()         # names kept exactly as given (not fitted to the base)

    def add(self, name, group, rgb, alpha, visible=True, keep=None):
        self.layers.append((name, group, rgb.astype(np.float32), np.clip(alpha, 0, 1).astype(np.float32), visible))
        if keep is not None:
            self.keep[name] = keep

    def finish(self):
        """Where nothing visible lies above a visible layer, its colour is
        solved from the base and what is under it, so the layers at rest
        composite to the base exactly (soft edges included). Where a soft
        edge is too faint to make the base colour over what is under it (a
        dark outline under the last few percent of a hair edge), that edge
        is made just opaque enough: the line then belongs to the upper layer
        and moves with it, instead of staying behind on the layer below."""
        cover, above = np.zeros(self.base.shape[:2], np.float32), []
        for _, _, _, a, vis in self.layers:
            above.append(cover.copy())
            if vis:
                cover = cover+a*(1-cover)
        under = np.full(self.base.shape, WHITE, np.float32)          # the page
        out = []
        for (name, group, rgb, a, vis), cov in zip(reversed(self.layers), reversed(above)):
            rgb = rgb.copy()
            if vis and name in self.given:
                under = a[..., None]*rgb+(1-a[..., None])*under
            elif vis:
                seen = (cov < .02) & (a > .02)
                if name in self.keep:
                    seen &= ~self.keep[name]
                gap = self.base-under
                need = np.where(gap < 0, -gap/np.maximum(under, 1.), gap/np.maximum(WHITE-under, 1.)).max(2)
                a = np.where(seen & (need > a), np.minimum(1., need*1.15), a).astype(np.float32)
                solved = np.clip((self.base-(1-a)[..., None]*under)/np.maximum(a, .02)[..., None], 0, 255)
                rgb[seen] = solved[seen]
                under = a[..., None]*rgb+(1-a[..., None])*under
            out.append((name, group, rgb, a, vis))
        return out[::-1]

    def save(self, folder):
        folder = Path(folder); (folder/'parts').mkdir(parents=True, exist_ok=True)
        layers = self.finish()
        for old in (folder/'parts').glob('*.png'):                  # parts of an earlier split
            if old.stem not in {l[0] for l in layers}:
                old.unlink()
        doc = dict(width=self.base.shape[1], height=self.base.shape[0], layers=[])
        for name, group, rgb, a, vis in reversed(layers):              # layout is bottom to top
            rgba = np.dstack([np.clip(rgb, 0, 255), a*255]).round().astype(np.uint8)
            Image.fromarray(rgba).save(folder/'parts'/f'{name}.png')
            doc['layers'].append(dict(name=name, texture=f'parts/{name}.png', group=group, visible=bool(vis)))
        (folder/'layout.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2)+'\n')
        return layers


def rest_image(layers):
    out = np.full(layers[0][2].shape, WHITE, np.float32)
    for name, group, rgb, a, vis in reversed(layers):
        if vis:
            out = a[..., None]*rgb+(1-a[..., None])*out
    return out


# ------------------------------------------------------------------ face
def _box(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)


def _centroid(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()])


def find_features(v3, v5, face, log):
    """Eyes, brows, nose and mouth as the blobs where the face differs from
    the blank face. Returns {key: grown blob mask}."""
    fx0, fy0, fx1, fy1 = _box(face)
    fw, fh = fx1-fx0, fy1-fy0
    cx = _centroid(face)[0]
    inner = erode(fill_holes(face), 5)
    d = np.abs(v3-v5).max(2)
    core = cv2.morphologyEx(_u8((d > 30) & inner), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    blobs = []
    for c in components(dilate(core, 7), 120):
        b = _box(c); cen = _centroid(c & core) if (c & core).any() else _centroid(c)
        blobs.append(dict(mask=c, box=b, c=cen, area=int((c & core).sum())))
    feats = {}
    for side, sel in (('L', lambda b: b['c'][0] < cx-.06*fw), ('R', lambda b: b['c'][0] > cx+.06*fw)):
        cand = [b for b in blobs if sel(b) and fy0+.2*fh < b['c'][1] < fy0+.8*fh]
        if not cand:
            raise SystemExit(f'找不到{side}眼')
        eye = max(cand, key=lambda b: b['area'])
        feats['eye_'+side] = eye
        eh = eye['box'][3]-eye['box'][1]
        brow = [b for b in blobs if sel(b) and b is not eye and b['box'][3] < eye['box'][1]+.15*eh
                and b['c'][1] > fy0-.05*fh and eye['box'][0]-.12*fw < b['c'][0] < eye['box'][2]+.12*fw]
        if brow:
            m = np.logical_or.reduce([b['mask'] for b in brow])
            feats['brow_'+side] = dict(mask=m, box=_box(m), c=_centroid(m), area=sum(b['area'] for b in brow))
    for side, other in (('L', 'R'), ('R', 'L')):
        if 'brow_'+side in feats or 'brow_'+other not in feats:
            continue
        # a brow close over its eye joins it into one blob: cut it off along the
        # empty row between them nearest the other brow's lower edge
        eye, ob = feats['eye_'+side], feats['brow_'+other]
        ink = (eye['mask'] & core).any(1)
        gaps = [y for y in range(int(np.nonzero(ink)[0][0]), int(eye['c'][1])) if not ink[y]]
        if not gaps:
            continue
        cut = min(gaps, key=lambda y: abs(y-ob['box'][3]))
        above = (np.arange(len(ink)) <= cut)[:, None]
        part = eye['mask'] & above & core
        pb, (ow, oh) = _box(part), ob['box'][2:]-ob['box'][:2]
        if abs(_centroid(part)[1]-ob['c'][1]) > max(8, oh) or pb[2]-pb[0] < .5*ow:
            continue                                     # not a brow (a crease line over the eye)
        for key, m in (('brow_'+side, eye['mask'] & above), ('eye_'+side, eye['mask'] & ~above)):
            feats[key] = dict(mask=m, box=_box(m), c=_centroid(m & core), area=int((m & core).sum()))
        log(f'  {"左" if side == "L" else "右"}眉和眼连成一块，在 y={cut} 处分开')
    used = [f['mask'] for f in feats.values()]
    eye_y = (feats['eye_L']['c'][1]+feats['eye_R']['c'][1])/2
    rest = [b for b in blobs if not any((b['mask'] & u).any() for u in used)
            and abs(b['c'][0]-cx) < .2*fw and b['c'][1] > eye_y]
    if not rest:
        raise SystemExit('找不到嘴')
    low = max(rest, key=lambda b: b['c'][1])
    # a thin mouth line fades in the middle and splits into pieces at one height
    parts = [b for b in rest if low['box'][1]-.03*fh <= b['c'][1] <= low['box'][3]+.03*fh]
    m = np.logical_or.reduce([b['mask'] for b in parts])
    mouth = dict(mask=m, box=_box(m), c=_centroid(m), area=sum(b['area'] for b in parts))
    feats['mouth'] = mouth
    nose = [b for b in rest if not any(b is p for p in parts) and b['c'][1] < mouth['box'][1]]
    if nose:
        m = np.logical_or.reduce([b['mask'] for b in nose])
        feats['nose'] = dict(mask=m, box=_box(m), c=_centroid(m), area=sum(b['area'] for b in nose))
    skipped = [b for b in blobs if not any((b['mask'] & f['mask']).any() for f in feats.values())]
    for b in skipped:
        log(f'  脸上未归类的一块（留在 Face_Skin）：{b["c"].round().astype(int).tolist()}，{b["area"]} px')
    return feats, (cx, fw, fh)


def split_eye(B, v6, blank, blob):
    """One eye -> white, iris, upper lash, crease, lower lash (masks).
    Everything in the eye that differs from the blank face lands in one of
    them: bright pixels joined to the iris are white, the rest is ink."""
    zone = dilate(blob, 10)
    ink = zone & (np.abs(B-blank).max(2) > 15)
    iris_m, _ = change_mask(B, v6, strong=40, close=5, min_area=20)
    iris_m &= zone
    eye_all = fill_holes(ink | iris_m) & zone
    lb, l6 = lum(B), lum(v6)
    inner = eye_all & ~iris_m
    white_l = np.median(l6[inner & (l6 > np.percentile(l6[inner], 60))]) if inner.any() else 240.
    lash_l = np.percentile(lb[ink & ~iris_m], 5) if (ink & ~iris_m).any() else 60.
    bright = eye_all & (l6 > .55*white_l+.45*lash_l)
    keep = [c for c in components(bright, 4) if (dilate(c, 1) & iris_m).any()] if iris_m.any() else []
    white = np.logical_or.reduce(keep) if keep else keep_large(bright, 30)
    white = fill_holes(cv2.morphologyEx(_u8(white | iris_m), cv2.MORPH_CLOSE, disk(1)) > 0) & eye_all
    rest = ink & ~white & ~iris_m
    ys = np.nonzero(white)[0]
    wc = ys.mean() if len(ys) else _centroid(blob)[1]
    upper = np.zeros(blob.shape, bool); lower = upper.copy(); crease = upper.copy()
    near_white = dilate(white, 4)
    rows = np.arange(blob.shape[0])[:, None]
    loose = []
    for c in components(rest, 4):
        cy = _centroid(c)[1]
        if cy > wc and not (c & near_white & (rows < wc)).any():
            lower |= c
        elif (c & near_white).any():
            upper |= c
        else:
            loose.append(c)
    # pieces off the lash: a crease when they sit above the lash line
    top = {x: ys.min() for x in np.nonzero(upper.any(0))[0] for ys in [np.nonzero(upper[:, x])[0]]}
    for c in loose:
        ys, xs = np.nonzero(c)
        tops = [top[x] for x in xs if x in top]
        if tops and ys.mean() < np.mean(tops)-1:
            crease |= c
        else:
            upper |= c
    # a crease joined to the lash (the shaded skin between them reads as ink)
    # is lighter ink above the dark lash line, in a long arc; left on the
    # lash it travels down with it and shows above the closed eye
    skin_l = np.median(lum(blank)[zone & ~eye_all]) if (zone & ~eye_all).any() else 230.
    dark = upper & (lb < lash_l+.3*(skin_l-lash_l))
    first = np.where(dark.any(0), dark.argmax(0), blob.shape[0])          # top of the dark line, per column
    cand = upper & ~dark & (rows < first[None, :]-1)
    if white.any():                                                     # over the eye, not at a pale lash end
        wx = np.nonzero(white.any(0))[0]
        cand[:, :wx.min()] = False; cand[:, wx.max()+1:] = False
    arc = np.zeros(blob.shape, bool)
    if upper.any():
        xs = np.nonzero(upper.any(0))[0]
        for c in components(cand, 4):
            cx = np.nonzero(c.any(0))[0]
            if cx.max()-cx.min()+1 >= .25*(xs.max()-xs.min()+1):
                arc |= c
    return dict(white=white, iris=iris_m, upper=upper & ~arc, lower=lower, crease=crease | arc)


def extend_iris(rgb, iris, white):
    """Complete the iris under the lids (an ellipse through its visible rim)."""
    cs, _ = cv2.findContours(_u8(iris), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return rgb, iris
    pts = np.vstack(cs).reshape(-1, 2)
    edge_ok = ~dilate(~white, 2)[pts[:, 1], pts[:, 0]]                 # rim inside the white
    if edge_ok.sum() < 6:
        return rgb, iris
    (ex, ey), (ax, ay), ang = cv2.fitEllipse(pts[edge_ok].astype(np.float32))
    full = np.zeros(iris.shape, np.uint8)
    cv2.ellipse(full, ((ex, ey), (ax, ay), ang), 1, -1)
    full = (full > 0) & dilate(white, 6)
    bb = _box(iris)
    if (full.sum() > 2.2*iris.sum()) or not (bb[0]-8 < ex < bb[2]+8):
        return rgb, iris                                                  # fit went wrong: keep as is
    grown = iris | full
    return nearest_fill(rgb, iris, grown), grown


def crease_arc(m, min_span=.25):
    """The lid crease drawn above a lash line, even where it joins the line
    (at a corner, through a lash tip): per column, a thin run of ink above
    the thickest run; only a long arc of them counts (a lash tip above the
    line is short)."""
    if not m.any():
        return m
    cut, line = np.zeros(m.shape, bool), np.zeros(m.shape, bool)
    for x in np.nonzero(m.any(0))[0]:
        ys = np.nonzero(m[:, x])[0]
        runs = np.split(ys, np.nonzero(np.diff(ys) > 1)[0]+1)
        thick = max(runs, key=len)
        line[thick, x] = True
        for r in runs:
            if r[-1] < thick[0] and len(r) <= max(4, len(thick)//2):
                cut[r, x] = True
    xs = np.nonzero(m.any(0))[0]
    width = xs.max()-xs.min()+1
    arc = np.zeros(m.shape, bool)
    for c in components(dilate(cut, 1) & ~line, 1):
        cx = np.nonzero(c.any(0))[0]
        if cx.max()-cx.min()+1 >= min_span*width:
            arc |= c
    return arc & ~dilate(line, 1)


def hue_distance(rgb, region, sigma=6):
    """Per pixel, how far its hue (Lab a, b; lightness ignored, so dark
    outlines keep the hue of what they outline) is from the region's mean
    hue around the nearest region pixel."""
    ab = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)[..., 1:].astype(np.float32)
    m = erode(region, 2)
    if not m.any():
        return np.full(m.shape, np.inf, np.float32)
    w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)
    mean = cv2.GaussianBlur(ab*m[..., None], (0, 0), sigma)/np.maximum(w, 1e-3)[..., None]
    _, lab = cv2.distanceTransformWithLabels(_u8(~m), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(m)
    return np.linalg.norm(ab-mean[ys[lab-1], xs[lab-1]], axis=2)


def stray_hair(rgb, body, hair, reach=14, max_area=400):
    """Thin hair ends lying on the clothes, which the label map paints as
    clothes: small pieces next to the hair whose colour is the nearby hair's,
    not the nearby clothes'. Left on the body they stay behind when the hair
    swings."""
    dh, db = hue_distance(rgb, hair), hue_distance(rgb, body)
    cand = dilate(hair, reach) & body & (dh < 20) & (dh < .6*db)
    small = np.zeros(cand.shape, bool)
    for c in components(cand, 1):
        if c.sum() <= max_area:
            small |= c
    return dilate(small, 1) & body


def extend_rows(rgb, mask, allowed, up=0, down=0, inset=3):
    """Continue each column of mask up or down into allowed pixels, with the
    colour a few px inside its end (hidden parts: neck under the chin, etc.)."""
    out_m = mask.copy(); out = rgb.copy()
    h = mask.shape[0]
    for x in np.nonzero(mask.any(0))[0]:
        ys = np.nonzero(mask[:, x])[0]
        if up:
            y0 = ys.min(); src = out[min(y0+inset, ys.max()), x]
            for y in range(y0-1, max(-1, y0-up-1), -1):
                if not allowed[y, x]:
                    break
                out_m[y, x] = True; out[y, x] = src
        if down:
            y1 = ys.max(); src = out[max(y1-inset, ys.min()), x]
            for y in range(y1+1, min(h, y1+down+1)):
                if not allowed[y, x]:
                    break
                out_m[y, x] = True; out[y, x] = src
    return out, out_m


def hair_behind(v3, hair, face, neck_top, darken=.9, behind=None, clothes=None, long_from=None):
    """Back hair continued behind the face, ears and neck top, so a turning
    head shows hair at its edge. Colour spread from the hair itself; where
    behind (the head-off variant) shows hair, its painted hair is used."""
    region = fill_holes(dilate(hair | face, 4)) | dilate(face, 6)
    rows = np.nonzero(hair.any(1))[0]
    cx = _centroid(face)[0]
    for y in rows:                                      # between hair hanging on both sides
        xs = np.nonzero(hair[y])[0]
        if xs.min() < cx < xs.max():
            region[y, xs.min():xs.max()+1] = True
    region |= neck_top
    region &= ~hair
    if not region.any():
        return v3, hair
    # strands continue straight down from the hair above each column;
    # columns with no hair above take the nearest hair
    out = v3.copy()
    h, w = hair.shape
    rows_i = np.arange(h)[:, None].repeat(w, 1)
    last = np.maximum.accumulate(np.where(hair, rows_i, -1), axis=0)     # lowest hair row above, per column
    src = region & (last >= 0)
    ys, xs = np.nonzero(src)
    out[ys, xs] = v3[np.maximum(last[ys, xs]-4, 0), xs]*darken           # a little inside the hair's edge line
    out = nearest_fill(out, hair | src, region)
    # from the chin down, the hair beside the neck is the hair's ends: take
    # the nearest hair in the same row, not the crown above
    fx0, fy0, fx1, fy1 = _box(face)
    hm = hair.astype(np.float32)[..., None]                       # local mean: no single outline pixel
    smooth = cv2.GaussianBlur(v3*hm, (0, 0), 4)/np.maximum(cv2.GaussianBlur(hm, (0, 0), 4)[..., None], 1e-3)
    for y in range(int(fy1-.1*(fy1-fy0)), h):
        todo = np.nonzero(region[y])[0]
        have = np.nonzero(hair[y])[0]
        if len(todo) and len(have):
            near = have[np.abs(have[None, :]-todo[:, None]).argmin(1)]
            out[y, todo] = smooth[y, near]*darken
            if long_from is not None and y > long_from:
                # long hair behind the body: the strands beside it mirrored in,
                # so hair (not a smear) shows when the hair swings off the arm
                src = 2*near-todo+6*np.sign(near-todo)            # past the outline at the edge
                ok = (src >= 0) & (src < w)
                ok[ok] = hair[y, src[ok]]
                out[y, todo[ok]] = v3[y, src[ok]]*darken
    out[region] = cv2.GaussianBlur(out, (0, 0), 1.2)[region]
    if behind is not None:
        # the head-off variant paints the hair behind the head: used behind
        # the face and neck and along the face edge (the outline drawn there
        # is the face's; left on the hair it would swing away from the face).
        # Where it shows clothes (hued like the clothes, not the hair) there
        # is no hair.
        bfig = figure_alpha(behind) > .5
        hairlike = np.ones(hair.shape, bool)
        if clothes is not None:
            # near the clothes (and where the neck was: the variant may draw
            # the collar's back there), a pixel not clearly hair-hued is clothes
            hairlike &= ~((dilate(clothes, 12) | neck_top) & (hue_distance(behind, clothes, 10) < hue_distance(behind, hair, 10)))
        use = (region | (dilate(face, 4) & hair)) & bfig & hairlike
        out[use] = behind[use]
        region &= ~(bfig & ~hairlike)
    return out, hair | region


# ------------------------------------------------------------------ plan
def plan(art, out, pad_top=0, hat='Hat', log=print):
    """The standard variant set (variants.ORDER) -> art folder."""
    art, out = Path(art), Path(out)
    missing = [n for n in REQUIRED if not (art/f'{n}.png').exists()]
    if missing:
        raise SystemExit('缺少变体：'+'、'.join(missing))
    size = Image.open(art/f'{BASE}.png').size
    B = load(art/f'{BASE}.png', pad_top)
    cache = {}

    def V(name):
        if name not in cache:
            img, (dx, dy), _ = align(load(art/f'{name}.png', pad_top, size), B)
            img, gain = match_colours(img, B) if name not in LABEL_MAPS else (img, '无（色块图）')
            log(f'  {name}: 对位偏移 {np.hypot(dx, dy):.1f} px，色差校正 {gain}')
            cache[name] = img
        return cache[name]
    have = lambda n: (art/f'{n}.png').exists()

    s1 = V('S1-segment')
    keys = [k for k in LABEL_ORDER if k != 'tail' or twin_tails(s1)]
    painted = np.stack([np.abs(s1-np.float32(PALETTE[k])).max(2) for k in keys], 2).min(2) < 110
    fig_a = figure_alpha(B, keep=erode(painted, 3))
    if have('B0-black-bg'):
        K = V('B0-black-bg').copy()
        K[:pad_top] = 0                                                  # the added rows are its page too
        fig_a = two_background_alpha(B, K, fig_a)
    fig = fig_a > .5
    S = label_map(s1, fig, keys)
    if 'tail' in S:
        # twin tails are back hair to this split; the See-through merge cuts them apart
        S['hair'] |= S.pop('tail')
    v3 = V(base_of('V5-blank-face', have)); v5 = V('V5-blank-face')     # the head without its front hair
    # V4 recolours only the head: an unchanged pixel that happens to be near
    # a palette colour (a mint cardigan next to neck cyan) is not a label
    v4 = V('V4-skin-mask')
    recoloured = dilate(np.abs(v4-v3).max(2) > 60, 1)
    U = label_map(v4, (figure_alpha(v3) > .5) & recoloured, ['face', 'neck', 'hair'], complete=False)
    face = keep_large(U['face'] | S['face'], 2000)
    neck = U['neck'] | S['neck']
    stack = Stack(B)
    # feather only toward other layers; toward the page the matte sets the edge
    soft = lambda m: feather(m | (dilate(m, 2) & ~fig))*fig_a

    # front hair and hat: visible region from the label map, the part hidden
    # under the layers above it from the peel variants
    above = np.zeros(fig.shape, bool)
    front = np.zeros(fig.shape, bool)                                 # everything a front hair layer covers
    fx0, fy0, fx1, fy1 = _box(face)
    cx = _centroid(face)[0]
    if have('V1-hat-off') and S['hat'].any():
        stack.add(hat, 'head', B, soft(S['hat']))
        above |= S['hat']
    strays = stray_hair(B, S['body'] & fig, S['side'] | S['hair'])
    rows = lambda m: np.nonzero(m.any(1))[0]
    if S['hair'].any() and S['neck'].any() and rows(S['hair']).max() > rows(S['neck']).max():
        # long hair: only the ends of the side locks swing with them; strands of
        # the long hair lying on the clothes stay on the body (below the
        # shoulders the long hair moves with the body anyway)
        near_side = dilate(S['side'], 24)
        strays = np.logical_or.reduce([c for c in components(strays, 1) if (c & near_side).any()] or [np.zeros(fig.shape, bool)])
    peels = [(name, key, base_of(wo, have), wo) for name, key, wo in
             [('Hair_Bang', 'bang', 'V3-bangs-off'), ('Hair_Side', 'side', 'V2-side-off')] if have(wo)]
    for name, key, w, wo in peels:
        hidden = dilate(above, 6)
        ext, _ = change_mask(V(w) if w != BASE else B, V(wo), strong=40, close=9, min_area=80)
        # plus its own outline and light edge, which the label map may leave a
        # few px short of (what changes when the layer is removed, next to it)
        m = S[key] | (ext & hidden & fig) | (ext & dilate(S[key], 6) & fig)
        front |= m
        # a feature line seen through a gap between strands (a brow end) is the
        # feature's: on the hair it would move off with the hair
        seen_feature = dilate(S[key] & face & (np.abs(v3-v5).max(2) > 25) & (np.abs(B-v3).max(2) < 20), 1) \
            if key == 'bang' else np.zeros(fig.shape, bool)
        hair_alpha = lambda mm: np.where(seen_feature, 0., soft(mm))
        colour = np.where(S[key][..., None], B, V(w) if w != BASE else B)
        if key == 'side':
            m |= strays                                                 # hair ends on the clothes swing with it
            left = np.arange(m.shape[1])[None, :] < cx                   # image left of the face centre
            for s, half in (('L', left), ('R', ~left)):
                ms = keep_large(m & half, 200)
                if ms.any():
                    stack.add(f'Hair_Side_{s}', 'head', colour, hair_alpha(ms))
        elif m.any():
            stack.add(name, 'head', colour, hair_alpha(m))
        above |= S[key]

    # features (on the head without front hair, against the blank face)
    feats, (cx, fw, fh) = find_features(v3, v5, face, log)
    v6 = V('V6-no-iris')
    # the base with its features blanked: v5 only where v3 has a feature;
    # front hair and loose strands the label map missed stay as drawn, so
    # they cancel out of every "with minus without" below
    featured = dilate(np.abs(v3-v5).max(2) > 15, 3) & ~above
    bare = np.where(featured[..., None], v5, B)
    # brow edits are of v3: v3 with only its brows blanked, so its eyes and
    # strands cancel out too
    brows_at = np.logical_or.reduce([feats[k]['mask'] for k in ('brow_L', 'brow_R') if k in feats] or [np.zeros(fig.shape, bool)])
    v3_bare = np.where(brows_at[..., None], v5, v3)
    blank = lambda name: v3_bare if BASES.get(name) == 'V3-bangs-off' else bare
    # an eye running under front hair is taken from v3 there: hidden at rest,
    # whole when the face turns under the hair
    under_hair = dilate(above, 3)
    Be = np.where(under_hair[..., None], v3, B)
    v6e = np.where(under_hair[..., None], v3, v6)
    bare_e = np.where(dilate(np.abs(v3-v5).max(2) > 15, 3)[..., None], v5, Be)
    brow_layers, eye_layers = [], {}
    for side in 'RL':
        e = split_eye(Be, v6e, bare_e, feats['eye_'+side]['mask'])
        near_hair = cv2.distanceTransform(_u8(~above), cv2.DIST_L2, 5) < 12
        for c in components(e['crease'], 1):                  # a strand end above the lash is not a crease
            if np.median(near_hair[c]):
                e['crease'] &= ~c
        white_rgb = v6e
        below = np.where(e['white'][..., None], white_rgb, bare_e)
        L = []
        D5 = V('D5-eyes-closed')
        box = feats['eye_'+side]['box']; ew, eh = box[2]-box[0], box[3]-box[1]
        zone = np.zeros(fig.shape, bool)
        zone[int(box[1]-.15*eh):int(box[3]+.6*eh), int(box[0]-.25*ew):int(box[2]+.25*ew)] = True
        if 'brow_'+side in feats:
            zone &= ~dilate(feats['brow_'+side]['mask'], 3)
        zone &= ~dilate(above, 3)                                  # under front hair: hidden anyway
        for nm, src in (('Smile', 'D6-eyes-smile'), ('Closed', 'D5-eyes-closed')):
            if have(src):
                # off the face outline, which an edit may redraw next to the lash
                # the closed eye keeps its crease, as the smile eye does: it shows
                # once the eye is all but shut (eyes.closed_swap)
                rgb, a = matte(V(src), blank(src), zone & erode(face, 5))
                a = tidy(a, hair=above)
                L.append((f'Eye_{nm}_{side}', rgb, a, False))
        if e['lower'].any():
            rgb, a = matte(Be, below, dilate(e['lower'], 2) & ~e['white'])
            L.append((f'Eye_Lower_{side}', rgb, a, True))
        if e['crease'].any():
            rgb, a = matte(Be, bare_e, dilate(e['crease'], 2))
            L.append((f'Eye_Crease_{side}', rgb, a, True))
        rgb, a = matte(Be, below, dilate(e['upper'], 2) & ~e['iris'])
        L.append((f'Eye_Open_{side}', rgb, tidy(a, 1., hair=above), True))
        iris_rgb, iris_m = extend_iris(Be, e['iris'], e['white'])
        a = np.maximum(matte(Be, white_rgb, dilate(e['iris'], 1) & e['white'])[1], (iris_m & ~e['iris']).astype(np.float32))
        a[erode(e['iris'], 1)] = 1.
        L.append((f'Iris_{side}', iris_rgb, a, True))
        under_lash = dilate(e['white'], 2) & e['upper']
        L.append((f'Eye_White_{side}', nearest_fill(white_rgb, e['white'], e['white'] | under_lash),
                  feather(e['white'] | under_lash, .5), True))
        eye_layers[side] = L
    # v3 against v5 differs only at the features (noise under 10), so a
    # faint part (a pale lip line, a nose highlight) counts as solid from 25
    for side in 'RL':
        if 'brow_'+side not in feats:
            continue
        b = feats['brow_'+side]
        bb = b['box']; bl = bb[2]-bb[0]
        zone = np.zeros(fig.shape, bool)
        zone[int(bb[1]-.5*bl):int(bb[3]+.35*bl), int(bb[0]-.35*bl):int(bb[2]+.35*bl)] = True
        zone &= face & ~dilate(feats['eye_'+side]['mask'] & (np.abs(v3-v5).max(2) > 15), 2)   # the eye's own lines
        for nm, src in BROWS.items():
            if have(src):
                rgb, a = matte(V(src), blank(src), zone)
                brow_layers.append((f'Brow_{nm}_{side}', rgb, tidy(a, .3), False))
        rgb, a = matte(v3, v5, dilate(b['mask'], 6), hi=25., grow=None)
        brow_layers.append((f'Brow_Neutral_{side}', rgb, a, True))
    for nm, rgb, a, vis in brow_layers:                       # R before L: L ends up below in layout
        stack.add(nm, 'head', rgb, a, vis)

    mouth = feats['mouth']; mb = mouth['box']
    mw = max(mb[2]-mb[0], .1*fw)
    mzone = np.zeros(fig.shape, bool)
    top = feats['nose']['box'][3]+4 if 'nose' in feats else mb[1]-.08*fh
    mzone[int(max(top, mb[1]-.1*fh)):int(min(fy1-3, mb[3]+.14*fh)), int(mouth['c'][0]-1.1*mw):int(mouth['c'][0]+1.1*mw)] = True
    mzone &= erode(face, 6) & ~dilate(above, 3)                 # off the jaw outline, which every edit redraws
    for nm in ['Frown', 'Line', 'O', 'Open']:
        if have(MOUTHS[nm]):
            rgb, a = matte(V(MOUTHS[nm]), blank(MOUTHS[nm]), mzone)
            a = tidy(a, .4)
            a = np.maximum(a, (fill_holes(a > .5) & ~(a > .5)).astype(np.float32))
            if nm in ('Open', 'O'):
                # an open mouth is one solid shape: a tongue as pale as the skin
                # mattes as a notch, which shows as a drip when the mouth half opens
                solid = _u8(a > .5)
                hull = np.zeros_like(solid)
                for c in cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                    cv2.fillPoly(hull, [cv2.convexHull(c)], 1)
                notch = keep_large(cv2.morphologyEx(hull & ~solid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0, 20)
                rgb = np.where(notch[..., None], V(MOUTHS[nm]), rgb)
                a = np.maximum(a, notch.astype(np.float32))
                # inside its outline every pixel is mouth, in the edit's own colour:
                # teeth as pale as the skin otherwise matte half clear, in the
                # colour of the lip line next to them (grey blocks along the upper
                # lip); a highlight the edit drew under the lower lip, apart, goes
                inside = erode(hull > 0, 1)
                rgb = np.where(inside[..., None], V(MOUTHS[nm]), rgb)
                a = np.where(inside, 1., a*dilate(hull > 0, 1)).astype(np.float32)
            stack.add(f'Mouth_{nm}', 'head', rgb, a, False)
    rgb, a = matte(v3, v5, dilate(mouth['mask'], 6), hi=25., grow=None)
    stack.add('Mouth_Closed', 'head', rgb, a)
    if 'nose' in feats:
        rgb, a = matte(v3, v5, dilate(feats['nose']['mask'], 6), hi=25., grow=None)
        stack.add('Nose', 'head', rgb, a)
    for side in 'RL':
        for nm, rgb, a, vis in eye_layers[side]:
            stack.add(nm, 'head', rgb, a, vis)

    # lash tips drawn over the front hair would stay on the hair when the eye
    # closes: they go on a lash layer above the hair (it moves like the lash),
    # and the hair under them takes the nearest hair colour. Found against the
    # closed-eye edit (the base with its hair untouched): darker in the base
    # than there, on the hair, by the lash
    hair_at = [i for i, l in enumerate(stack.layers) if l[0].startswith(('Hair_Bang', 'Hair_Side'))]
    if hair_at and have('D5-eyes-closed'):
        front_hair = np.logical_or.reduce([stack.layers[i][3] > .5 for i in hair_at])
        d5 = V('D5-eyes-closed')
        overs = {}
        for side in 'RL':
            lash = next(a for nm, rgb, a, vis in eye_layers[side] if nm.startswith('Eye_Open_'))
            over = front_hair & dilate(lash > .1, 4) & (lum(d5)-lum(B) > 30)
            over = np.logical_or.reduce(components(over, 3) or [np.zeros(fig.shape, bool)])
            if over.sum() >= 8:
                overs[side] = over
        if overs:
            under = dilate(np.logical_or.reduce(list(overs.values())), 2)
            fill = nearest_fill(B, front_hair & ~under, under)
            fill = np.where(under[..., None], cv2.GaussianBlur(fill, (0, 0), 1.), B)
            for i in hair_at:
                nm, g, rgb, a, vis = stack.layers[i]
                stack.layers[i] = (nm, g, np.where(under[..., None], fill, rgb), a, vis)
            for side, over in overs.items():
                rgb, a = matte(B, fill, dilate(over, 1), grow=None)
                stack.layers.insert(hair_at[0], (f'Eye_Open_Over_{side}', 'head', rgb, a, True))
                log(f'  压在头发上的睫毛 {side}：{int(over.sum())} px')

    # face skin, with its outline; blank skin under the features
    hair_u = U['hair'] | S['hair']
    face_m = fill_holes(face | (dilate(face, 3) & fig & ~neck & ~hair_u & ~S['body']))
    # feature pixels stay blank skin here: whatever a feature layer missed
    # shows up in the rest check instead of being painted on the skin
    # (a feature is also on v3; a line only on the base is front hair the
    # label map missed, and the skin takes it)
    marks = np.logical_or.reduce([dilate(f['mask'], 6) for f in feats.values()]) & (np.abs(B-v5).max(2) > 15) \
        & (np.abs(v3-v5).max(2) > 15)
    # around an eye the skin is blank skin outright: lid shading and lash
    # shadow left on it would show above the closed eye
    marks |= np.logical_or.reduce([dilate(feats['eye_'+s]['mask'], 4) for s in 'LR']) & ~dilate(above, 3)
    face_rgb = v5
    if have('V7-ears-clear') and have('V8-head-off'):
        # ears: v7 draws them whole (no strands over the top). Around where
        # it differs from v3, the skin-hued pixels beside the face are ear,
        # with the dark outline next to them
        v7 = V('V7-ears-clear')
        # (large changes only: the edit re-renders a few px here and there)
        cleared = dilate(keep_large(np.abs(v7-v3).max(2) > 25, 400), 15)
        ear = cleared & dilate(face_m, 20) & ~face_m & (figure_alpha(v7) > .5)
        ear &= hue_distance(v7, face_m, 10) < hue_distance(v7, hair_u & ~face_m, 10)
        ear |= dilate(ear | face_m, 2) & cleared & (lum(v7) < 150) & ~face_m
        ear = fill_holes(ear | face_m, 400) & ~face_m
        ear &= dilate(erode(ear | face_m, 2), 3)                  # a loose strand beside it is thin
        ear &= ~dilate(featured, 4)                                 # an eye corner is not ear
        ear = np.logical_or.reduce([c for c in components(ear, 30) if (dilate(c, 2) & face_m).any()] or [np.zeros(fig.shape, bool)])
        # (around the ear only: the eye corner stays blank skin)
        face_rgb = np.where((ear | (dilate(ear, 25) & face_m & ~featured))[..., None], v7, v5)
        face_m |= ear
        log(f'  耳朵补全：{int(ear.sum())} px')
    stack.add('Face_Skin', 'head', face_rgb, soft(face_m), keep=marks)

    # body: everything that is not head or neck
    head_any = S['hat'] | S['bang'] | S['side'] | face_m | neck | hair_u
    body = fig & ~head_any
    body = keep_large(body, 400) | (strays & ~head_any)
    # under the hair ends taken off it, the clothes are filled from around
    body_rgb = B.copy()
    if strays.any():
        body_rgb = cv2.inpaint(np.clip(B, 0, 255).astype(np.uint8), _u8(dilate(strays, 1)), 4, cv2.INPAINT_TELEA).astype(np.float32)
    under_dangles = None
    if have('V9-dangles-off'):
        # dangles (a pendant): what V9 took away, each piece its own layer over
        # the body; under it the body is V9's clothes, seen when it swings
        # (kept as V9 drew it, not solved from the base under the dangle's soft rim)
        v9 = V('V9-dangles-off')
        under_dangles = np.zeros(fig.shape, bool)
        gone, _ = change_mask(B, v9, min_area=150)
        pieces = sorted(components(gone & dilate(body, 2), 150), key=lambda c: _box(c)[0])
        for k, c in enumerate(pieces, 1):
            # its whole outline and soft edge: every pixel near it where V9 differs
            rgb, a = matte(B, v9, dilate(c, 6), grow=None)
            stack.add(f'Dangle_{k}', 'head' if _box(c)[1] < fy1 else 'body', rgb, a*fig_a)
            body_rgb = np.where(dilate(c, 6)[..., None], v9, body_rgb)
            under_dangles |= dilate(c, 6)
            log(f'  挂件 Dangle_{k}：{int(c.sum())} px')
    stack.add('Body', 'body', body_rgb, soft(body), keep=under_dangles)

    # neck: continued up behind the chin and down behind the collar
    # (from a few px below the chin: the rows touching it carry the chin line)
    neck_rgb, neck_m = extend_rows(v3, neck & ~dilate(face_m, 4), face_m | neck, up=int(.13*fh)+4)
    neck_rgb, neck_m = extend_rows(neck_rgb, neck_m, body, down=int(.06*fh))
    stack.add('Neck', 'neck', neck_rgb, soft(neck_m))

    # back hair, painted on behind the face and the neck top
    neck_top = neck_m & (np.mgrid[:fig.shape[0], :1][0] < fy1+.12*fh)
    # back hair: as drawn where it shows; under front hair, from v3 (front hair removed)
    long_from = rows(S['neck']).max() if S['hair'].any() and S['neck'].any() and rows(S['hair']).max() > rows(S['neck']).max() else None
    hair_rgb, hair_m = hair_behind(np.where((S['hair'] & ~front)[..., None], B, v3), hair_u, face_m, neck_top,
                                   behind=V('V8-head-off') if have('V8-head-off') else None, clothes=S['body'],
                                   long_from=long_from)
    stack.add('Hair_Back', 'head', hair_rgb, soft(hair_m))          # paper gaps stay open

    layers = stack.save(out)
    rest = rest_image(layers)
    err = np.abs(rest-B).max(2)*fig                            # the paper is not part of the model
    log(f'  静止合成与原图（人物范围）：最大差 {err.max():.1f}，平均 {err.mean():.3f}（0–255）')
    sheet(layers, B, out/'拆层总览.png')
    meta = dict(pad_top=pad_top, size=list(B.shape[1::-1]), layers=[l[0] for l in layers][::-1],
                rest_max_diff=round(float(err.max()), 2))
    (out/'split.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    return layers


def sheet(layers, base, path, cols=8, cell=220):
    """Every layer cropped to its bounding box on a checker board."""
    tiles = []
    for name, group, rgb, a, vis in reversed(layers):
        if not (a > .02).any():
            continue
        x0, y0, x1, y1 = _box(a > .02).astype(int)
        crop_a = a[y0:y1+1, x0:x1+1, None]; crop = rgb[y0:y1+1, x0:x1+1]
        h, w = crop_a.shape[:2]
        yy, xx = np.mgrid[:h, :w]
        board = np.where(((yy//8+xx//8) % 2)[..., None] == 0, 235., 205.)
        img = crop_a*crop+(1-crop_a)*board
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        im.thumbnail((cell, cell-24))
        tiles.append((name+('' if vis else '（隐藏）'), im))
    from PIL import ImageDraw, ImageFont
    rows = (len(tiles)+cols-1)//cols
    out = Image.new('RGB', (cols*cell, rows*cell), 'white')
    dr = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', 14)
    except OSError:
        font = None
    for i, (nm, im) in enumerate(tiles):
        x, y = i % cols*cell, i//cols*cell
        out.paste(im, (x+(cell-im.width)//2, y+22))
        dr.text((x+4, y+3), nm, fill=(60, 40, 40), font=font)
    out.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('art'); ap.add_argument('out')
    ap.add_argument('--pad-top', type=int, default=0)
    ap.add_argument('--hat', default='Hat')
    a = ap.parse_args()
    plan(a.art, a.out, a.pad_top, a.hat)


if __name__ == '__main__':
    main()
