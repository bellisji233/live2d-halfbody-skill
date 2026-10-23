"""Test set for split.plan: the standard variants made from a character
whose layers are already known, with the drift and noise an image model
adds to every edit.

  python -m rigkit.synth ART_DIR OUT_DIR [--hat LAYER] [--clean]

ART_DIR is a layered art folder (layout.json + parts/). OUT_DIR gets
A0-base.png and every variant in variants.ORDER, plus truth.json naming
which source layers each split layer should match.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
import cv2
from PIL import Image

from .split import PALETTE

FEATURE = ('Eye_', 'Iris_', 'Brow_', 'Nose', 'Mouth_')


def category(layer, hat):
    n = layer['name']
    if n == hat:
        return 'hat'
    if n.startswith('Hair_Bang'):
        return 'bang'
    if n.startswith('Hair_Side'):
        return 'side'
    if n.startswith('Hair_'):
        return 'hair'
    if n == 'Neck' or layer['group'] == 'neck':
        return 'neck'
    if layer['group'] in ('body',):
        return 'body'
    if layer['group'] == 'pose':
        return None
    return 'face'


class Layers:
    def __init__(self, art, hat):
        self.art = Path(art)
        self.layout = json.loads((self.art/'layout.json').read_text())
        self.items = []
        for l in self.layout['layers']:                       # bottom to top
            rgba = np.asarray(Image.open(self.art/l['texture']).convert('RGBA')).astype(np.float32)
            self.items.append(dict(name=l['name'], cat=category(l, hat), rgb=rgba[..., :3], a=rgba[..., 3]/255.,
                                   visible=l.get('visible', True)))
        self.size = (self.layout['height'], self.layout['width'])

    def default(self):
        return {it['name'] for it in self.items if it['visible'] and it['cat']}

    def composite(self, names):
        out = np.full(self.size+(3,), 255., np.float32)
        for it in self.items:
            if it['name'] in names:
                a = it['a'][..., None]
                out = a*it['rgb']+(1-a)*out
        return out

    def labels(self, names):
        """Category of the topmost layer that is more than half opaque."""
        lab = np.full(self.size, '', object)
        for it in self.items:
            if it['name'] in names:
                lab[it['a'] > .5] = it['cat']
        return lab

    def names(self, pred, names):
        return {n for n in names if pred(n)}


def perturb(img, rng, shift=2.5, warp=.8, blur=.5, noise=1.5, nearest=False):
    h, w = img.shape[:2]
    fx = cv2.GaussianBlur(rng.normal(size=(h//16+1, w//16+1)).astype(np.float32), (0, 0), 2.5)
    fy = cv2.GaussianBlur(rng.normal(size=(h//16+1, w//16+1)).astype(np.float32), (0, 0), 2.5)
    fx = cv2.resize(fx/(np.abs(fx).max()+1e-6), (w, h))*warp
    fy = cv2.resize(fy/(np.abs(fy).max()+1e-6), (w, h))*warp
    dx, dy = rng.uniform(-shift, shift, 2)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = cv2.remap(img, (xx+fx-dx).astype(np.float32), (yy+fy-dy).astype(np.float32), cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    if blur and not nearest:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    out = out*rng.uniform(.98, 1.02, 3)+rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('art'); ap.add_argument('out')
    ap.add_argument('--hat', default='Hat')
    ap.add_argument('--clean', action='store_true', help='no drift or noise')
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    L = Layers(a.art, a.hat)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    v0 = L.default()
    has = lambda pre: {n for n in (it['name'] for it in L.items) if n.startswith(pre)}
    cat = {it['name']: it['cat'] for it in L.items}
    v1 = {n for n in v0 if cat[n] != 'hat'}
    v2 = {n for n in v1 if cat[n] != 'side'}
    v3 = {n for n in v2 if cat[n] != 'bang'}
    eye_open = {n for n in v0 if n.startswith(('Eye_White', 'Eye_Open', 'Eye_Crease', 'Iris_', 'Eye_Lower'))}
    sets = {
        'A0-base': v0, 'V1-hat-off': v1, 'V2-side-off': v2, 'V3-bangs-off': v3,
        'V5-blank-face': {n for n in v3 if not n.startswith(FEATURE)},
        'V6-no-iris': {n for n in v0 if not n.startswith('Iris_')},
        'D5-eyes-closed': (v0-eye_open) | has('Eye_Closed_'),
        'D6-eyes-smile': (v0-eye_open) | has('Eye_Smile_'),
    }
    for nm, var in [('D1-mouth-open', 'Open'), ('D2-mouth-o', 'O'), ('D3-mouth-line', 'Line'), ('D4-mouth-frown', 'Frown')]:
        if f'Mouth_{var}' in cat:
            sets[nm] = (v0-{'Mouth_Closed'}) | {f'Mouth_{var}'}
    for nm, var in [('D7-brows-angry', 'Angry'), ('D8-brows-troubled', 'Troubled')]:
        if has(f'Brow_{var}_'):
            sets[nm] = (v3-has('Brow_Neutral_')) | has(f'Brow_{var}_')
    for nm, names in sets.items():
        img = L.composite(names)
        if nm != 'A0-base' and not a.clean:
            img = perturb(img, rng)
        Image.fromarray(img.round().astype(np.uint8)).save(out/f'{nm}.png')
    for nm, names, keep in [('S1-segment', v0, None), ('V4-skin-mask', v3, ('face', 'neck', 'hair'))]:
        img = L.composite(names)
        lab = L.labels(names)
        for k, col in PALETTE.items():
            if keep is None or k in keep:
                img[lab == k] = col
        if not a.clean:
            img = perturb(img, rng, warp=2., noise=4., nearest=True)
        Image.fromarray(img.round().astype(np.uint8)).save(out/f'{nm}.png')
    truth = {it['name']: it['cat'] for it in L.items if it['cat']}
    (out/'truth.json').write_text(json.dumps(dict(art=str(Path(a.art).resolve()), hat=a.hat, layers=truth),
                                             ensure_ascii=False, indent=2)+'\n')
    print(f'{len(sets)+2} images -> {out}')


if __name__ == '__main__':
    main()
