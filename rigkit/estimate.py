"""New character: draft character.json from its layers, plus a marked image.

  python -m rigkit.estimate ART_DIR CHARACTER_DIR [--name 名字]

Measures the face, features, neck, waist and hair from the standard layers
(see the skill's layer list) and writes CHARACTER_DIR/character.json and
CHARACTER_DIR/标注.png: the rest image with every estimated position drawn
on it. Check the image; fix numbers in character.json where a mark is off.
An existing character.json is never overwritten (writes character.draft.json).

Amplitudes scale with the face size, from the values tuned on the first
character (face 335 px wide, 384 px tall).
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REF_FACE_W, REF_FACE_H = 335., 384.
BODY_CELLS = 40


def _alpha(art, layout, name):
    for l in layout['layers']:
        if l['name'] == name:
            return np.asarray(Image.open(art/l['texture']).convert('RGBA'))[:, :, 3]
    return None


def _bbox(a, thr=0):
    ys, xs = np.where(a > thr)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], float) if len(xs) else None


def _ink_centre(art, layout, name):
    """Centre of the dark lines of a layer (ignores a skin patch around them)."""
    for l in layout['layers']:
        if l['name'] == name:
            rgba = np.asarray(Image.open(art/l['texture']).convert('RGBA')).astype(float)
    ink = (rgba[:, :, 3] > 128) & (rgba[:, :, :3]@[.299, .587, .114] < 170)
    ys, xs = np.where(ink)
    return [round(float((xs.min()+xs.max())/2)), round(float((ys.min()+ys.max())/2))]


def _face_sides(fa, box):
    """Left and right of the face. Ears drawn on the face skin stick out of
    its side outline: over the middle of the face each side is fitted with
    a parabola, and an outward run longer than a tenth of the face height
    is an ear, left out. Without one the bounding box stands."""
    x0, y0, x1, y1 = box
    fh = y1-y0
    ys = np.arange(int(y0+.25*fh), int(y0+.85*fh))
    ys = ys[fa[ys].any(1)]
    out = [x0, x1]
    for k, edge, sign in ((0, lambda r: np.nonzero(r)[0].min(), -1), (1, lambda r: np.nonzero(r)[0].max(), 1)):
        e = np.array([edge(fa[y]) for y in ys], float)
        keep = np.ones(len(ys), bool)
        for _ in range(4):
            fit = np.polyval(np.polyfit(ys[keep], e[keep], 2), ys)
            keep = sign*(e-fit) < 6
        bump = ~keep
        runs = np.split(np.nonzero(bump)[0], np.nonzero(np.diff(np.nonzero(bump)[0]) > 1)[0]+1) if bump.any() else []
        if any(len(r) > .1*fh for r in runs):
            inner = np.where(keep, e, fit)
            out[k] = float(inner.min() if k == 0 else inner.max())
    return out


def _centre(b):
    return [round(float((b[0]+b[2])/2)), round(float((b[1]+b[3])/2))]


def estimate(art: Path, name: str):
    layout = json.loads((art/'layout.json').read_text())
    W, H = layout['width'], layout['height']
    names = [l['name'] for l in layout['layers']]
    A = lambda n: _alpha(art, layout, n)
    B = lambda n, thr=0: _bbox(A(n), thr) if n in names else None
    missing = [n for n in ('Face_Skin', 'Neck', 'Eye_White_L', 'Eye_White_R', 'Eye_Open_L', 'Eye_Open_R',
                           'Iris_L', 'Iris_R', 'Eye_Closed_L', 'Eye_Closed_R', 'Mouth_Closed', 'Mouth_Open')
               if n not in names]
    if missing:
        raise SystemExit('缺少图层：'+'、'.join(missing))
    face = B('Face_Skin', 128)
    face[0], face[2] = _face_sides(A('Face_Skin') > 128, face)
    eye_l, eye_r = _centre(B('Eye_White_L', 128)), _centre(B('Eye_White_R', 128))
    cy = (eye_l[1]+eye_r[1])/2
    cx = (face[0]+face[2])/2
    fw, fh = face[2]-face[0], face[3]-face[1]
    sw, sh = fw/REF_FACE_W, fh/REF_FACE_H
    chin = face[3]
    feats = {'eye_L': eye_l, 'eye_R': eye_r}
    for side in 'LR':
        b = B(f'Brow_Neutral_{side}', 64)
        feats['brow_'+side] = _centre(b) if b is not None else [feats['eye_'+side][0], round(cy-.23*fh)]
    nose = B('Nose', 32)
    mouth_closed = B('Mouth_Closed', 128)
    feats['nose'] = _centre(nose) if nose is not None else [round(cx), round((cy+mouth_closed[1])/2)]
    feats['mouth'] = _ink_centre(art, layout, 'Mouth_Closed')
    neck = B('Neck', 128)
    neck_cx = (neck[0]+neck[2])/2
    pivot = [round(float(neck_cx)), round(float(chin+.1*fh))]

    # chin line: the bottom edge of the face at the centre and either side
    fa = A('Face_Skin') > 128
    def bottom(x):
        ys = np.where(fa[:, int(round(x))])[0]
        return float(ys.max()) if len(ys) else float(chin)
    chin_v = [[round(neck_cx-.14*fw), round(bottom(neck_cx-.14*fw))], [round(neck_cx), round(bottom(neck_cx)-1)],
              [round(neck_cx+.14*fw), round(bottom(neck_cx+.14*fw))]]

    eye_half = round(float(np.mean([(B(f'Eye_White_{s}', 128)[2]-B(f'Eye_White_{s}', 128)[0])/2*1.04 for s in 'LR'])))
    mo = B('Mouth_Open', 0)
    mw = (mouth_closed[2]-mouth_closed[0])
    variants = {k: v for k, v in {'-2': 'Frown', '-1': 'Line', '1': 'O', '2': 'Open'}.items() if f'Mouth_{v}' in names}

    def three(lock, param, long=False):
        """Three segments, each a beat after the one above (rig._segment,
        vts.CHAIN); a long lock hangs on a longer pendulum."""
        lock.update(tip=param+'2', mid=param+'3', bend=[round(23.4*sw, 1), round(26*sw, 1)])
        if long:
            lock['chain'] = {'length': [9., 9., 8.]}
        return lock

    locks = {}
    top = face[1]
    # pitch: how far each lock bounces up and down on a nod (ParamHairPitch)
    for nm, param, swing, pitch in [('Hair_Side_L', 'ParamHairSideL', 37.5, 26.5), ('Hair_Side_R', 'ParamHairSideR', 37.5, 26.5),
                                    ('Hair_Bang', 'ParamHairBang', 17, 17.5), ('Hair_Bun', 'ParamHairBun', 10, 17.5)]:
        b = B(nm, 0)
        if b is None:
            continue
        if nm == 'Hair_Bun':
            root = [round(float(b[3]-14)), -round(float((b[3]-b[1])*.77))]
        elif nm == 'Hair_Bang':
            root = [round(float(top-16)), round(float(b[3]-top+16))]
        else:
            root = [round(float(top-16)), round(float(chin-top+16))]
        lock = dict(param=param, swing=round(swing*sw, 1), pitch=round(pitch*sh, 1), root=root)
        if 'Side' in nm:
            three(lock, param)
        locks[nm] = lock
    b = B('Hair_Back', 0)
    if b is not None and b[3] > chin:
        # the back hair hangs below the jaw: its ends swing behind the side hair
        y0 = round(float(face[1]+face[3])/2)
        locks['Hair_Back'] = three(dict(param='ParamHairBack', swing=round(27.5*sw, 1), pitch=round(17.5*sh, 1),
                                        root=[y0, round(float(b[3]-y0))], seam=False), 'ParamHairBack', long=True)
    for nm in ('Hair_Tail_L', 'Hair_Tail_R'):
        b = B(nm, 0)
        if b is None:
            continue
        # a twin tail swings from its hair tie, freer further down
        y0 = round(float(b[1]+.12*(b[3]-b[1])))
        param = 'ParamHair'+nm[5:].replace('_', '')
        locks[nm] = three(dict(param=param, swing=round(37.5*sw, 1), pitch=round(22*sh, 1), root=[y0, round(float(b[3]-y0))],
                               seam=False), param, long=True)
    if 'Hair_Tail_L' in locks and 'Hair_Back' in locks:
        # the tails swing; what hangs at the nape stays close to the head
        locks['Hair_Back']['swing'] = round(locks['Hair_Back']['swing']*.3, 1)
        for k in ('tip', 'mid', 'bend', 'chain'):       # its lower segments swing as far as any lock's
            locks['Hair_Back'].pop(k, None)
    head_layers = [n for n in names if n.startswith(('Hair_', 'Face_', 'Ear_'))]
    hb = [B(n, 0) for n in head_layers if B(n, 0) is not None]
    # whatever else turns with the head (a hat above the hair) needs the lattice
    # too: a mesh is cut where the lattice ends
    worn = [l['name'] for l in layout['layers'] if l['group'] == 'head' and l['name'] not in head_layers
            and not l['name'].startswith(('Eye_', 'Iris_', 'Brow_', 'Mouth_', 'Nose', 'Cheek_'))]
    hb = np.array(hb+[B(n, 8) for n in worn if B(n, 8) is not None])
    hx0, hy0 = hb[:, :2].min(0)-64; hx1, hy1 = hb[:, 2:].max(0)+64
    side_bottom = max([B(n, 0)[3] for n in ('Hair_Side_L', 'Hair_Side_R') if n in names] or [chin+30])

    body_layers = [l['name'] for l in layout['layers'] if l['group'] == 'body']
    bb = np.array([B(n, 0) for n in body_layers])
    belt = B('Belt', 0) if 'Belt' in names else None
    neck_y = chin+.22*fh
    shoulder_y = neck_y+.21*fh
    waist_y = float(belt[3]-8) if belt is not None else float(min(H-40, shoulder_y+1.1*fh))
    if 'Hair_Back' in locks and B('Hair_Back', 0)[3] > shoulder_y+.3*fh:
        # long hair: the head carries it at the jaw, the body from below the shoulders
        locks['Hair_Back']['body_from'] = [round(float(chin)), round(float(shoulder_y+.2*fh-chin))]
    radius = round(float((bb[:, 2].max()-bb[:, 0].min())/2*.78)) if len(bb) else round(.78*fw)
    bcx = round(float(neck_cx))
    # evenly spaced lines: on an uneven grid, points between two keyforms
    # bend where they cross a line and thin triangles fold
    lines = [round(float(v), 2) for v in np.linspace(-300, max(W, H)+300, BODY_CELLS+1)]
    xs = ys = lines

    cfg = {
        'name': name, 'model_id': 'Model', 'canvas': [W, H], 'art': '', 'prepared': 'prepared',
        'layers': {},
        'head': {
            'pivot': pivot, 'tilt_deg': 10,
            'face': {'centre': [round(float(cx)), round(float(cy))], 'rx': round(float(fw/2*.97)),
                     'ry_up': round(float(cy-face[1])), 'ry_down': round(float(chin-cy))},
            'silhouette': 2.0, 'yaw_px': round(38*sw, 1), 'pitch_px': [round(22.5*sh, 1), round(16*sh, 1)], 'rigid': .6,
            'features': feats,
            'nose_turn': {'strength': .33, 'show': .75, 'fade': .65},
        },
        'neck': {'layer': 'Neck', 'chin_v': chin_v, 'chin_v_clear': 5, 'tilt_band': [round(float(chin+.12*fh)), round(47*sh)],
                 'hair_hold': [2, round(45*sw)], 'hair_face_clear': [10, 15],
                 'behind_jaw_from': [round(float(chin-.15*fh)), round(50*sh)]},
        # half shut: the crease fades out and the lower lid waits until halfway (second character, accepted);
        # all but shut, the drawn closed eye takes over from the pressed lash (twin tails, accepted)
        'eyes': {'half_width': eye_half, 'closed_art': {'ink_below': 150}, 'iris_travel': [round(4*sw, 1), round(2.5*sh, 1)],
                 'crease_fade': [.4, .85], 'lower_from': .45, 'closed_swap': [.12, .15]},
        'brows': {'tail_x': {}, 'length': 82},
        'mouth': {
            'open_layer': 'Mouth_Open', 'curve': {'centre': round(float((mo[0]+mo[2])/2)), 'half': round(float((mo[2]-mo[0])/2))},
            'open_band': [round(float(mo[1]-15*sh)), round(float(mo[3]+26*sh))], 'open_centre': feats['mouth'][0],
            'open_half': round(float((mo[2]-mo[0])/2+30*sw)), 'open_feather': 20,
            'closed_line': [feats['mouth'][1], 9], 'variants': variants, 'open_from': .1,
            'form': {'centre': feats['mouth'], 'half': round(float(mw/2)), 'lift': round(5*sh, 1), 'drop': round(4*sh, 1)}},
        'hair': {
            'locks': locks, 'anchor_layer': 'Face_Ear_Contour', 'anchor_pin': [45, 55], 'seam': [3, 30], 'root_power': 1.35,
            'bend': {'centre_x': round(float(cx)), 'width': round(float(fw/2*1.47)), 'amount': .18},
            'lift': [-4.5, 2.5],
            'tips': {'from_y': round(float(cy-50*sh)), 'span': round(float(side_bottom-(cy-50*sh))), 'amount': round(26*sw, 1), 'lift': 4},
            'lattice': {'step': 32, 'x': [int(max(0, hx0)), int(min(W, hx1))+1], 'y': [int(max(0, hy0)), int(min(H, hy1))+1],
                        'extra_points': [v for v in feats.values()]+[pivot]}},
        'body': {'centre_x': bcx, 'neck_y': round(float(neck_y)), 'shoulder_y': round(float(shoulder_y)),
                 'waist_y': round(waist_y), 'radius': radius, 'turn': round(24*sw, 1), 'shift': round(6*sw, 1), 'side': .35,
                 'lean_y': round(12*sh, 1), 'lean_scale': .035, 'pivot': [bcx, round(waist_y+38*sh)], 'tilt_deg': .6,
                 'breath': {'head': round(5*sh, 1), 'shoulder': round(12*sh, 1), 'widen': .02, 'scale': .002},
                 'grid': {'xs': xs, 'ys': ys}},
    }
    for side in 'LR':
        b = B(f'Brow_Neutral_{side}', 64)
        if b is not None:
            cfg['brows']['tail_x'][side] = round(float(b[0]+5)) if side == 'L' else round(float(b[2]-6))
            cfg['brows']['length'] = round(float((b[2]-b[0])*.95))
    if 'Hair_Bun' in names:
        b = B('Hair_Bun', 0)
        cfg['head']['bun'] = {'layer': 'Hair_Bun', 'centre': _centre(b), 'radius': [round(float((b[2]-b[0])/2*.7)), round(float((b[3]-b[1])/2*.85))],
                              'turn': [.24, .16], 'shift': [round(-11*sw), round(9*sh)],
                              'blend_x': [round(float(b[0]+40)), 75], 'blend_y': [round(float(b[3]-35)), 70]}
        cfg['layers']['Hair_Bun'] = {'warp': 'Bun'}
    poses = [n for n in names if n.startswith('Pose_')]
    if poses:
        cfg['poses'] = {'layers': [[n, 'ParamPose'+n[5:]] for n in poses], 'priority': poses[::-1]}
    if any(n.startswith('Cheek_Flush_') for n in names):
        cfg['cheek'] = {'param': 'ParamCheek'}
    group = {l['name']: l['group'] for l in layout['layers']}
    for n in names:
        if n.startswith('Dangle_'):
            # hangs from the middle of its top edge
            ys, xs = np.nonzero(A(n) > 128)
            pivot = [round(float(xs[ys <= ys.min()+3].mean())), int(ys.min())]
            cfg.setdefault('dangles', {})[n] = {'param': 'Param'+n.replace('_', ''), 'pivot': pivot, 'deg': 14,
                                                'follows': 'head' if group[n] == 'head' else 'body'}
    cfg['vts'] = {'name': name, 'file_name': 'Model'}
    return cfg


def draw(art: Path, cfg, out: Path):
    """Rest image (all visible layers) with the estimated positions on it."""
    layout = json.loads((art/'layout.json').read_text())
    W, H = layout['width'], layout['height']
    img = Image.new('RGBA', (W, H), (255, 255, 255, 255))
    for l in layout['layers']:
        if l.get('visible', True):
            img.alpha_composite(Image.open(art/l['texture']).convert('RGBA'))
    img = img.convert('RGB')
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', 18)
    except OSError:
        font = ImageFont.load_default()
    red, blue, green, purple = (220, 40, 60), (40, 90, 220), (20, 150, 70), (150, 60, 190)
    h = cfg['head']; f = h['face']
    cx, cy = f['centre']
    d.arc([cx-f['rx'], cy-f['ry_up'], cx+f['rx'], cy+f['ry_up']], 180, 360, fill=blue, width=2)
    d.arc([cx-f['rx'], cy-f['ry_down'], cx+f['rx'], cy+f['ry_down']], 0, 180, fill=blue, width=2)
    d.text((cx+f['rx']+6, cy-10), '脸', fill=blue, font=font)
    labels = {'eye_L': '眼', 'eye_R': '眼', 'brow_L': '眉', 'brow_R': '眉', 'nose': '鼻', 'mouth': '嘴'}
    for k, (x, y) in h['features'].items():
        d.ellipse([x-5, y-5, x+5, y+5], outline=red, width=2)
        d.text((x+8, y-22), labels.get(k, k), fill=red, font=font)
    px, py = h['pivot']
    d.line([px-14, py, px+14, py], fill=purple, width=3); d.line([px, py-14, px, py+14], fill=purple, width=3)
    d.text((px+16, py-10), '歪头转轴', fill=purple, font=font)
    for (x, y) in cfg['neck']['chin_v']:
        d.ellipse([x-3, y-3, x+3, y+3], fill=purple)
    b = cfg['body']
    for key, label in (('neck_y', '颈根'), ('shoulder_y', '肩'), ('waist_y', '腰')):
        y = b[key]
        d.line([b['centre_x']-b['radius'], y, b['centre_x']+b['radius'], y], fill=green, width=2)
        d.text((b['centre_x']+b['radius']+6, y-10), label, fill=green, font=font)
    d.line([b['centre_x'], b['neck_y'], b['centre_x'], b['waist_y']], fill=green, width=1)
    bx, by = b['pivot']
    d.line([bx-14, by, bx+14, by], fill=green, width=3); d.line([bx, by-14, bx, by+14], fill=green, width=3)
    d.text((bx+16, by-10), '身体倾斜转轴', fill=green, font=font)
    hair_names = {'Hair_Side_L': '左侧发', 'Hair_Side_R': '右侧发', 'Hair_Bang': '刘海', 'Hair_Bun': '发髻', 'Hair_Back': '后发',
                  'Hair_Tail_L': '左马尾', 'Hair_Tail_R': '右马尾'}
    roots = {}
    for nm, lock in cfg['hair']['locks'].items():
        roots.setdefault(lock['root'][0], []).append(hair_names.get(nm, nm))
    for y0, nms in roots.items():                              # locks sharing a root line get one label
        d.line([cx-f['rx']*1.6, y0, cx+f['rx']*1.6, y0], fill=(200, 120, 0), width=1)
        d.text((cx-f['rx']*1.6, y0-20), '发根：'+'、'.join(nms), fill=(200, 120, 0), font=font)
    if 'bun' in h:
        bc, br = h['bun']['centre'], h['bun']['radius']
        d.ellipse([bc[0]-br[0], bc[1]-br[1], bc[0]+br[0], bc[1]+br[1]], outline=(200, 120, 0), width=2)
    img.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('art')
    ap.add_argument('character_dir')
    ap.add_argument('--name', default='新角色')
    a = ap.parse_args()
    art, out = Path(a.art).resolve(), Path(a.character_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = estimate(art, a.name)
    cfg['art'] = os.path.relpath(art, out)
    target = out/'character.json'
    if target.exists():
        target = out/'character.draft.json'
    target.write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+'\n')
    draw(art, cfg, out/'标注.png')
    print(target, out/'标注.png')


if __name__ == '__main__':
    main()
