"""Side-by-side checks of built models: pose sheets, zoomed zones, a motion
clip, and folded triangles. Native Core evaluates every frame.

  python -m rigkit.preview sheet CHARACTER_JSON OUT_DIR LABEL=RUNTIME [LABEL=RUNTIME ...]
  python -m rigkit.preview clip  CHARACTER_JSON OUT.mp4 LABEL=RUNTIME [LABEL=RUNTIME ...]
  python -m rigkit.preview flips RUNTIME [N]

sheet: 对比-<姿势>.png (head crop of each runtime + difference to the first)
and 放大-<姿势>.png (the fixed check zones, 3x). clip: head turn with hair
follow-through, nod, tilt, blink, mouth, breath, body follow, blush.
"""
from __future__ import annotations
from pathlib import Path
import argparse, os, subprocess, tempfile
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import Character
from .render import render
from .core import Model

BG = (255, 222, 226)               # VTS-like pink: shows pale seams
SMILE = dict(ParamEyeLOpen=0, ParamEyeROpen=0, ParamEyeSmile=1, ParamMouthOpenY=.6, ParamMouthVariant=1)
POSES = [
    ('静止', {}),
    ('右转', dict(ParamAngleX=30)),
    ('左转', dict(ParamAngleX=-30)),
    ('抬头', dict(ParamAngleY=30)),
    ('低头', dict(ParamAngleY=-30)),
    ('右上歪头', dict(ParamAngleX=30, ParamAngleY=30, ParamAngleZ=30, ParamBodyAngleZ=10)),
    ('左下歪头', dict(ParamAngleX=-30, ParamAngleY=-30, ParamAngleZ=-30, ParamBodyAngleZ=-10)),
    ('甩发', dict(ParamAngleX=30, ParamHairSideL=-1, ParamHairSideR=-1, ParamHairSideL2=-1, ParamHairSideR2=-1,
                ParamHairBang=-1, ParamHairBun=1, ParamHairBack=-1, ParamHairBack2=-1)),
    ('反向甩发', dict(ParamAngleX=-30, ParamHairSideL=1, ParamHairSideR=1, ParamHairSideL2=1, ParamHairSideR2=1,
                  ParamHairBang=1, ParamHairBun=-1, ParamHairBack=1, ParamHairBack2=1)),
    ('笑', dict(SMILE, ParamAngleX=15, ParamAngleY=15)),
    ('闭眼张嘴', dict(ParamEyeLOpen=0, ParamEyeROpen=0, ParamMouthOpenY=1)),
]


def manifest(runtime):
    return next(Path(runtime).glob('*.model3.json')).name


def _font(size):
    try:
        return ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', size)
    except OSError:
        return ImageFont.load_default()


def boxes(char):
    """Head crop and the fixed check zones, from the character's positions."""
    f = char['head']['face']; feats = char['head']['features']
    cx, cy = f['centre']; rx = f['rx']; up, down = f['ry_up'], f['ry_down']
    head = (int(cx-2*rx), int(cy-1.7*up), int(cx+2*rx), int(cy+1.4*down))
    bl, br = feats['brow_L'], feats['brow_R']
    zones = {
        '眉毛和刘海': (int(bl[0]-60), int(bl[1]-60), int(br[0]+60), int(bl[1]+30)),
        '左脸边': (int(cx-rx-30), int(cy), int(cx-rx+90), int(cy+down)),
        '右脸边': (int(cx+rx-90), int(cy), int(cx+rx+30), int(cy+down)),
        '下巴和脖子': (int(cx-90), int(cy+down-60), int(cx+90), int(cy+down+30)),
        '左太阳穴和耳朵': (int(cx-rx-50), int(cy-up*.8), int(cx-rx+80), int(cy+.6*down)),
        '右太阳穴和耳朵': (int(cx+rx-80), int(cy-up*.8), int(cx+rx+50), int(cy+.6*down)),
    }
    return head, zones


def sheet(char, out, labelled):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    head, zones = boxes(char)
    font, small = _font(22), _font(16)
    dangles = [d['param'] for d in char.get('dangles', {}).values()]
    # long hair, twin tails and dangles move below the head: the whole canvas in every pose
    below = bool(dangles) or any(lk.get('body_from') or n.startswith('Hair_Tail_') for n, lk in char['hair']['locks'].items())
    for label, p in POSES:
        if label in ('甩发', '反向甩发'):
            p = {**p, **{d: -1. if label == '甩发' else 1. for d in dangles}}
        ims = [render(rt, manifest(rt), p, None, BG).convert('RGB') for _, rt in labelled]
        tiles = [(im.crop(head), f'{name}  {label}') for (name, _), im in zip(labelled, ims)]
        if len(ims) > 1:
            d = np.abs(np.asarray(ims[0], int)-np.asarray(ims[-1], int)).max(2)
            heat = np.asarray(ims[0].convert('L').convert('RGB'), float)*.35+160
            hot = np.clip(d*6, 0, 255)
            heat[..., 0] = np.maximum(heat[..., 0], hot); heat[..., 1] *= 1-hot/400; heat[..., 2] *= 1-hot/400
            tiles.append((Image.fromarray(np.clip(heat, 0, 255).astype(np.uint8)).crop(head), '差异（红色，放大 6 倍）'))
        w, h = tiles[0][0].size
        sh = Image.new('RGB', (len(tiles)*(w+8)-8, h+40), 'white'); dr = ImageDraw.Draw(sh)
        for k, (im, cap) in enumerate(tiles):
            sh.paste(im, (k*(w+8), 40)); dr.text((k*(w+8)+8, 8), cap, fill=(90, 50, 40), font=font)
        sh.save(out/f'对比-{label}.png')
        if label == '静止' or below:
            # the whole canvas once: stray pixels far from the head show up here
            small_ims = [im.resize((im.width*2//5, im.height*2//5), Image.LANCZOS) for im in ims]
            sw, shh = small_ims[0].size
            whole = Image.new('RGB', (len(small_ims)*(sw+8)-8, shh+40), 'white'); dw = ImageDraw.Draw(whole)
            for k, (im, (name, _)) in enumerate(zip(small_ims, labelled)):
                whole.paste(im, (k*(sw+8), 40)); dw.text((k*(sw+8)+8, 8), f'{name}  整张画布'+('' if label == '静止' else f'  {label}'), fill=(90, 50, 40), font=font)
            whole.save(out/('整张画布.png' if label == '静止' else f'整张画布-{label}.png'))
        cols = []
        for zn, (x0, y0, x1, y1) in zones.items():
            crops = [im.crop((x0, y0, x1, y1)).resize(((x1-x0)*3, (y1-y0)*3), Image.LANCZOS) for im in ims]
            zh = crops[0].height
            col = Image.new('RGB', (crops[0].width, len(crops)*(zh+34)), 'white'); cd = ImageDraw.Draw(col)
            for k, (c, (name, _)) in enumerate(zip(crops, labelled)):
                cd.text((6, k*(zh+34)+4), f'{zn} · {name}' if k == 0 else name, fill=(90, 50, 40), font=small)
                col.paste(c, (0, k*(zh+34)+28))
            cols.append(col)
        zone = Image.new('RGB', (sum(c.width for c in cols)+10*(len(cols)-1), max(c.height for c in cols)), 'white')
        x = 0
        for c in cols:
            zone.paste(c, (x, 0)); x += c.width+10
        zone.save(out/f'放大-{label}.png')
    return out


# ------------------------------------------------------------------ clip
# head keys (time s, AngleX, AngleY, AngleZ)
HEAD_KEYS = [(0, 0, 0, 0), (.9, 30, 0, 0), (1.6, 30, 0, 0), (3.1, -30, 0, 0), (3.8, -30, 0, 0), (4.8, 0, 0, 0),
             (5.6, 0, 30, 0), (6.2, 0, 30, 0), (7.4, 0, -30, 0), (8.0, 0, -30, 0), (8.8, 0, 0, 0),
             (9.6, 0, 0, 30), (10.4, 0, 0, 30), (11.6, 0, 0, -30), (12.4, 0, 0, -30), (13.2, 0, 0, 0), (21., 0, 0, 0)]
SEGMENTS = [(0., '左右转头'), (4.8, '抬头低头'), (8.8, '歪头'), (13.2, '眨眼'), (15., '说话'), (17., '呼吸'), (19.5, '脸红')]
END = 21.
# hair springs: (angular frequency, damping ratio, output gain); tips trail their lock
SPRINGS = {'ParamHairSideL': (7.2, .48, 2.25), 'ParamHairSideR': (7.9, .5, 2.15),
           'ParamHairBang': (10.4, .57, 1.65), 'ParamHairBun': (8.8, .54, 1.35),
           'ParamHairBack': (6.4, .46, 1.9)}
PITCH_SPRING = (10, .62, 1.7)


def _interp(keys, t):
    for (ta, *a), (tb, *b) in zip(keys, keys[1:]):
        if t <= tb:
            q = max(0., (t-ta)/(tb-ta)); e = q*q*(3-2*q)
            return [x+(y-x)*e for x, y in zip(a, b)]
    return list(keys[-1][1:])


def schedule(char, fps):
    tips = {lk['param']: lk.get('tip') for lk in char['hair']['locks'].values()}
    mids = {lk['param']: lk.get('mid') for lk in char['hair']['locks'].values()}
    springs = {pid: SPRINGS.get(pid, (8.8, .54, 1.5)) for pid in tips}
    springs['ParamHairPitch'] = PITCH_SPRING
    for d in char.get('dangles', {}).values():
        springs[d['param']] = (9.5, .3, 1.8)
    states = {pid: dict(x=0., v=0., tip=0., tipV=0.) for pid in springs}
    bx = by = 0.
    k = 1-np.exp(-1/(fps*.3))
    frames = []
    for f in range(int(END*fps)+1):
        t = f/fps
        x, y, z = _interp(HEAD_KEYS, t)
        out = {}
        steps = max(1, int(np.ceil(240/fps))); h = 1/fps/steps
        for pid, (w, damping, gain) in springs.items():
            s = states[pid]
            goal = .65*y/30 if pid == 'ParamHairPitch' else .48*x/30+.3*z/30
            for _ in range(steps):
                s['v'] += (w*w*(goal-s['x'])-2*damping*w*s['v'])*h
                s['x'] += s['v']*h
                wt = w*.78
                s['tipV'] += (wt*wt*(s['x']-s['tip'])-2*(damping+.06)*wt*s['tipV'])*h
                s['tip'] += s['tipV']*h
            out[pid] = float(np.clip((.3*(s['x']-goal)+.7*(s['tip']-goal))*gain, -1, 1))
            if tips.get(pid):
                out[tips[pid]] = float(np.clip((s['tip']-s['x'])*gain*1.6, -1, 1))
            if mids.get(pid):
                # the middle segment bends half as far, between the lock and its tip
                out[mids[pid]] = float(np.clip((s['tip']-s['x'])*gain*.8, -1, 1))
        bx += (x/3-bx)*k; by += (y/3-by)*k
        blink = 1.
        for t0 in (13.6, 14.4):
            if t0 <= t < t0+.25:
                blink = abs((t-t0)/.125-1)
        talk = max(0., np.sin((t-15.)*11)*.5+.4) if 15. <= t < 17. else 0.
        breath = .5-.5*np.cos(2*np.pi*(t-17.)/2.5) if 17. <= t < 19.5 else 0.
        cheek = _interp([(0, 0), (19.5, 0), (20.1, 1), (END, 1)], t)[0]
        frames.append(dict(ParamAngleX=x, ParamAngleY=y, ParamAngleZ=z, ParamBodyAngleZ=z/3,
                           ParamBodyAngleX=bx, ParamBodyAngleY=by, ParamEyeLOpen=blink, ParamEyeROpen=blink,
                           ParamMouthOpenY=talk, ParamMouthForm=.6 if talk else 0., ParamBreath=breath,
                           ParamCheek=cheek, **out, _caption=[c for s, c in SEGMENTS if t >= s][-1]))
    return frames


def _job(args):
    index, params, labelled, crop, tmp = args
    cap = params.pop('_caption')
    tiles = []
    for label, rt in labelled:
        im = render(rt, manifest(rt), params, None, BG).crop(crop).convert('RGB')
        im = im.resize((int(im.width*.62)//2*2, int(im.height*.62)//2*2), Image.LANCZOS)
        ImageDraw.Draw(im).text((14, 10), label, fill=(120, 60, 50), font=_font(22))
        tiles.append(im)
    w, h = tiles[0].size
    sh = Image.new('RGB', (w*len(tiles)+8*(len(tiles)-1), h+44), 'white')
    for i, t in enumerate(tiles):
        sh.paste(t, (i*(w+8), 0))
    ImageDraw.Draw(sh).text((14, h+8), cap, fill=(60, 40, 40), font=_font(26))
    sh.save(f'{tmp}/{index:04d}.png')


def clip(char, out, labelled, fps=24):
    f = char['head']['face']
    cx, cy = f['centre']
    W, H = char.canvas
    half = int(2.5*f['rx'])
    crop = (max(0, int(cx-half)), max(0, int(cy-2.2*f['ry_up'])), min(W, int(cx+half)), min(H, int(cy+3.0*f['ry_down'])))
    frames = schedule(char, fps)
    with tempfile.TemporaryDirectory() as tmp:
        with Pool(max(1, (os.cpu_count() or 2)-1)) as pool:
            pool.map(_job, [(i, p, labelled, crop, tmp) for i, p in enumerate(frames)], chunksize=4)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{tmp}/%04d.png',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(out)], check=True)
    return out


# ------------------------------------------------------------------ flips
def flips(runtime, n=600, seed=16):
    """Folded triangles over random combinations of every parameter."""
    m = Model(str(next(Path(runtime).glob('*.moc3'))))
    m.reset_params(); m.update(); g = m.geometry()

    def area(P, t):
        a, b, c = P[t[:, 0]], P[t[:, 1]], P[t[:, 2]]
        return (b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1])-(c[:, 0]-a[:, 0])*(b[:, 1]-a[:, 1])
    rest = [area(P, t.astype(int)) for P, t in zip(g['positions'], g['triangles'])]
    rng = np.random.default_rng(seed)
    total, worst = 0, {}
    for _ in range(n):
        # arm poses are switches: random values in between are not a pose
        p = {i: float(rng.uniform(a, b)) for i, a, b in zip(m.param_ids, m.param_min, m.param_max)
             if not i.startswith('ParamPose')}
        m.reset_params(); m.set_params(p); m.update(); g = m.geometry()
        for i, (P, t) in enumerate(zip(g['positions'], g['triangles'])):
            if g['opacities'][i] <= 0 or not len(t):
                continue
            s = area(P, t.astype(int))
            bad = int(((np.sign(s) != np.sign(rest[i])) & (np.abs(rest[i]) > 1e-12)).sum())
            if bad:
                total += bad
                worst[m.drawable_ids[i]] = worst.get(m.drawable_ids[i], 0)+bad
    return total, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['sheet', 'clip', 'flips'])
    ap.add_argument('args', nargs='+')
    a = ap.parse_args()
    if a.mode == 'flips':
        total, worst = flips(a.args[0], int(a.args[1]) if len(a.args) > 1 else 600)
        print('folded triangles', total, worst)
        return
    char = Character(a.args[0])
    labelled = [tuple(x.split('=', 1)) for x in a.args[2:]]
    if a.mode == 'sheet':
        print(sheet(char, a.args[1], labelled))
    else:
        print(clip(char, a.args[1], labelled))


if __name__ == '__main__':
    main()
