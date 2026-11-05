"""Check a character's layers before rigging.

  python -m rigkit.check_art CHARACTER_JSON

Writes CHARACTER_DIR/素材检查/ with a report (报告.md) and images:

  图层清单   standard layers present / missing, unknown names, canvas size
  露底       each moving layer is shifted as far as it moves; any spot inside
             the character that turns transparent is a hidden area nobody
             painted (layers behind it must continue under it)
  原画对照   the rest stack against the original illustration: lines that
             no layer carries, or extra lines, show up as differences
  夹带       layers with pieces far away from their main body (a brow
             carrying a hair strand, a cheek carrying the jaw line)
  透出背景   gaps a white page hides: the art has the figure (not page
             white) where no layer covers it at rest, and white clothes
             gone in a gesture; with the stack at rest and in each gesture
             on a cyan page (青底.png)
"""
from __future__ import annotations
import sys, json
import numpy as np
import cv2
from PIL import Image, ImageFont

from .config import Character
from .split import load, align, art_figure, figure_alpha

CYAN = (78, 205, 235)                   # a VTS-like page: white clothes and pale skin show their gaps

REQUIRED = ['Face_Skin', 'Neck', 'Hair_Back',
            'Eye_White_L', 'Eye_White_R', 'Iris_L', 'Iris_R', 'Eye_Open_L', 'Eye_Open_R',
            'Eye_Closed_L', 'Eye_Closed_R', 'Brow_Neutral_L', 'Brow_Neutral_R',
            'Mouth_Closed', 'Mouth_Open', 'Nose', 'Ear_L', 'Ear_R']
OPTIONAL = ['Eye_Smile_L', 'Eye_Smile_R', 'Eye_Lower_L', 'Eye_Lower_R', 'Eye_Crease_L', 'Eye_Crease_R',
            'Brow_Angry_L', 'Brow_Angry_R', 'Brow_Troubled_L', 'Brow_Troubled_R', 'Mouth_O', 'Mouth_Line',
            'Mouth_Frown', 'Blush_L', 'Blush_R', 'Cheek_Flush_L', 'Cheek_Flush_R', 'Hair_Side_L', 'Hair_Side_R', 'Hair_Bang', 'Hair_Front', 'Hair_Bun', 'Hair_Tail_L', 'Hair_Tail_R', 'Glasses',
            'Eye_Open_Over_L', 'Eye_Open_Over_R']       # lash tips drawn over the front hair
# how far each kind of layer moves (px on a ~335 px wide face), for the shift test
MOVES = [('Hair_Side_', 30., 8.), ('Hair_Bang', 14., 8.), ('Hair_Bun', 14., 10.),
         ('Eye_', 12., 6.), ('Iris_', 12., 6.), ('Brow_', 12., 8.), ('Mouth_', 12., 6.), ('Nose', 14., 6.)]


def _font(size=18):
    try:
        return ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', size)
    except OSError:
        return ImageFont.load_default()


def _shift(a, dx, dy):
    out = np.zeros_like(a)
    h, w = a.shape
    xs, xd = (slice(0, w-dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w+dx))
    ys, yd = (slice(0, h-dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h+dy))
    out[yd, xd] = a[ys, xs]
    return out


def _over(alphas):
    acc = np.zeros_like(alphas[0])
    for a in alphas:
        acc = a+acc*(1-a)
    return acc


def check(char_path):
    char = Character(char_path)
    out = char.dir/'素材检查'
    out.mkdir(exist_ok=True)
    layers = [l for l in char.layout['layers']]
    names = [l['name'] for l in layers]
    W, H = char.canvas
    report = ['# 素材检查', '']

    # ---- 图层清单 ----
    missing = [n for n in REQUIRED if n not in names and not (n == 'Hair_Back' and any(x.startswith('Hair_Back') for x in names))]
    known = set(REQUIRED+OPTIONAL)
    other_head = [n for n in names if n not in known and not n.startswith(('Pose_', 'Hair_Back', 'Dangle_'))
                  and next(l for l in layers if l['name'] == n)['group'] == 'head']
    bad_size = []
    rgba = {}
    for l in layers:
        im = Image.open(char.art/l['texture'])
        if im.size != (W, H):
            bad_size.append(f"{l['name']} {im.size[0]}×{im.size[1]}")
        rgba[l['name']] = np.asarray(im.convert('RGBA').resize((W, H)) if im.size != (W, H) else im.convert('RGBA'))
    report += ['## 图层清单', '']
    report.append(f'- 图层 {len(names)} 个，画布 {W}×{H}')
    report.append('- 缺少：'+('、'.join(missing) if missing else '无'))
    if other_head:
        report.append('- 头部非标准名（按脸皮跟随转头）：'+'、'.join(other_head))
    if bad_size:
        report.append('- 尺寸与画布不一致：'+'、'.join(bad_size))
    report.append('')

    # ---- 露底 ----
    visible = [l['name'] for l in layers if l.get('visible', True)]
    A = {n: rgba[n][:, :, 3].astype(np.float32)/255 for n in names}
    rest = _over([A[n] for n in visible])
    inside = cv2.erode((rest > .9).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    scale = 1.
    if 'Face_Skin' in A:
        ys, xs = np.where(A['Face_Skin'] > .5)
        scale = (xs.max()-xs.min())/335.
    base = np.asarray(Image.alpha_composite(Image.new('RGBA', (W, H), (255, 255, 255, 255)),
                                            _stack(rgba, visible)).convert('RGB'))
    marked = base.copy()
    rows = []
    for n in visible:
        move = next(((dx, dy) for p, dx, dy in MOVES if n.startswith(p)), None)
        if move is None:
            continue
        dx, dy = int(round(move[0]*scale)), int(round(move[1]*scale))
        hole = np.zeros((H, W), bool)
        for sx, sy in ((dx, 0), (-dx, 0), (0, dy), (0, -dy)):
            moved = [(_shift(A[m], sx, sy) if m == n else A[m]) for m in visible]
            hole |= inside & (_over(moved) < .5) & (A[n] > .5)
        hole = cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        if hole.sum() > 20:
            rows.append((n, int(hole.sum())))
            marked[hole] = (marked[hole]*.3+np.array([255, 0, 200])*.7).astype(np.uint8)
    Image.fromarray(marked).save(out/'露底.png')
    report += ['## 露底', '', '各部件按动起来的幅度挪动后，角色内部出现的透明处（`露底.png` 里品红色）。'
               '这些地方需要下面一层继续画过去。', '']
    report += [f'- {n}：{px} 像素' for n, px in sorted(rows, key=lambda r: -r[1])] or ['- 无']
    report.append('')

    # ---- 原画对照 ----
    ref_cfg = char.get('reference')
    ref_path = char.art_path(ref_cfg['image']) if ref_cfg else None
    report += ['## 原画对照', '']
    if ref_path and ref_path.exists():
        ref = np.asarray(Image.open(ref_path).convert('RGB')).astype(float)
        shift_y = ref_cfg.get('shift_y', 0)
        full = np.full((H, W, 3), 255.)
        full[shift_y:shift_y+ref.shape[0], :ref.shape[1]] = ref[:H-shift_y, :W]
        d = np.abs(full-base.astype(float)).max(2)
        lines = (d > 40) & inside
        lines = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
        diff = (base*.35+165).astype(np.uint8)
        diff[lines] = (220, 30, 60)
        Image.fromarray(diff).save(out/'原画对照.png')
        report.append(f'静止叠图与原画相差明显的像素 {int(lines.sum())} 个（`原画对照.png` 红色）。'
                      '红色成线的地方，是原画上有、图层里没有的线，或反过来。')
    else:
        report.append('未设置原画（character.json 的 reference），跳过。')
    report.append('')

    # ---- 透出背景 ----
    report += ['## 透出背景', '']
    groups = {l['name']: l['group'] for l in layers}
    poses = [n for n in names if groups.get(n) == 'pose']
    stacks = [('静止', visible)]+[(n.replace('Pose_', ''), [m for m in names if m == n or (
        m in visible and groups.get(m) not in ('body', 'pose'))]) for n in poses]
    tiles = []
    for label, stack in stacks:
        im = Image.alpha_composite(Image.new('RGBA', (W, H), CYAN+(255,)), _stack(rgba, stack)).convert('RGB')
        tiles.append((label, im.resize((W*480//H, 480))))
    sheet_im = Image.new('RGB', (sum(t[1].size[0] for t in tiles), 480), 'white')
    x = 0
    for _, im in tiles:
        sheet_im.paste(im, (x, 0))
        x += im.size[0]
    sheet_im.save(out/'青底.png')
    rows = []
    src, meta = char.art.parent, char.art/'split.json'
    if (src/'A0-base.png').exists() and meta.exists():
        pad = json.loads(meta.read_text()).get('pad_top', 0)
        B = load(src/'A0-base.png', pad)
        s1 = src/'S1-segment.png'
        fig = art_figure(B, align(load(s1, pad), B)[0])[0] if s1.exists() else figure_alpha(B)
        if fig.shape == rest.shape:
            paper = (B.min(2) > 240) & (np.ptp(B, 2) < 8)
            # not on or next to page white (the soft rim of a gap between strands)
            near_paper = cv2.dilate(paper.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            hole = cv2.morphologyEx(((fig > .5) & (rest < .5) & ~near_paper).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            # what runs off the bottom of the canvas (hands cut by the frame) is out of the picture
            n_, lab, st, _ = cv2.connectedComponentsWithStats(hole)
            hole = np.isin(lab, [i for i in range(1, n_) if st[i, 1]+st[i, 3] < H-4 and st[i, 4] >= 50])
            if hole.sum() > 20:
                rows.append(f'- 静止：原画有、图层没盖住 {int(hole.sum())} 像素')
    base_rgb = np.asarray(_stack(rgba, visible).convert('RGB')).astype(int)
    white = (rest > .5) & (base_rgb.min(2) > 235)
    for label, stack in stacks[1:]:
        gone = cv2.morphologyEx((white & (_over([A[n] for n in stack]) < .5)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        if gone.sum() > 20:
            rows.append(f'- {label}：静止时白色、手势时透明 {int(gone.sum())} 像素（手臂挪开的地方除外，其余是缺口）')
    report += rows or ['- 无']
    report += ['', '`青底.png`：静止和各手势放在青色背景上，白衣服、浅色皮肤处的缺口在白底上看不出。', '']

    # ---- 夹带 ----
    report += ['## 夹带', '']
    stray = []
    for n in names:
        a = (A[n] > .2).astype(np.uint8)
        k, lab, st, _ = cv2.connectedComponentsWithStats(a)
        if k <= 2:
            continue
        main = 1+np.argmax(st[1:, 4])
        big = st[main, 4]
        far = []
        mx, my = st[main, 0]+st[main, 2]/2, st[main, 1]+st[main, 3]/2
        for i in range(1, k):
            if i == main or st[i, 4] < max(30, .02*big):
                continue
            cx, cy = st[i, 0]+st[i, 2]/2, st[i, 1]+st[i, 3]/2
            if np.hypot(cx-mx, cy-my) > .6*max(st[main, 2], st[main, 3]):
                far.append(int(st[i, 4]))
        if far:
            stray.append((n, far))
    report += [f'- {n}：离主体较远的碎块 {len(f)} 处（{"、".join(str(x) for x in f)} 像素）' for n, f in stray] or ['- 无']
    report.append('')
    (out/'报告.md').write_text('\n'.join(report)+'\n')
    print(out/'报告.md')
    return out


def _stack(rgba, names):
    im = Image.new('RGBA', next(iter(rgba.values())).shape[1::-1], (0, 0, 0, 0))
    for n in names:
        im.alpha_composite(Image.fromarray(rgba[n]))
    return im


if __name__ == '__main__':
    check(sys.argv[1])
