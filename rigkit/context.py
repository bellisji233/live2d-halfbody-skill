"""Data the motion reads off the artwork: lid and mouth curves, hair weights,
distance maps, and the lattice hair motion is interpolated on.

Everything here is computed from the character's layers and character.json;
nothing is drawn or changed.
"""
from __future__ import annotations
import numpy as np
import cv2


def smooth(t):
    t = np.clip(t, 0, 1)
    return t*t*(3-2*t)


def sample(field, pts):
    """Nearest-pixel lookup of a canvas-sized map at canvas points."""
    h, w = field.shape[:2]
    ix = np.clip(np.rint(pts[:, 0]).astype(int), 0, w-1)
    iy = np.clip(np.rint(pts[:, 1]).astype(int), 0, h-1)
    return field[iy, ix]


def _alpha(path):
    return cv2.imread(str(path), -1)[:, :, 3]


class Context:
    def __init__(self, char):
        self.char = char
        W, H = char.canvas
        self.size = (W, H)
        self._eye_window = {}
        self._eyes()
        self._mouth()
        self._hair_weights()
        self._distance_maps()
        self.lattice = Lattice(char['hair']['lattice'], (W, H))

    # ---------------------------------------------------------------- eyes
    def _eyes(self):
        c = self.char
        e = c['eyes']
        half = e['half_width']
        closed_cfg = e.get('closed_art', {})
        ink_below = closed_cfg.get('ink_below', 150)
        self.eye_curves, self.lash_bands, self.crease_curves = {}, {}, {}
        for side in 'LR':
            centre = c['head']['features']['eye_'+side][0]
            if 'mask' in e:
                a = cv2.imread(str(c.art_path(e['mask'][side])), 0) > 128
            else:
                a = _alpha(c.part('Eye_White_'+side)) > 128      # the eye opening
            cols = np.where(a.any(axis=0))[0]
            upper = np.array([np.where(a[:, x])[0].min() for x in cols])
            lower = np.array([np.where(a[:, x])[0].max() for x in cols])
            u = (cols-centre)/half
            # the closed lash line: the lowest ink row per column of the
            # closed-eye drawing (a full image with a row range, or the layer)
            if 'image' in closed_cfg:
                r0, r1 = closed_cfg['rows']
                ink = np.zeros(a.shape, bool)
                ink[r0+closed_cfg['shift_y']:r1+closed_cfg['shift_y']] = \
                    cv2.imread(str(c.art_path(closed_cfg['image'])))[r0:r1, :, 2] < ink_below
            else:
                art = cv2.imread(str(c.part('Eye_Closed_'+side)), -1)
                ink = (art[:, :, 3] > 128) & (art[:, :, 2] < ink_below)
            # the closed lash only: a speck the edit left near it (below the
            # corner) would pull the closed line down there, and the lash with it
            n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
            if n > 2:
                big = st[1:, cv2.CC_STAT_AREA].max()
                ink = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= .3*big])
            closed, cx = [], []
            for x in cols:
                yy = np.where(ink[:, x])[0]
                if len(yy) > 4:
                    cx.append(x); closed.append(yy.max())
            pc = np.polyfit((np.array(cx)-centre)/half, closed, 2)
            self.eye_curves[side] = (np.polyfit(u, upper, 2), np.polyfit(u, lower, 2), pc)
            # vertical extent of the lash ink per column, smoothed
            art = cv2.imread(str(c.part('Eye_Open_'+side)), -1)
            ink = (art[:, :, 3] > 80) & (art[:, :, 2] < 175)
            xc = np.where(ink.any(axis=0))[0]
            top = np.array([np.where(ink[:, x])[0].min() for x in xc], float)
            bot = np.array([np.where(ink[:, x])[0].max() for x in xc], float)
            top = cv2.GaussianBlur(top[None, :], (0, 0), 4).ravel()
            bot = cv2.GaussianBlur(bot[None, :], (0, 0), 4).ravel()
            self.lash_bands[side] = (xc, top, bot)
            if not c.part('Eye_Crease_'+side).exists():
                continue                                          # crease layers are optional
            fold = cv2.imread(str(c.part('Eye_Crease_'+side)), -1)
            ink = (fold[:, :, 3] > 80) & (fold[:, :, 2] < 205)
            cols = np.where(ink.any(axis=0))[0]
            if len(cols) < 3:
                continue                                          # no line drawn in it
            cy = np.array([np.where(ink[:, x])[0].mean() for x in cols])
            self.crease_curves[side] = np.polyfit((cols-centre)/half, cy, 2)

    def eye_window(self, side):
        """Box shared by the lash and the eye mask (canvas px): both layers'
        pixels plus a margin; character.json may fix its rows (lid_window)."""
        if side not in self._eye_window:
            self._eye_window[side] = self._measure_eye_window(side)
        return self._eye_window[side]

    def _measure_eye_window(self, side):
        W, H = self.size
        boxes = []
        for nm in ('Eye_Open_'+side, 'Eye_White_'+side):
            ys, xs = np.where(_alpha(self.char.part(nm)) > 0)
            boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        b = np.array(boxes)
        x0, y0 = b[:, :2].min(0)-3
        x1, y1 = b[:, 2:].max(0)+4
        win = self.char['eyes'].get('lid_window')
        if win:
            y0, y1 = win
        return int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))

    def eye_contours(self, x, side):
        e = self.char['eyes']
        centre = self.char['head']['features']['eye_'+side][0]
        u = np.clip((x-centre)/e['half_width'], -1, 1)
        pu, pl, pc = self.eye_curves[side]
        upper = np.polyval(pu, u)
        lower = np.maximum(upper+10, np.polyval(pl, u))
        closed = np.polyval(pc, u)
        return upper, lower, closed

    # --------------------------------------------------------------- mouth
    def _mouth(self):
        m = self.char['mouth']
        mouth = _alpha(self.char.part(m['open_layer'])) > 128
        cols = np.where(mouth.any(axis=0))[0]
        upper = np.array([np.where(mouth[:, x])[0].min() for x in cols])
        lower = np.array([np.where(mouth[:, x])[0].max() for x in cols])
        u = (cols-m['curve']['centre'])/m['curve']['half']
        self.mouth_curves = (np.polyfit(u, upper, 4), np.polyfit(u, lower, 4))
        # the lips as drawn, column by column (lightly smoothed), soft rim
        # included: a curve fitted over the whole mouth runs inside the steep
        # sides at the corners, and the outline left outside it fans out into
        # streaks as the mouth closes
        drawn = _alpha(self.char.part(m['open_layer'])) > 16
        cols = np.where(drawn.any(axis=0))[0]
        top = np.array([np.where(drawn[:, x])[0].min() for x in cols])
        bottom = np.array([np.where(drawn[:, x])[0].max() for x in cols])
        smooth1 = lambda a: cv2.GaussianBlur(a.astype(np.float32)[None, :], (0, 0), 1.).ravel()
        self.mouth_edges = (cols.astype(np.float32), smooth1(top), smooth1(bottom))

    def mouth_contours(self, x):
        m = self.char['mouth']
        u = np.clip((x-m['curve']['centre'])/m['curve']['half'], -.98, .98)
        cols, up, lo = self.mouth_edges
        upper = np.interp(x, cols, up)
        lower = np.maximum(upper+3, np.interp(x, cols, lo))
        y0, k = m['closed_line']
        return upper, lower, y0-k*u*u

    # ---------------------------------------------------------------- hair
    def _hair_weights(self):
        """How freely each point of a lock swings: 0 at the root, at seams
        with the other locks and on the face contour; up to 1 at free ends."""
        h = self.char['hair']
        W, H = self.size
        masks = {nm: _alpha(self.char.part(nm)) > 0 for nm in h['locks']}
        anchor_pin = 1.
        anchor = self.char.part(h['anchor_layer'])
        if anchor.exists():
            ad = cv2.distanceTransform((_alpha(anchor) == 0).astype(np.uint8), cv2.DIST_L2, 5)
            a0, a1 = h['anchor_pin']
            anchor_pin = smooth((ad-a0)/a1)
        yy, xx = np.mgrid[:H, :W]
        s0, s1 = h['seam']
        self.hair_weights = {}
        # locks drawn behind the others (back hair) neither pin nor are pinned at seams
        seamed = [n for n in masks if h['locks'][n].get('seam', True)]
        for nm, mask in masks.items():
            seam = 1.
            if nm in seamed:
                others = np.logical_or.reduce([masks[n] for n in seamed if n != nm])
                d = cv2.distanceTransform((~others).astype(np.uint8), cv2.DIST_L2, 5)
                seam = smooth((d-s0)/s1)
            y0, length = h['locks'][nm]['root']
            root = np.clip((yy-y0)/length, 0, 1) if length > 0 else np.clip((y0-yy)/-length, 0, 1)
            # a lock lying on the face (bangs) is not held by the face anchor
            pin = anchor_pin if h['locks'][nm].get('anchor', True) else 1.
            self.hair_weights[nm] = seam*root**h['root_power']*pin

    # ------------------------------------------------------------ distances
    def _distance_maps(self):
        """Distance to the face (hair touching it moves with it) and to the
        fixed neck below the chin line (hair hanging there stays put)."""
        face = _alpha(self.char.part('Face_Skin')) > 128
        self.face_dist, _ = cv2.distanceTransformWithLabels((~face).astype(np.uint8), cv2.DIST_L2, 5,
                                                            labelType=cv2.DIST_LABEL_PIXEL)
        n = self.char['neck']
        neck = _alpha(self.char.part(n['layer'])) > 128
        yy, xx = np.mgrid[:neck.shape[0], :neck.shape[1]]
        neck &= yy >= self.chin_v(xx)+n['chin_v_clear']
        self.neck_dist = cv2.distanceTransform((~neck).astype(np.uint8), cv2.DIST_L2, 5)

    def chin_v(self, x):
        (xa, ya), (xb, yb), (xc, yc) = self.char['neck']['chin_v']
        return np.where(x <= xb, ya+(x-xa)*(yb-ya)/(xb-xa), yb+(x-xb)*(yc-yb)/(xc-xb))


class Lattice:
    """Points hair motion is evaluated on; mesh vertices take the motion of
    their lattice triangle (barycentric). Pieces cut from one lock then move
    exactly alike along the cut."""

    def __init__(self, cfg, size):
        step = cfg['step']
        gx, gy = np.meshgrid(np.arange(cfg['x'][0], cfg['x'][1], step), np.arange(cfg['y'][0], cfg['y'][1], step))
        pts = np.c_[gx.ravel(), gy.ravel()]
        extra = np.asarray(cfg.get('extra_points', []), float).reshape(-1, 2)
        self.points = np.unique(np.vstack([pts, extra]), axis=0)
        sub = cv2.Subdiv2D((0, 0, size[0]+1, size[1]+1))
        for p in self.points:
            sub.insert(tuple(map(float, p)))
        tri = sub.getTriangleList().reshape(-1, 3, 2)
        inds = ((tri[:, :, None, :]-self.points[None, None, :, :])**2).sum(3).argmin(2)
        self.cells = inds[np.max(np.abs(self.points[inds]-tri), axis=(1, 2)) < .001]
        self._cache = {}

    def coordinates(self, v):
        key = v.tobytes()
        if key not in self._cache:
            P, C = self.points, self.cells
            a, b, c = P[C[:, 0]], P[C[:, 1]], P[C[:, 2]]
            e = b-a; f = c-a; det = e[:, 0]*f[:, 1]-e[:, 1]*f[:, 0]
            d = v[:, None, :]-a
            u = (d[:, :, 0]*f[:, 1]-d[:, :, 1]*f[:, 0])/det
            w = (e[:, 0]*d[:, :, 1]-e[:, 1]*d[:, :, 0])/det
            score = np.minimum(np.minimum(u, w), 1-u-w)
            ii = score.argmax(1); jj = np.arange(len(v))
            weights = np.c_[1-u[jj, ii]-w[jj, ii], u[jj, ii], w[jj, ii]]
            self._cache[key] = (C[ii], weights)
        return self._cache[key]

    def interpolate(self, moved, v):
        ids, w = self.coordinates(np.ascontiguousarray(v, dtype=np.float64))
        return np.sum(moved[ids]*w[:, :, None], axis=1)
