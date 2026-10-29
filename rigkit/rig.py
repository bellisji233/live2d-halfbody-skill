"""The rig: parameter table, which parameters move each layer, and where every
vertex goes for any parameter values.

apply(name, vertices, p) is the single source of truth. The MOC writer
samples it at key values and stores the result as a deformer tree; the
preview and the checks call it directly.

Order inside apply, per layer:
  hair tip bend -> mouth corners -> own motion (lids, brows, mouth shapes,
  hair swing) -> head (surface turn, tilt about the neck) -> breath scale
  and body tilt about the waist.
Hair is evaluated on a lattice and interpolated, so pieces cut from one
lock move alike along every cut.
"""
from __future__ import annotations
import itertools
import numpy as np

from .config import Character
from .context import Context, sample, smooth
from .surface import Surface

HEAD = ['ParamAngleX', 'ParamAngleY', 'ParamAngleZ']
COMMON = ['ParamBodyAngleZ', 'ParamBreath']
EYE_OPEN_KEYS = [0., .08, .25, .5, .75, 1.]
MOUTH_OPEN_KEYS = [0., .04, .15, .4, 1.]


def mouth_open_keys(start=0.):
    """MOUTH_OPEN_KEYS starting at start instead of 0 (the mouth stays closed below it)."""
    return MOUTH_OPEN_KEYS if not start else [0.]+[round(start+k*(1-start), 4) for k in MOUTH_OPEN_KEYS]
LEGACY_POSE_KEYS = [0., .49, .5, 1.49, 1.5, 2.49, 2.5, 3.]
SWITCH_KEYS = [0., .49, .5, 1.]   # two keys around .5: no crossfade between poses
SIGNED = [-1., 0., 1.]
# a brow under the front hair shows through it at this share of its opacity
# (brows.through_bangs; 0: hidden as before)
THROUGH_BANGS = .4


def _rotate(v, deg, pivot):
    a = -np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return (v-pivot)@np.array([[c, s], [-s, c]])+pivot


class Rig:
    def __init__(self, char: Character, names):
        self.char = char
        self.layers = char.layers(names)
        self.ctx = Context(char)
        self.surface = Surface(char['head'])
        self.locks = char['hair']['locks']
        self.params = self._params()
        self.keys = {a: v for a, v, _ in self.params}
        self.default = {a: d for a, _, d in self.params}
        self.axes = {nm: self._axes(L) for nm, L in self.layers.items()}
        poses = char.get('poses', {})
        self.pose_layers = [nm for nm, _ in poses.get('layers', [])]
        self.pose_switch = dict(poses.get('layers', []))
        self.masks = {nm: ['Eye_Mask_'+L.side] for nm, L in self.layers.items()
                      if L.role in ('iris', 'eye_white') and 'Eye_Mask_'+L.side in self.layers}

    # ------------------------------------------------------------ parameters
    def _params(self):
        c = self.char
        P = [(a, [-30., 0., 30.], 0.) for a in HEAD]
        # with the drawn closed eye, keys at both ends of the swap: it is all but
        # instant, so no half-faded double lash shows while the eye is narrowed
        eye_keys = sorted(set(EYE_OPEN_KEYS)|{float(k) for k in c['eyes'].get('closed_swap', [])})
        P += [('ParamBodyAngleZ', [-10., 0., 10.], 0.), ('ParamBreath', [0., 1.], 0.),
              ('ParamEyeLOpen', eye_keys, 1.), ('ParamEyeROpen', eye_keys, 1.),
              ('ParamEyeSmile', [0., 1.], 0.), ('ParamEyeBallX', SIGNED, 0.), ('ParamEyeBallY', SIGNED, 0.),
              ('ParamBrowForm', SIGNED, 0.), ('ParamMouthOpenY', mouth_open_keys(c['mouth'].get('open_from', 0.)), 0.),
              ('ParamMouthVariant', sorted({0.}|{float(k) for k in c['mouth']['variants']}), 0.)]
        poses = c.get('poses', {})
        if poses.get('legacy_param'):
            P.append((poses['legacy_param'], LEGACY_POSE_KEYS, 0.))
        P += [(lk['param'], SIGNED, 0.) for lk in self.locks.values()]
        P.append(('ParamHairPitch', SIGNED, 0.))
        switch = dict(poses.get('layers', []))
        P += [(switch[nm], SWITCH_KEYS, 0.) for nm in poses.get('priority', [])]
        P += [('ParamBodyAngleX', [-10., 0., 10.], 0.), ('ParamBodyAngleY', [-10., 0., 10.], 0.)]
        P += [(lk['tip'], SIGNED, 0.) for lk in self.locks.values() if lk.get('tip')]
        P.append(('ParamMouthForm', SIGNED, 0.))
        if c.get('cheek'):
            P.append((c['cheek']['param'], [0., 1.], 0.))
        P += [(d['param'], SIGNED, 0.) for d in c.get('dangles', {}).values()]
        return P

    def _axes(self, L):
        if L.copy_of:
            return self._axes(self.layers[L.copy_of])
        own = []
        if L.role.startswith(('eye', 'iris')):
            own = ['ParamEye'+L.side+'Open', 'ParamEyeSmile']
            if L.role == 'iris':
                own += ['ParamEyeBallX', 'ParamEyeBallY']
        elif L.role == 'brow':
            own = ['ParamBrowForm']
        elif L.role == 'mouth':
            own = ['ParamMouthOpenY', 'ParamMouthVariant', 'ParamMouthForm']
        elif L.role == 'hair' and L.lock:
            lk = self.locks[L.lock]
            own = [lk['param']]+([lk['tip']] if lk.get('tip') else [])+['ParamHairPitch']
        elif L.role == 'cheek':
            own = [self.char['cheek']['param']]
        elif L.role == 'dangle':
            own = [self.char['dangles'][L.name]['param']]
            if L.group == 'body':
                return own+COMMON
        if L.group in ('body', 'pose'):
            poses = self.char.get('poses', {})
            legacy = [poses['legacy_param']] if poses.get('legacy_param') else []
            switch = dict(poses.get('layers', []))
            return legacy+COMMON+[switch[nm] for nm in poses.get('priority', [])]
        return own+(HEAD if L.group in ('head', 'neck') else [])+COMMON

    def combos(self, axes):
        """Every key combination of axes (first axis fastest)."""
        return [dict(zip(axes, reversed(v))) for v in itertools.product(*[self.keys[a] for a in reversed(axes)])]

    @property
    def groups(self):
        return {nm: L.group for nm, L in self.layers.items()}

    @property
    def swing_loose(self):
        return {nm for nm, L in self.layers.items() if L.role == 'hair' and L.lock}

    # ---------------------------------------------------------------- motion
    def through_opacity(self):
        return float(self.char['brows'].get('through_bangs', THROUGH_BANGS))

    def apply(self, name, vertices, p, visible=True):
        L = self.layers[name]
        if L.copy_of:
            pts, op = self.apply(L.copy_of, vertices, p, visible)
            return pts, op*self.through_opacity()
        p = {**self.default, **p}
        if L.role == 'hair':
            moved, op = self._apply(L, self.ctx.lattice.points, p, visible)
            return self.ctx.lattice.interpolate(moved, vertices), op
        pts, op = self._apply(L, vertices, p, visible)
        if L.role in ('nose', 'nose_turn'):
            n = self.char['head']['nose_turn']
            x = p['ParamAngleX']/30*n['strength']
            if L.role == 'nose_turn':
                op = n['show']*max(0, x if L.side == 'R' else -x)
            else:
                op *= 1-n['fade']*abs(x)
        return pts, op

    def _apply(self, L, vertices, p, visible):
        v = np.asarray(vertices, float)
        lock = self.locks.get(L.lock) if L.lock else None
        if lock and lock.get('tip') and p.get(lock['tip'], 0):
            v = self._tip(v, sample(self.ctx.hair_weights[L.lock], v), p[lock['tip']])
        if L.role == 'mouth' and p.get('ParamMouthForm', 0):
            v = self._mouth_form(v, p['ParamMouthForm'])
        v, op = self._own(L, v, p, visible)
        if L.group in ('head', 'neck'):
            v = self._head(L, v, p)
        b = self.char['body']
        pivot = np.array(b['pivot'], float)
        v[:, 1] = pivot[1]+(v[:, 1]-pivot[1])*(1+b['breath']['scale']*p.get('ParamBreath', 0))
        v = _rotate(v, p.get('ParamBodyAngleZ', 0)*b['tilt_deg'], pivot)
        if L.role == 'cheek':
            op *= float(p.get(self.char['cheek']['param'], 0))
        return v, float(op)

    # own parameters ------------------------------------------------------
    def _own(self, L, vertices, p, visible):
        v = vertices.copy()
        op = 1. if visible else 0.
        if L.group in ('body', 'pose') and L.role != 'dangle':
            pose = int(np.clip(np.floor(self._pose(p)+.5), 0, len(self.pose_layers)))
            selected = self.pose_layers[pose-1] if pose else None
            op = float(L.name == selected) if L.group == 'pose' else float(pose == 0)
        if L.role.startswith(('eye', 'iris')):
            v, op = self._eye(L, v, p)
        elif L.role == 'brow':
            v, op = self._brow(L, v, p)
        elif L.role == 'mouth':
            v, op = self._mouth(L, v, p, op)
        if L.role == 'hair' and L.lock:
            v = self._swing(L, vertices, v, p)
        elif L.role == 'dangle':
            d = self.char['dangles'][L.name]
            v = _rotate(v, d['deg']*p.get(d['param'], 0), np.array(d['pivot'], float))
        return v, op

    def _pose(self, p):
        """Arm pose: a switch at >= .5 wins, in priority order."""
        poses = self.char.get('poses', {})
        for nm in poses.get('priority', []):
            if p.get(self.pose_switch[nm], 0) >= .5:
                return self.pose_layers.index(nm)+1
        return p.get(poses.get('legacy_param'), 0) if poses.get('legacy_param') else 0

    def _eye(self, L, v, p):
        side = L.side
        ctx = self.ctx
        opening = p.get('ParamEye'+side+'Open', 1.)
        smile = p.get('ParamEyeSmile', 0.)
        x = v[:, 0]
        # eyes.closed_swap [lo, hi]: closing past hi, the drawn closed eye fades in
        # over the lash pressed down to it; below lo it shows alone (a lash
        # pressed flat frays: its strokes and pale fringe squeeze into jags)
        lo_hi = self.char['eyes'].get('closed_swap')
        keep = float(np.clip((opening-lo_hi[0])/(lo_hi[1]-lo_hi[0]), 0, 1)) if lo_hi else 1.
        if L.role == 'eye_lash':
            # the painted lash travels down to the closed line, keeping its ink
            xc, top, bot = ctx.lash_bands[side]
            st = np.interp(x, xc, top); sb = np.interp(x, xc, bot)
            _, _, closed = ctx.eye_contours(x, side)
            scale = np.minimum(1, 8/np.maximum(6, sb-st))
            target = closed+(v[:, 1]-sb)*scale
            v[:, 1] = v[:, 1]*opening+target*(1-opening)
            return v, (1-smile)*keep
        e = self.char['eyes']
        # the lower lid rises only once the eye is this far closed (1: all the way)
        low = min(1., opening/e.get('lower_from', 1.))
        if L.role == 'eye_mask':
            upper, lower, closed = ctx.eye_contours(x, side)
            nu = closed+(upper-closed)*opening
            nl = closed+1+(lower-closed-1)*low
            v[:, 1] = self._eyelid_warp(v[:, 1], upper, lower, nu, nl, side)
            return v, 0.
        if L.role == 'eye_white':
            return v, np.clip(opening/.08, 0, 1)*(1-smile)*keep        # gone once the drawn closed eye shows
        if L.role == 'eye_lower':
            upper, lower, closed = ctx.eye_contours(x, side)
            v[:, 1] += (closed-lower)*(1-low)
            return v, np.clip(opening/.5, 0, 1)*(1-smile)
        if L.role == 'eye_crease':
            centre = self.char['head']['features']['eye_'+side][0]
            fold = np.polyval(ctx.crease_curves[side], (x-centre)/e['half_width'])
            _, _, closed = ctx.eye_contours(x, side)
            v[:, 1] += (closed-12-fold)*(1-opening)
            # the fold fades as the lid comes down (by default only at the very end)
            lo, hi = e.get('crease_fade', [0., .08])
            return v, float(np.clip((opening-lo)/(hi-lo), 0, 1))*(1-smile)
        if L.role == 'iris':
            tx, ty = self.char['eyes']['iris_travel']
            v[:, 0] += tx*p.get('ParamEyeBallX', 0)
            v[:, 1] += ty*p.get('ParamEyeBallY', 0)
            return v, np.clip(opening/.08, 0, 1)*(1-smile)*keep
        if L.role == 'eye_closed':
            xc, top, bot = ctx.lash_bands[side]
            sb = np.interp(x, xc, bot)
            _, _, closed = ctx.eye_contours(x, side)
            v[:, 1] += (sb-closed)*opening
            return v, (1-smile)*(1-keep) if lo_hi else 0.
        if L.role == 'eye_smile':
            return v, smile
        raise ValueError(L.role)

    def _eyelid_warp(self, y, upper, lower, nu, nl, side):
        """Move the lid opening; the lash bands keep their thickness and the
        window around the eye stays put."""
        _, top, _, bottom = self.ctx.eye_window(side)
        out = y.copy()
        anchors = [np.full_like(y, top), upper-30, upper, lower, lower+5, np.full_like(y, bottom)]
        targets = [anchors[0], nu-30, nu, nl, nl+5, anchors[-1]]
        for a, b, c, d in zip(anchors, anchors[1:], targets, targets[1:]):
            mask = (y >= a) & (y <= b)
            out[mask] = (c+(y-a)*(d-c)/np.maximum(b-a, 1))[mask]
        return out

    def _brow(self, L, v, p):
        f = p.get('ParamBrowForm', 0)
        op = float(L.variant == 'Neutral')
        if op:
            # one painted brow bends from the tail toward the inner end
            b = self.char['brows']
            left = L.side == 'L'
            tail_x = b['tail_x'][L.side]
            t = np.clip((v[:, 0]-tail_x)/b['length'] if left else (tail_x-v[:, 0])/b['length'], 0, 1)
            tail = np.clip((t-.2)/.35, 0, 1); tail = tail*tail*(3-2*tail)
            if f < 0:
                v[:, 0] += (-f)*(22*t)*tail*(1 if left else -1)
                v[:, 1] += (-f)*(-3+43*t-4*t*t)*tail
            else:
                v[:, 1] += f*(12-43*t+3*np.sin(np.pi*t))*tail
        return v, op

    def _mouth(self, L, v, p, op):
        m = self.char['mouth']
        # below open_from the mouth stays closed: a webcam reads a closed mouth
        # as slightly open, and a barely open mouth reads as a crooked one
        start = m.get('open_from', 0.)
        o = max(0., p.get('ParamMouthOpenY', 0)-start)/(1-start)
        variant = p.get('ParamMouthVariant', 0)
        selected = {int(k): s for k, s in m['variants'].items()}.get(int(round(variant)))
        tag = L.variant
        if selected:
            op = float(tag == selected)
        else:
            op = (1-min(o/.04, 1)) if tag == 'Closed' else (min(o/.04, 1) if tag == 'Open' else 0.)
        if tag == 'Open' and not selected:
            # the open mouth grows from the closed line: the whole layer (the mouth
            # alone) is scaled toward it, so the lips land on it as the mouth closes.
            # Fitted to each column's own lips instead, the steep sides at the
            # corners shear and the outline fans out into streaks
            _, _, closed = self.ctx.mouth_contours(v[:, 0])
            top, bottom = m['open_band']
            warped = closed+(v[:, 1]-closed)*o
            w = np.clip((m['open_half']-np.abs(v[:, 0]-m['open_centre']))/m['open_feather'], 0, 1)
            inside = (v[:, 1] > top) & (v[:, 1] < bottom)
            v[inside, 1] += (warped-v[:, 1])[inside]*w[inside]
        return v, op

    def _mouth_form(self, v, form):
        """Corners up (form 1) or down (-1); the middle of the lips stays."""
        f = self.char['mouth']['form']
        cx, cy = f['centre']
        u = np.abs(v[:, 0]-cx)/f['half']
        corner = np.where(u <= 1, u*u, np.clip(1-(u-1)/.5, 0, 1))
        near = 1-smooth((np.abs(v[:, 1]-cy)-12)/18)
        out = v.copy()
        out[:, 1] -= (f['lift'] if form > 0 else f['drop'])*form*corner*near
        return out

    # hair ----------------------------------------------------------------
    def _swing(self, L, vertices, v, p):
        h = self.char['hair']
        lk = self.locks[L.lock]
        w = sample(self.ctx.hair_weights[L.lock], vertices)
        swing = p.get(lk['param'], 0)
        bend_cfg = h['bend']
        phase = (vertices[:, 0]-bend_cfg['centre_x'])/bend_cfg['width']
        bend = 1+bend_cfg['amount']*np.sin(phase*np.pi)
        v[:, 0] += lk['swing']*w*swing*bend
        side_lift, sag = h['lift']
        v[:, 1] += (side_lift*phase*swing+sag*swing*swing)*w
        v[:, 1] += lk['pitch']*p.get('ParamHairPitch', 0)*w
        return v

    def _tip(self, v, w, tip):
        """The lower part of a lock bends on its own (second physics segment)."""
        h = self.char['hair']
        t = h['tips']
        s = smooth((v[:, 1]-t['from_y'])/t['span'])
        wt = w*s
        phase = (v[:, 0]-h['bend']['centre_x'])/h['bend']['width']
        out = v.copy()
        out[:, 0] += t['amount']*wt*tip*(1+h['bend']['amount']*np.sin(phase*np.pi))
        out[:, 1] -= t['lift']*tip*tip*wt
        return out

    # head ----------------------------------------------------------------
    def _head(self, L, v, p):
        head = self.char['head']
        pivot = np.array(head['pivot'], float)
        projected = self._turn(L, v, p['ParamAngleX']/30, p['ParamAngleY']/30)
        transformed = _rotate(projected, p.get('ParamAngleZ', 0)*head['tilt_deg']/30, pivot)
        lock = self.locks.get(L.lock) if L.lock else None
        if lock and lock.get('body_from'):
            # long hair: below the jaw it is handed over to the body
            y0, band = lock['body_from']
            t = smooth((v[:, 1]-y0)/band)
            return transformed+(v-transformed)*t[:, None]
        if L.group == 'neck':
            # the neck top follows the head tilt, its base stays on the body
            y0, band = self.char['neck']['tilt_band']
            t = smooth((y0-v[:, 1])/band)
            return projected+(transformed-projected)*t[:, None]
        return transformed

    def _turn(self, L, v, yaw, pitch):
        if L.role == 'neck' or (yaw == 0 and pitch == 0):
            return v.copy()
        f = self.char['head']['features']
        centre = None
        if L.role.startswith(('eye', 'iris')):
            centre = f['eye_'+L.side]
        elif L.role == 'brow':
            centre = f['brow_'+L.side]
        elif L.role == 'mouth':
            centre = f['mouth']
        elif L.role in ('nose', 'nose_turn'):
            centre = f['nose']
        out = self.surface.feature(v, centre, yaw, pitch) if centre else self.surface.move(v, yaw, pitch)
        if L.role == 'hair' and L.neck_hold != 'none':
            # strands hanging beside the neck stay with the (fixed) neck
            n = self.char['neck']
            dn = sample(self.ctx.neck_dist, v)
            h0, h1 = n['hair_hold']
            c0, c1 = n['hair_face_clear']
            w = np.clip(1-(dn-h0)/h1, 0, 1)
            w = w*w*(3-2*w)*np.clip((sample(self.ctx.face_dist, v)-c0)/c1, 0, 1)
            if L.neck_hold == 'behind_jaw':
                # hair behind the jaw beside the neck top stays with the neck
                # even where it touches the face
                near = np.clip(1-(dn-h0)/h1, 0, 1)
                y0, band = n['behind_jaw_from']
                low = np.clip((v[:, 1]-y0)/band, 0, 1)
                w = np.maximum(w, near*near*(3-2*near)*low*low*(3-2*low))
            out = out+(v-out)*w[:, None]
            bun = self.char['head'].get('bun')
            if bun and L.name == bun['layer']:
                out = self.surface.bun_sphere(v, out, yaw, pitch)
        return out

    # body ----------------------------------------------------------------
    @property
    def body_extra(self):
        return ['ParamBodyAngleX', 'ParamBodyAngleY']

    @property
    def body_grid(self):
        g = self.char['body']['grid']
        return dict(xs=np.asarray(g['xs'], float), ys=np.asarray(g['ys'], float))

    def body_field(self, pts, values):
        """Body turn (X), lean (Y) and breath on rest canvas points. Above
        the neck base it is a pure translation (the head rides along); the
        torso turns like a cylinder; the waist and below stay put."""
        b = self.char['body']
        p = np.asarray(pts, float)
        x = values.get('ParamBodyAngleX', 0.)/10
        y = values.get('ParamBodyAngleY', 0.)/10
        neck, shoulder, waist = b['neck_y'], b['shoulder_y'], b['waist_y']
        h = smooth((waist-p[:, 1])/(waist-neck))
        q = smooth((p[:, 1]-neck)/(shoulder-neck))
        u = np.clip((p[:, 0]-b['centre_x'])/b['radius'], -1, 1)
        profile = (1-q)+q*(b['side']+(1-b['side'])*np.sqrt(1-u*u))
        dx = h*x*(b['turn']*profile+b['shift'])-(p[:, 0]-b['centre_x'])*b['lean_scale']*y*h*q
        dy = -h*y*b['lean_y']
        out = p+np.stack([dx, dy], 1)
        s = values.get('ParamBreath', 0.)
        if not s:
            return out
        br = b['breath']
        yy = p[:, 1]
        hb = smooth((waist-yy)/(waist-shoulder))
        rise = np.where(yy < shoulder, br['head']+(br['shoulder']-br['head'])*q, br['shoulder']*hb)
        out = out.copy()
        out[:, 1] -= s*rise
        out[:, 0] += s*br['widen']*(p[:, 0]-b['centre_x'])*q*hb
        return out


