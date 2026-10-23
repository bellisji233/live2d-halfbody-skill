"""Head turn: one continuous surface moves every head layer (Live2D XY method).

* X moves only horizontally, Y only vertically.
* The face is a dome: displacement is largest on the centre line and falls
  off toward the contour; the far side compresses, the near side widens.
  The near contour stays by the ear, so the near cheek shows more skin.
* The axis is the neck: chin and nose tip move most; the crown barely moves
  and the back-hair silhouette moves slightly against the turn.
* Eyes, brows, nose and mouth move as one piece each, riding the surface.

Displacement at the full key (ParamAngle +-30):
  dx = yaw_px * P(x, y) * sign(yaw),  dy = -pitch_px * Q(x, y) * pitch
P and Q are smooth functions of an ellipse coordinate fitted to the face.
"""
import numpy as np


class Surface:
    def __init__(self, head):
        f = head['face']
        self.cx, self.cy = f['centre']
        self.rx, self.ry_up, self.ry_down = f['rx'], f['ry_up'], f['ry_down']
        self.r_sil = head['silhouette']
        self.a = head['yaw_px']
        self.b_up, self.b_down = head['pitch_px']
        self.rigid = head['rigid']
        self.nose = np.array(head['features']['nose'], float)
        self.mouth = np.array(head['features']['mouth'], float)
        self.bun = head.get('bun')

    def _ellipse(self, v, s):
        dx = (v[:, 0]-self.cx)/self.rx
        dy = (v[:, 1]-self.cy)/np.where(v[:, 1] < self.cy, self.ry_up, self.ry_down)
        r = np.hypot(dx, dy)
        # alpha = 0 on the far side, pi on the near side, +pi/2 at the top
        return r, np.arctan2(-dy, s*dx)

    @staticmethod
    def _gauss(v, centre, sigma):
        d = v-np.asarray(centre)
        return np.exp(-(d*d).sum(1)/(2*sigma*sigma))

    def _surface(self, r, edge, back):
        dome = 1-np.minimum(r, 1)**3
        inside = edge+(1-edge)*dome
        t = np.clip((r-1)/(self.r_sil-1), 0, 1)
        t = t*t*(3-2*t)
        return np.where(r <= 1, inside, edge+(back-edge)*t)

    def yaw_profile(self, v, s):
        r, a = self._ellipse(v, s)
        c, sn = np.cos(a), np.sin(a)
        chin = np.exp(-((np.angle(np.exp(1j*(a+np.pi/2))))/.45)**2)
        edge = .3+.25*np.maximum(0, c)+.45*np.maximum(0, sn)**1.5+.7*chin
        back = -.2*c*c
        p = self._surface(r, edge, back)
        p += .22*self._gauss(v, self.nose, 22)+.08*self._gauss(v, self.mouth, 30)
        return p

    def pitch_profile(self, v):
        r, a = self._ellipse(v, 1)
        c, sn = np.cos(a), np.sin(a)
        edge = .38+.22*np.maximum(0, -sn)+.07*np.maximum(0, sn)
        back = .1*c*c-.35*np.maximum(0, sn)**2-.12*np.maximum(0, -sn)**2
        q = self._surface(r, edge, back)
        q += .3*self._gauss(v, self.nose, 22)+.15*self._gauss(v, self.mouth, 30)
        return q

    def field(self, v, yaw, pitch):
        """Pixel displacement of rest points v for yaw/pitch in [-1, 1]."""
        v = np.asarray(v, float)
        d = np.zeros_like(v)
        if yaw:
            d[:, 0] = self.a*abs(yaw)*np.sign(yaw)*self.yaw_profile(v, np.sign(yaw))
        if pitch:
            d[:, 1] = -(self.b_up if pitch > 0 else self.b_down)*pitch*self.pitch_profile(v)
        return d

    def move(self, v, yaw, pitch):
        return v+self.field(v, yaw, pitch)

    def feature(self, v, centre, yaw, pitch):
        """A part that moves as one piece: the surface's local affine map at
        its centre, with the gradient damped (rigid share)."""
        c = np.array([centre], float)
        h = 2.
        d0 = self.field(c, yaw, pitch)[0]
        jx = (self.field(c+[h, 0], yaw, pitch)[0]-self.field(c-[h, 0], yaw, pitch)[0])/(2*h)
        jy = (self.field(c+[0, h], yaw, pitch)[0]-self.field(c-[0, h], yaw, pitch)[0])/(2*h)
        q = v-c[0]
        return v+d0+self.rigid*(np.outer(q[:, 0], jx)+np.outer(q[:, 1], jy))

    def bun_sphere(self, v, out, yaw, pitch):
        """A bun as a ball behind the crown: it shifts against the turn and its
        surface rotates, so the knot's texture flows instead of sliding."""
        b = self.bun
        C = np.array(b['centre'], float); R = np.array(b['radius'], float)
        u = (v-C)/R
        z = np.sqrt(np.clip(1-(u*u).sum(1), 0, 1))
        phi = b['turn'][0]*yaw; psi = b['turn'][1]*pitch
        ux = u[:, 0]*np.cos(phi)+z*np.sin(phi)
        z2 = z*np.cos(phi)-u[:, 0]*np.sin(phi)
        uy = u[:, 1]*np.cos(psi)-z2*np.sin(psi)
        ball = C+np.c_[ux, uy]*R+[b['shift'][0]*yaw, b['shift'][1]*pitch]
        (x0, xw), (y0, yw) = b['blend_x'], b['blend_y']
        w = np.clip((v[:, 0]-x0)/xw, 0, 1)*np.clip((y0-v[:, 1])/yw, 0, 1)
        w = w*w*(3-2*w)
        return out+(ball-out)*w[:, None]
