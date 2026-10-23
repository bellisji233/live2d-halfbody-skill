"""用 Core 求值 + numpy 光栅化，把 moc3 渲染成 PNG。

不需要 Cubism SDK：几何来自原生 Core，三角形自己画。
支持剪贴蒙版（把蒙版网格画进 alpha 缓冲再相乘）。
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
import numpy as np
from PIL import Image

from .core import Model, core_version


def _raster(target, P, UV, tris, tex, opacity, alpha_only=False):
    """把带纹理的三角形画进 target（H,W,4 float32，预乘 alpha 的 over 合成）。"""
    h, w = target.shape[:2]
    th, tw = tex.shape[:2]
    # GPU rasterization owns each shared edge once (top-left fill rule).
    # A software >= epsilon test admits both adjoining triangles. Claiming
    # the pixel once prevents translucent ink from darkening on mesh seams.
    covered = np.zeros((h, w), dtype=bool)
    for tri in tris:
        a, b, c = P[tri]
        ua, ub, uc = UV[tri]
        x0 = int(max(0, np.floor(min(a[0], b[0], c[0]))))
        x1 = int(min(w, np.ceil(max(a[0], b[0], c[0])) + 1))
        y0 = int(max(0, np.floor(min(a[1], b[1], c[1]))))
        y1 = int(min(h, np.ceil(max(a[1], b[1], c[1])) + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        v0, v1 = b - a, c - a
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1) + .5, np.arange(y0, y1) + .5)
        dx, dy = xs - a[0], ys - a[1]
        w1 = (dx * v1[1] - v1[0] * dy) / den
        w2 = (v0[0] * dy - dx * v0[1]) / den
        w0 = 1 - w1 - w2
        e = -1e-6
        inside = (w0 >= e) & (w1 >= e) & (w2 >= e)
        ownership = covered[y0:y1, x0:x1]
        inside &= ~ownership
        ownership |= inside
        if not inside.any():
            continue
        u = (w0 * ua[0] + w1 * ub[0] + w2 * uc[0]) * tw
        # Core 返回的 V 以纹理底边为 0（GL 约定），这里按行从上往下采样，所以取反
        v = (1.0 - (w0 * ua[1] + w1 * ub[1] + w2 * uc[1])) * th
        # 双线性采样
        u = np.clip(u - .5, 0, tw - 1.001)
        v = np.clip(v - .5, 0, th - 1.001)
        u0i, v0i = u.astype(np.int32), v.astype(np.int32)
        fu, fv = (u - u0i)[..., None], (v - v0i)[..., None]
        t00 = tex[v0i, u0i]; t10 = tex[v0i, u0i + 1]
        t01 = tex[v0i + 1, u0i]; t11 = tex[v0i + 1, u0i + 1]
        src = (t00 * (1 - fu) * (1 - fv) + t10 * fu * (1 - fv)
               + t01 * (1 - fu) * fv + t11 * fu * fv)
        sa = src[..., 3] * opacity * inside
        sub = target[y0:y1, x0:x1]
        if alpha_only:
            sub[..., 3] = np.maximum(sub[..., 3], sa)
        else:
            al = sa[..., None]
            sub[...] = np.concatenate([src[..., :3], np.ones_like(al)], -1) * al + sub * (1 - al)


def render(model_dir: str | Path, manifest: str, params: dict | None = None,
           size: int | None = None, background=None,
           only: list[str] | None = None) -> Image.Image:
    d = Path(model_dir)
    man = json.loads((d / manifest).read_text())
    model = Model(str(d / man["FileReferences"]["Moc"]))
    texs = [np.asarray(Image.open(d / t).convert("RGBA"), np.float32) / 255
            for t in man["FileReferences"]["Textures"]]
    model.reset_params()
    if params:
        model.set_params(params)
    model.update()
    g = model.geometry()

    cw, ch = model.canvas_px
    W = int(round(cw)) if size is None else size
    scale = W / cw
    H = int(round(ch * scale))

    def to_px(pts):
        q = model.to_pixels(pts)
        return q * scale

    canvas = np.zeros((H, W, 4), np.float32)
    if background is not None:
        canvas[..., :3] = np.array(background, np.float32)[:3] / 255
        canvas[..., 3] = 1.0

    def bbox_of(P):
        x0 = int(max(0, np.floor(P[:, 0].min()) - 1)); x1 = int(min(W, np.ceil(P[:, 0].max()) + 2))
        y0 = int(max(0, np.floor(P[:, 1].min()) - 1)); y1 = int(min(H, np.ceil(P[:, 1].max()) + 2))
        return x0, y0, x1, y1

    for d_i in np.argsort(g["render_orders"], kind="stable"):
        if only is not None and model.drawable_ids[d_i] not in only:
            continue
        op = float(g["opacities"][d_i])
        if len(g["triangles"][d_i]) == 0 or op <= 0:
            continue
        P, UV = to_px(g["positions"][d_i]), g["uvs"][d_i]
        x0, y0, x1, y1 = bbox_of(P)
        if x1 <= x0 or y1 <= y0:
            continue
        # 每个部件先画进自己的缓冲再整体合成。
        # 直接往画布上逐三角形叠加的话，相邻三角形共享边上的像素会被画两次，
        # 不透明度 <1 时叠成深色竖条。
        layer = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
        off = np.array([x0, y0], np.float32)
        _raster(layer, P - off, UV, g["triangles"][d_i], texs[g["texture_indices"][d_i]], 1.0)
        for mi in g["masks"][d_i]:
            mp = to_px(g["positions"][mi])
            mbuf = np.zeros_like(layer)
            _raster(mbuf, mp - off, g["uvs"][mi], g["triangles"][mi],
                    texs[g["texture_indices"][mi]], 1.0, alpha_only=True)
            layer *= mbuf[..., 3:4]
        if op < 1.0:
            layer *= op
        sub = canvas[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = layer + sub * (1 - layer[..., 3:4])

    rgb = np.clip(canvas[..., :3], 0, 1)
    a = np.clip(canvas[..., 3], 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rgb = np.where(a[..., None] > 1e-6, rgb / np.maximum(a[..., None], 1e-6), 0)
    out = np.concatenate([rgb, a[..., None]], -1)
    return Image.fromarray((out * 255 + .5).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--manifest", default="Halfbody.model3.json")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--param", action="append", default=[], metavar="ID=VALUE")
    ap.add_argument("--bg", default=None, help="背景色，如 f6f2ee；省略为透明")
    ap.add_argument("--only", default=None, help="只画这些部件，逗号分隔")
    a = ap.parse_args()
    params = {}
    for kv in a.param:
        k, v = kv.split("=")
        params[k] = float(v)
    bg = None
    if a.bg:
        s = a.bg.lstrip("#")
        bg = tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    img = render(a.model_dir, a.manifest, params or None, a.size, bg,
                 only=a.only.split(",") if a.only else None)
    img.save(a.out)
    print(f"Core {core_version()} → {a.out}  {img.size[0]}×{img.size[1]}")


if __name__ == "__main__":
    main()
