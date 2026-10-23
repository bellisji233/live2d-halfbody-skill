"""Cubism Core (ctypes) 封装。

Core 库的位置：环境变量 CUBISM_CORE；没设时依次找 Cubism Editor 自带的
JNI 库（导出完整 csm* C API）和 Cubism SDK for Native 里的 Core 动态库。
Core 由 Live2D 提供，不随本工具分发。

坐标约定（实测 Core 6.0.513）：
- moc3 文件里存的 y 向下：(py_from_top - origin_y) / ppu
- Core 返回的 y 向上，即上面那个值取反
- 画布 size / origin 是像素，ppu 把像素换算成模型单位
"""
from __future__ import annotations
import ctypes, os
import numpy as np

CANDIDATES = [
    "/Applications/Live2D Cubism 5.4 alpha/res/libLive2DCubismCoreJNI.dylib",
    "/Applications/Live2D Cubism 5.3/res/libLive2DCubismCoreJNI.dylib",
    "/Applications/Live2D Cubism 5.2/res/libLive2DCubismCoreJNI.dylib",
    "CubismSdkForNative/Core/dll/macos/libLive2DCubismCore.dylib",
    "CubismSdkForNative/Core/dll/windows/x86_64/Live2DCubismCore.dll",
    "CubismSdkForNative/Core/dll/linux/x86_64/libLive2DCubismCore.so",
]


def _find_core():
    if os.environ.get("CUBISM_CORE"):
        return os.environ["CUBISM_CORE"]
    for c in CANDIDATES:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("找不到 Cubism Core：设置环境变量 CUBISM_CORE 指向 Core 动态库")
ALIGN_MOC, ALIGN_MODEL = 64, 16

_lib = None
_log_cb = None
_log_sink: list[str] = []


def lib():
    global _lib, _log_cb
    if _lib is None:
        _lib = ctypes.CDLL(_find_core())
        proto = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
        _log_cb = proto(lambda m: _log_sink.append(m.decode("utf-8", "replace")))
        _lib.csmSetLogFunction(_log_cb)
        for name, res, args in [
            ("csmGetVersion", ctypes.c_uint, []),
            ("csmGetMocVersion", ctypes.c_uint, [ctypes.c_void_p, ctypes.c_uint]),
            ("csmReviveMocInPlace", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_uint]),
            ("csmGetSizeofModel", ctypes.c_uint, [ctypes.c_void_p]),
            ("csmInitializeModelInPlace", ctypes.c_void_p,
             [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]),
            ("csmUpdateModel", None, [ctypes.c_void_p]),
            ("csmResetDrawableDynamicFlags", None, [ctypes.c_void_p]),
            ("csmGetParameterCount", ctypes.c_int, [ctypes.c_void_p]),
            ("csmGetParameterIds", ctypes.POINTER(ctypes.c_char_p), [ctypes.c_void_p]),
            ("csmGetParameterValues", ctypes.POINTER(ctypes.c_float), [ctypes.c_void_p]),
            ("csmGetParameterDefaultValues", ctypes.POINTER(ctypes.c_float), [ctypes.c_void_p]),
            ("csmGetDrawableCount", ctypes.c_int, [ctypes.c_void_p]),
            ("csmGetDrawableIds", ctypes.POINTER(ctypes.c_char_p), [ctypes.c_void_p]),
        ]:
            f = getattr(_lib, name)
            f.restype, f.argtypes = res, args
    return _lib


def drain_log() -> list[str]:
    out = list(_log_sink)
    _log_sink.clear()
    return out


class Model:
    """一个已加载的模型。持有对齐缓冲区，不能让它被回收。"""

    def __init__(self, moc_path: str):
        L = lib()
        data = open(moc_path, "rb").read()
        self._moc_buf, moc_addr = _aligned(data, ALIGN_MOC)
        self.file_moc_version = L.csmGetMocVersion(moc_addr, len(data))
        self._moc = L.csmReviveMocInPlace(moc_addr, len(data))
        if not self._moc:
            raise RuntimeError(f"moc3 加载失败：{moc_path}\n" + "\n".join(drain_log()))
        size = L.csmGetSizeofModel(self._moc)
        self._model_buf = ctypes.create_string_buffer(size + ALIGN_MODEL)
        base = ctypes.addressof(self._model_buf)
        self._ptr = L.csmInitializeModelInPlace(self._moc, base + (-base) % ALIGN_MODEL, size)
        if not self._ptr:
            raise RuntimeError("模型初始化失败\n" + "\n".join(drain_log()))
        self.n_param = L.csmGetParameterCount(self._ptr)
        self.n_draw = L.csmGetDrawableCount(self._ptr)
        ids = L.csmGetParameterIds(self._ptr)
        self.param_ids = [ids[i].decode() for i in range(self.n_param)]
        dids = L.csmGetDrawableIds(self._ptr)
        self.drawable_ids = [dids[i].decode() for i in range(self.n_draw)]
        self._pvals = L.csmGetParameterValues(self._ptr)
        rng = []
        for fn in ("csmGetParameterMinimumValues", "csmGetParameterMaximumValues"):
            f = getattr(L, fn)
            f.restype, f.argtypes = ctypes.POINTER(ctypes.c_float), [ctypes.c_void_p]
            v = f(self._ptr)
            rng.append([v[i] for i in range(self.n_param)])
        self.param_min, self.param_max = rng
        size2 = (ctypes.c_float * 2)()
        org2 = (ctypes.c_float * 2)()
        ppu = ctypes.c_float()
        L.csmReadCanvasInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float * 2),
                                        ctypes.POINTER(ctypes.c_float * 2), ctypes.POINTER(ctypes.c_float)]
        L.csmReadCanvasInfo(self._ptr, ctypes.byref(size2), ctypes.byref(org2), ctypes.byref(ppu))
        self.canvas_px = (size2[0], size2[1])
        self.origin_px = (org2[0], org2[1])
        self.ppu = ppu.value

    def set_params(self, values: dict[str, float]):
        for i, k in enumerate(self.param_ids):
            if k in values:
                self._pvals[i] = float(values[k])

    def reset_params(self):
        L = lib()
        d = L.csmGetParameterDefaultValues(self._ptr)
        for i in range(self.n_param):
            self._pvals[i] = d[i]

    def update(self):
        L = lib()
        L.csmUpdateModel(self._ptr)
        L.csmResetDrawableDynamicFlags(self._ptr)

    # --- 几何读取 ---
    def _ints(self, fn) -> np.ndarray:
        f = getattr(lib(), fn)
        f.restype, f.argtypes = ctypes.POINTER(ctypes.c_int), [ctypes.c_void_p]
        return np.ctypeslib.as_array(f(self._ptr), (self.n_draw,)).copy()

    def _floats(self, fn) -> np.ndarray:
        f = getattr(lib(), fn)
        f.restype, f.argtypes = ctypes.POINTER(ctypes.c_float), [ctypes.c_void_p]
        return np.ctypeslib.as_array(f(self._ptr), (self.n_draw,)).copy()

    def geometry(self) -> dict:
        L = lib()
        vc = self._ints("csmGetDrawableVertexCounts")
        ic = self._ints("csmGetDrawableIndexCounts")
        mc = self._ints("csmGetDrawableMaskCounts")

        def vec2(fn):
            f = getattr(L, fn)
            f.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
            f.argtypes = [ctypes.c_void_p]
            p = f(self._ptr)
            return [np.ctypeslib.as_array(p[i], (int(vc[i]) * 2,)).reshape(-1, 2).copy()
                    for i in range(self.n_draw)]

        f = L.csmGetDrawableIndices
        f.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_ushort))
        f.argtypes = [ctypes.c_void_p]
        ip = f(self._ptr)
        tris = [np.ctypeslib.as_array(ip[i], (int(ic[i]),)).reshape(-1, 3).copy()
                if ic[i] else np.zeros((0, 3), np.uint16) for i in range(self.n_draw)]

        f = L.csmGetDrawableMasks
        f.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
        f.argtypes = [ctypes.c_void_p]
        mp = f(self._ptr)
        masks = [[int(mp[i][j]) for j in range(int(mc[i]))] for i in range(self.n_draw)]

        return dict(
            positions=vec2("csmGetDrawableVertexPositions"),
            uvs=vec2("csmGetDrawableVertexUvs"),
            triangles=tris,
            masks=masks,
            opacities=self._floats("csmGetDrawableOpacities"),
            draw_orders=self._ints("csmGetDrawableDrawOrders"),
            render_orders=self._ints("csmGetDrawableRenderOrders") if hasattr(L, "csmGetDrawableRenderOrders") else self._ints("csmGetDrawableDrawOrders"),
            texture_indices=self._ints("csmGetDrawableTextureIndices"),
            constant_flags=self._u8("csmGetDrawableConstantFlags"),
        )

    def _u8(self, fn) -> np.ndarray:
        f = getattr(lib(), fn)
        f.restype, f.argtypes = ctypes.POINTER(ctypes.c_ubyte), [ctypes.c_void_p]
        return np.ctypeslib.as_array(f(self._ptr), (self.n_draw,)).copy()

    def to_pixels(self, pts: np.ndarray) -> np.ndarray:
        """模型单位 → 画布像素（左上角原点，y 向下）。"""
        ox, oy = self.origin_px
        return np.stack([pts[:, 0] * self.ppu + ox, oy - pts[:, 1] * self.ppu], 1)


def _aligned(data: bytes, align: int):
    buf = ctypes.create_string_buffer(len(data) + align)
    addr = ctypes.addressof(buf)
    off = (-addr) % align
    ctypes.memmove(addr + off, data, len(data))
    return buf, addr + off


def core_version() -> str:
    v = lib().csmGetVersion()
    return f"{(v >> 24) & 0xFF}.{(v >> 16) & 0xFF}.{v & 0xFFFF}"
