"""Check what a run needs before any image is generated: Python and its
libraries, the Codex CLI and its login, a See-through token, Cubism Core,
ffmpeg and the VTube Studio models folder. Token files are only looked
for, never read out.

  python -m rigkit.doctor

✗ blocks a step (fix it first), ! only limits one (no token, no ffmpeg;
no Codex: the images come from another image tool).
Exits 1 when anything is ✗.
"""
from __future__ import annotations
from importlib.util import find_spec
from pathlib import Path
import os, shutil, subprocess, sys

LIBS = {'numpy': 'numpy', 'cv2': 'opencv-python', 'PIL': 'Pillow',
        'gradio_client': 'gradio_client', 'psd_tools': 'psd-tools'}


def checks():
    """(ok, required, name, detail) for each thing a run uses."""
    v = sys.version.split()[0]
    yield sys.version_info >= (3, 10), True, 'Python', v if sys.version_info >= (3, 10) else f'{v}，要 3.10 以上'
    missing = [pip for mod, pip in LIBS.items() if find_spec(mod) is None]
    yield not missing, True, 'Python 库', '齐全' if not missing else f'缺 {"、".join(missing)}：pip install -r requirements.txt'
    if missing:
        return

    from .variants import CODEX
    other = '；用其他图像工具出图时不需要'
    if not (os.path.isfile(CODEX) and os.access(CODEX, os.X_OK)):
        yield False, False, 'Codex 命令行', '找不到：装 ChatGPT 桌面版或单独装 codex，不在 PATH 里时把环境变量 CODEX 指向它'+other
    else:
        try:
            r = subprocess.run([CODEX, 'login', 'status'], capture_output=True, text=True, timeout=30)
            logged = r.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            logged = False
        yield logged, False, 'Codex 命令行', CODEX if logged else f'{CODEX} 未登录：运行 codex login，用 ChatGPT 账号登录'+other

    hf = bool(os.environ.get('HF_TOKEN')) or Path('~/.cache/huggingface/token').expanduser().exists()
    ms = bool(os.environ.get('MODELSCOPE_API_TOKEN')) or Path('~/.cache/modelscope/token').expanduser().exists()
    sites = '、'.join(n for n, has in (('Hugging Face', hf), ('魔搭', ms)) if has)
    yield hf or ms, False, 'See-through 令牌', sites or (
        '没有：Hugging Face 不登录的额度不一定够跑一次。建一个 Read token 放进 HF_TOKEN 或 '
        '~/.cache/huggingface/token，或魔搭访问令牌放进 MODELSCOPE_API_TOKEN 或 ~/.cache/modelscope/token')

    from .core import _find_core, core_version
    try:
        path = _find_core()
        yield True, True, 'Cubism Core', f'{core_version()}（{path}）'
    except (OSError, AttributeError) as e:
        yield False, True, 'Cubism Core', str(e)

    yield bool(shutil.which('ffmpeg')), False, 'ffmpeg', '有' if shutil.which('ffmpeg') else '没有：自查短片出不了，其余不受影响'

    from .vts import vts_models_dir
    d = vts_models_dir()
    yield d.is_dir(), True, 'VTube Studio', str(d) if d.is_dir() else f'找不到模型文件夹 {d}：装 Steam 版，或把环境变量 VTS_MODELS 指向 Live2DModels 文件夹'


def main():
    bad = 0
    for ok, required, name, detail in checks():
        bad += required and not ok
        print(f"{'✓' if ok else '✗' if required else '!'} {name}：{detail}")
    print('可以开始' if not bad else f'{bad} 项要先处理')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
