"""Make the edit variants of a base illustration with an image model, and
check each one: how far it drifted from its base, and where it changed.

  python -m rigkit.variants ART_DIR [NAME ...]
  python -m rigkit.variants ART_DIR A0-base     # the illustration itself

ART_DIR holds A0-base.png and prompts/<NAME>.txt. A0-base is generated
new, with 参考.* in ART_DIR attached as a reference when there is one;
the user confirms it before anything else is made. Each variant is an edit
of the image named in split.BASES (default the base illustration), past
the hat and bangs steps a character leaves out. Missing
outputs are generated through the Codex CLI (CODEX env var or the copy
bundled with ChatGPT.app), one at a time. For every variant a check image
check/<NAME>.png shows base | variant | difference, and check/report.json
lists the drift (px) and the share of the canvas that changed.

A variant is usable when its drift is within 4 px and the change sits
only where the edit asked for. Redo a failed one (delete it and rerun)
at most three times; past that, change the prompt or the plan.
"""
from __future__ import annotations
from pathlib import Path
import json, os, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .split import load, align, base_of, ORDER

CODEX = os.environ.get('CODEX', '/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex')
BASE = 'A0-base'
REFERENCE = '参考'


def reference(art: Path):
    return next(iter(sorted(art.glob(f'{REFERENCE}.*'))), None)


def generate(art: Path, name: str, base: str):
    out = art/f'{name}.png'
    prompt = (art/'prompts'/f'{name}.txt').read_text()
    if name == BASE:
        ref = reference(art)
        task = '任务：用图像生成工具生成一张新图，附带的图片只作参考。' if ref else '任务：用图像生成工具生成一张新图。'
        images = ['-i', str(ref)] if ref else []
    else:
        task = '任务：用图像生成工具编辑附带的图片，一次生成一张。'
        images = ['-i', str(art/f'{base}.png')]
    request =(task+'提示词必须原样使用下面 PROMPT 段落的全文，不要改写、不要加内容。\n'
               f'生成后把输出图片复制为 {out} 。不要抠图、不要缩放或修补输出，不要改动任何其他文件。\n'
               '最后只回复两行：输出文件路径、生成工具的原始输出文件路径。\n\nPROMPT:\n'+prompt)
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write(request)
    try:
        with open(f.name) as stdin:
            subprocess.run([CODEX, 'exec', '-s', 'workspace-write', '--skip-git-repo-check', '-C', str(art),
                            '-c', 'model_reasoning_effort="medium"', *images, '-'],
                           stdin=stdin, check=True, capture_output=True, text=True)
    finally:
        os.unlink(f.name)
    if not out.exists():
        raise RuntimeError(f'{name}: no output')
    return out


def check(art: Path, name: str, base: str):
    ref = load(art/f'{base}.png')
    img = load(art/f'{name}.png')
    if img.shape != ref.shape:
        img = np.asarray(Image.fromarray(img.astype(np.uint8)).resize(ref.shape[1::-1], Image.LANCZOS)).astype(np.float32)
    aligned, (dx, dy), resp = align(img, ref)
    d = np.abs(aligned-ref).max(2)
    changed = float((d > 40).mean())
    heat = (ref.mean(2, keepdims=True).repeat(3, 2)*.35+160)
    hot = np.clip(d*4, 0, 255)
    heat[..., 0] = np.maximum(heat[..., 0], hot); heat[..., 1] *= 1-hot/400; heat[..., 2] *= 1-hot/400
    tiles = [ref, aligned, heat]
    w = ref.shape[1]//2
    sheet = Image.new('RGB', (3*w+16, ref.shape[0]//2+36), 'white')
    dr = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc', 20)
    except OSError:
        font = None
    for k, (t, cap) in enumerate(zip(tiles, [base, name, f'差异  偏移 {np.hypot(dx, dy):.1f} px'])):
        im = Image.fromarray(np.clip(t, 0, 255).astype(np.uint8)).resize((w, ref.shape[0]//2), Image.LANCZOS)
        sheet.paste(im, (k*(w+8), 36)); dr.text((k*(w+8)+6, 6), cap, fill=(80, 40, 40), font=font)
    (art/'check').mkdir(exist_ok=True)
    sheet.save(art/'check'/f'{name}.png')
    return dict(name=name, base=base, drift_px=round(float(np.hypot(dx, dy)), 2), shift=[round(dx, 2), round(dy, 2)],
                changed=round(changed, 4), phase_response=round(resp, 3))


def main():
    art = Path(sys.argv[1]).resolve()
    names = sys.argv[2:] or ORDER
    report_path = art/'check'/'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for name in names:
        if name == BASE:
            if (art/f'{BASE}.png').exists():
                print(f'{BASE}: 已有，不满意时删掉重出'); continue
            if '[[' in (art/'prompts'/f'{BASE}.txt').read_text():
                print(f'{BASE}: 提示词里还有待填的 [[…]]，按角色填好再生成'); continue
            ref = reference(art)
            print(f'{BASE}: generating' + (f' from reference {ref.name}' if ref else ''), flush=True)
            try:
                generate(art, BASE, None)
            except (subprocess.CalledProcessError, RuntimeError) as e:
                tail = (getattr(e, 'stderr', '') or getattr(e, 'stdout', '') or str(e)).strip().splitlines()[-3:]
                print(f'{BASE}: 生成失败', *tail, sep='\n  ')
                break
            print(f'{BASE}: 已生成，给用户确认；不满意删掉重出', flush=True)
            continue
        base = base_of(name, lambda n: (art/'prompts'/f'{n}.txt').exists())
        if not (art/'prompts'/f'{name}.txt').exists():
            continue                                    # left out for this character (no hat, no dangles)
        if not (art/f'{base}.png').exists():
            print(f'{name}: waiting for {base}'); continue
        if not (art/f'{name}.png').exists():
            if '[[' in (art/'prompts'/f'{name}.txt').read_text():
                print(f'{name}: 提示词里还有待填的 [[…]]，按角色填好再生成'); continue
            print(f'{name}: generating from {base}', flush=True)
            try:
                generate(art, name, base)
            except (subprocess.CalledProcessError, RuntimeError) as e:
                tail = (getattr(e, 'stderr', '') or getattr(e, 'stdout', '') or str(e)).strip().splitlines()[-3:]
                print(f'{name}: 生成失败，已停止（额度用完时等恢复后重跑，已生成的会跳过）', *tail, sep='\n  ')
                break
        r = check(art, name, base)
        report[name] = r
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(f"{name}: drift {r['drift_px']} px, changed {r['changed']*100:.1f}%", flush=True)


if __name__ == '__main__':
    main()
