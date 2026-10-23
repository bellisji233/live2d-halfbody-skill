# Half-body Live2D Avatar Skill

English | [简体中文](README.zh-CN.md)

Give an AI agent (such as Claude Code) one character illustration or a text description, and it builds a half-body Live2D avatar for streaming in VTube Studio, step by step. No manual modeling, no Cubism Editor, no mesh editing.

![Second character: rest pose, three gestures, head turn during a gesture, blush](assets/示例.jpg)

![Two characters in VTube Studio: talking, blinking, head turns, hair and pendant swinging](assets/demo.gif)

![Third character: no bangs, hair tucked behind the ears, glasses, earrings](assets/demo-glasses.gif)

## Features

- Head turns left/right and up/down (±30); the body leans when the head tilts; body follow and breathing
- Blinking and winking, eye tracking, eyebrow movement, lip sync, smiling mouth corners
- Hair physics in sections: bangs, side locks (root and tip swing separately), back hair or bun; long hair follows the body below the shoulders. Bangs are optional, and the side hair can be tucked behind the ears
- Necklace pendant swings with the body; drop earrings swing from the earlobes
- Glasses follow the face
- Hotkeys: smiling eyes, O-shaped mouth, angry, troubled; wave, open hands and pointing gestures; blush

Not supported: large head turns, full body, arms following the camera continuously. Not yet tested: twin tails (in progress).

The output is a model file VTube Studio loads directly (.moc3), not a Cubism Editor project, so the Editor can't open it. For production-grade results (finer physics, more expressions), import the separated layers into Cubism Editor and rig them again. Suited to front-facing half-body characters with short hair or hair past the shoulders.

## How the illustration is adjusted

The agent redraws your illustration or description into a version that separates cleanly into layers, and you approve it before the next step. The face, hair color, eyes, clothing, palette and art style stay the same; these are adjusted:

- **Framing**: front-facing half body from the top of the head to the waist, plain white background; arms hang at the sides with a gap from the body, hands empty. The three gestures are edited from this pose.
- **Bangs**: optional. With bangs, a straight-cut fringe ending above the eyebrows, both eyebrows fully visible so they can move; without, the whole forehead shows.
- **Side hair**: either one smooth section on each side of the face, covering the ear and ending at the chin, swinging on its own when the head turns; or tucked behind the ears, which stay visible.
- **Long hair**: hangs behind the neck, shoulders and back, showing only beside the neck and outside the arms; not over the shoulders, chest or arms.
- **Neck**: a stretch of neck shows between the chin and the collar, so the chin slides over it when the head turns.
- **Color and light**: hair and skin in different colors, hair with a thin outline; flat lighting, no cast shadows. Shadows would shift out of place once the parts move.
- **Accessories**: glasses stay, drawn with a thin frame and clear lenses without glare; drop earrings stay, with the hair behind the ears; chokers and hanging ribbons are removed; a necklace stays only as a thin chain on the clothing with at most one pendant; hats sit on top of the head without covering the forehead or eyebrows.
- **Blush**: soft blush on the cheeks.

For hairstyles or accessories outside this list (twin tails, animal ears), ask the agent to extend the skill: add drawing rules, adjust the layer-splitting and rigging code, then check the result in VTube Studio.

## What you do

1. Approve the illustration the agent generates.
2. Agree to upload it to [See-through](https://github.com/shitagaki-lab/see-through) for layer separation (Hugging Face, ModelScope as backup).
3. Check the avatar in VTube Studio against the checklist, then reply "pass" or "fail: what's wrong".

## Requirements

- An AI agent that runs skills, such as Claude Code
- Python 3.10+, ffmpeg
- A ChatGPT account: images are generated with the Codex CLI (bundled with the ChatGPT desktop app)
- An account for See-through, at least one of:
  - Hugging Face, free (preferred, a few minutes per run): create a Read token and put it in the `HF_TOKEN` environment variable or `~/.cache/huggingface/token`. Without logging in, the daily free quota may not cover one run; logged in, about 1–2 runs a day
  - ModelScope, free (backup; queued, a run can take half an hour; higher resolution): create an access token and put it in `MODELSCOPE_API_TOKEN` or `~/.cache/modelscope/token`
- VTube Studio
- Live2D Cubism Core: included with Cubism Editor, or download the free Cubism SDK for Native

Tested on macOS. Not tested on Windows.

## Getting started

1. Put this repository in your agent's skill folder. For Claude Code: `git clone https://github.com/bellisji233/live2d-halfbody-skill ~/.claude/skills/live2d-halfbody`
2. Install dependencies: `pip install -r requirements.txt`
3. Tell the agent "Make a VTube Studio avatar from this illustration" with the image path, or describe the character you want.

From illustration to VTube Studio, one character takes 22–24 generated images, one See-through run, and a few minutes to build the model.

## Language

The skill instructions (`SKILL.md`, `references/`) and script messages are in Chinese. The agent reads them as they are; talk to it in any language.

## License

MIT, see [LICENSE](LICENSE). `rigkit/vendor/py-moc3` is MIT, license included. Live2D Cubism Core is not distributed with this skill; it is used under Live2D's license.
