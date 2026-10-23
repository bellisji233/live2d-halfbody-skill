---
name: live2d-halfbody
description: 从一张立绘或一段角色描述，做出能在 VTube Studio 里直播用的半身 Live2D 皮套，不用手动建模、不开 Cubism Editor：图像模型出图、See-through 拆层、代码补细节和绑定、自查后装进 VTS。默认带小角度转头、歪头、身体跟随、呼吸、眨眼和单眼眨眼、口型、四个表情、头发分段摆动（短发或过肩长发）、挂件摆动、招手/摊手/指向三个手势和脸红。用于用户想新做一个皮套（有立绘、只有描述、或已有分层文件）、检查皮套素材、或调整已有角色的动作幅度时。
---

# 半身皮套：从一张立绘到 VTube Studio

给不想手动建模、不想调网格的人。一张立绘或一段角色描述进去，Agent 一步步做出能在 VTS 里直播用的半身 Live2D 皮套。用户只做三件事：

1. **确认立绘**（第 1 步）；
2. **同意把立绘传到 See-through**（第 3 步，Hugging Face，备用魔搭）；
3. **在 VTS 里看效果**，回复 `通过` 或 `不通过：<现象>`（第 5 步）。

其余由 Agent 完成，并在交给用户前自查。

## 跑一次就有的功能

| 功能 | 驱动 |
|---|---|
| 转头左右、上下（±30），歪头时身体跟着倾斜 | 摄像头 |
| 身体跟随：左右转、前后、呼吸起伏 | 摄像头、VTS 自动呼吸 |
| 眨眼、单眼眨眼；半闭眼时双眼皮淡出 | 摄像头（左右眼分开） |
| 眼珠跟随、眉毛上下 | 摄像头 |
| 张嘴说话、嘴角笑 | 摄像头 |
| 头发分段摆动：刘海、两侧（发根和发梢两段）、后发或发髻；低头抬头时跟随。过肩长发在肩膀以下跟身体走 | 物理 |
| 挂件（项链吊坠）摆动 | 物理 |
| 表情：笑眼 N1、O 型嘴 N2、急眼 N3、困扰 N4，清除 N0 | 快捷键 |
| 手势：招手 N5、摊手 N6、指向 N7 | 快捷键（也配了摄像头手势，见“已知问题”） |
| 脸红 N8 | 快捷键 |

做不到：大角度转头（要侧脸素材）、全身、手臂跟着摄像头连续动（要分段手臂素材）、双马尾。产出是 VTS 直接用的 `.moc3`，不是 Cubism Editor 的工程文件（`.cmo3`），Editor 打不开。精修级的效果（更细的物理、更多表情、商业级网格）要把 `layers-st/` 的图层导进 Cubism Editor 重新绑定。

## 文件夹

```
素材/角色名/                 立绘、提示词、变体、See-through 结果、图层
  A0-base.png                定稿立绘
  prompts/                   从 references/提示词/标准/ 复制，填好 [[…]]
  check/                     每张变体的对位检查图
  seethrough.psd             See-through 的原始结果
  seethrough/                拆成每层一张 PNG
  layers/                    自拆的图层：只作 See-through 合成的底，不单独交付
  layers-st/                 最终图层（See-through 合成，含手势和脸红）
  R0-rest.png                最终图层的静止合成，手势和脸红变体的底图
角色/角色名/
  character.json             位置、幅度、图层对照
  标注.png                    character.json 里的位置画在立绘上
  素材检查/  prepared/  build/runtime/  自查/  vts/
```

## 步骤

### 1. 立绘 —— 用户确认

把 `references/提示词/标准/` 整个复制到 `素材/角色名/prompts/`，先填好 `A0-base.txt` 里的 `[[…]]`（外貌、短发或长发）。

- 只有描述：照描述填，生成立绘。
- 有立绘：存为 `素材/角色名/参考.png`，生成时作为参考图附上，得到一张按绑定要求画的版本（脖子露出、刘海不遮眉、侧发盖住耳朵到下巴为止、平光、白底、手臂和身体之间留缝）。原图不满足这些要求时，直接拆会在动起来时出问题。

```
python -m rigkit.variants 素材/角色名 A0-base
```

给用户看，用户确认后才往下做。

已有按 [图层清单](references/图层清单.md) 分好层的文件时，跳到第 4 步。

### 2. 出图

把 `prompts/` 里其余的 `[[…]]` 换成这个角色的描述；没有帽子删掉 V1，没有挂件删掉 V9。然后：

```
python -m rigkit.variants 素材/角色名
```

依次生成 17～19 张“整张图改一处”的变体，每张出对位检查图。偏移 4 px 以内、改动只在该改的地方算合格；不合格删掉重出，同一张最多三次。细节见 [出图](references/出图.md)。

### 3. 拆层 —— 用户同意上传

先用变体自拆，作为合成的底：

```
python -m rigkit.split 素材/角色名 素材/角色名/layers --pad-top 90
```

再问用户是否同意把立绘传到 See-through，同意后：

```
python -m rigkit.seethrough fetch 素材/角色名/A0-base.png 素材/角色名/seethrough
python -m rigkit.seethrough split 素材/角色名 素材/角色名/layers 素材/角色名/layers-st --pad-top 90
```

`fetch` 先试 Hugging Face，没跑成再换魔搭（`--site hf` 或 `--site modelscope` 指定一个），各用能给的最高分辨率。也可以把网页版、ComfyUI 或自己显卡跑出的 PSD 交给它：`fetch 那个.psd 素材/角色名/seethrough`。

最终图层必须走 See-through：自拆的图层动起来边缘有碎点、被挡住的头发补不全，不单独交付。当天没有额度时，换另一个网站、等第二天，或者用 PSD。

`split` 会逐条打印它对 See-through 图层做的修正（裁掉画到别处的、挪回归错的配饰、补漏画的部件），规则见 [规则](references/规则.md) 的“See-through 合成”。看 `layers-st/拆层总览.png`：每层只有自己的东西，静止合成和立绘只差在画风。

然后做手势和脸红：

```
python -m rigkit.extras 素材/角色名 素材/角色名/layers-st        # 写出 R0-rest.png
python -m rigkit.variants 素材/角色名 F1-blush P1-wave P2-explain P3-point
python -m rigkit.extras 素材/角色名 素材/角色名/layers-st        # 加进图层
```

看 `layers-st/手势和脸红.png`：每行是静止、变体、模型里打开后的样子，手臂和身体接得上、手的边缘干净才往下做。

### 4. 生成并自查

```
python -m rigkit.estimate 素材/角色名/layers-st 角色/角色名 --name 名字
python -m rigkit.check_art 角色/角色名/character.json
python -m rigkit.prepare 角色/角色名/character.json
python -m rigkit.build 角色/角色名/character.json             # 几分钟
```

- `estimate` 不覆盖已有的 `character.json`，写成 `character.draft.json`；图层改过要重估时，比对后换上。
- `标注.png`：脸的椭圆、眼眉鼻嘴、歪头转轴、颈根、肩、腰、发根线不在对的位置时，改 `character.json` 里对应的数字，重新 prepare 和 build。
- `素材检查/`：`露底` 有品红、`原画对照` 有成线的红色时，回到第 3 步。

自查：

```
python -m rigkit.preview flips 角色/角色名/build/runtime
python -m rigkit.preview sheet 角色/角色名/character.json 角色/角色名/自查 新版=角色/角色名/build/runtime
python -m rigkit.preview clip 角色/角色名/character.json 角色/角色名/自查/动作.mp4 新版=角色/角色名/build/runtime
```

翻折三角形应为 0。看放大图和短片里的固定检查区（眉毛和刘海、脸边、下巴和脖子、太阳穴和耳朵、眼角和睫毛、嘴角），有问题先修，再交给用户。有上一版时把上一版放在前面并排比。

### 5. 装进 VTS —— 用户验收

```
python -m rigkit.vts 角色/角色名/character.json --install
```

新模型可以在 VTS 开着时装；覆盖已装的模型要先退出 VTS，这时保留用户在 VTS 里调过的设置，只补缺的快捷键和映射。VTS 开着又要换新版时，改 `vts.name` 和 `vts.folder` 装成新文件夹。

给用户并排图、短片和下面这张清单，回复 `通过` 或 `不通过：<现象>`：

| 看什么 | 怎么做 |
|---|---|
| 转头、歪头 | 左右上下转头、歪头：脸、头发、帽子、耳朵跟着转，没有裂缝、碎点、留在原地的东西 |
| 眼睛 | 眨眼、单眼眨眼、半闭眼；转头时睫毛不穿过头发 |
| 嘴 | 说话、笑；闭着嘴转头时不歪嘴 |
| 头发 | 甩头：刘海、侧发、后发分段摆，发梢比发根慢；长发在肩膀以下跟身体 |
| 挂件 | 身体动时吊坠摆动，底下的衣服完整 |
| 身体 | 身体跟随左右转、前后，呼吸起伏 |
| 表情 | N1～N4 切换，N0 清除 |
| 手势 | N5～N7：身体画风不跳，手的边缘干净；手势打开时转头，手臂跟着身体走 |
| 脸红 | N8：边缘柔和；甩头时头发下面不露边 |

## 调幅度

动作幅度都在 `character.json`：

| 想改 | 字段 |
|---|---|
| 歪头角度 | `head.tilt_deg`（满幅时的度数） |
| 转头时脸走多远 | `head.yaw_px`、`head.pitch_px` |
| 身体倾斜 | `body.tilt_deg`（每单位参数的度数，满幅 ×10） |
| 呼吸起伏 | `body.breath.head`、`body.breath.shoulder`（像素） |
| 身体跟随 | `body.turn`、`body.shift`、`body.lean_y` |
| 头发摆幅 | `hair.locks.<图层>.swing`；发梢 `hair.tips.amount` |
| 物理手感 | `hair.locks.<图层>.physics`：`delay` 越小越拖、`length` 越长越慢 |
| 挂件摆幅 | `dangles.<图层>.deg`（满幅时的度数） |
| 半闭眼 | `eyes.crease_fade`（双眼皮淡出的区间）、`eyes.lower_from`（下眼睑开始合的位置） |
| 嘴从哪里开始张 | `mouth.open_from`：摄像头读数低于它时保持闭嘴 |

幅度按 VTS 半身取景和真人动作判断：正常歪头约 15°，模型只转 3° 就等于没有。改完重新 build。其余字段见 [配置说明](references/配置说明.md)。

## 规则

全部规则和来由见 [规则](references/规则.md)。最常被违反的几条：

- 动手前先查 Live2D 的标准做法，说清楚打算怎么做。
- 切图用 See-through，图像模型只补它给不了的细节（表情、被挡住的部分、手势）。
- 每轮给同尺寸并排图和短片，由人判断；翻折数、像素差只用来排查破面。
- 同一个症状修了三轮还在，停止改代码，改用素材解决。
- 图像模型只派“整张图改一处”，不派“只留一部分”；要的部分由代码用差值算。
- 传图到外部服务前先问用户；账号和 token 由用户自己注册、自己填，不在对话里复述 token。
- 安装时不覆盖用户的 vtube.json，覆盖已装模型时 VTS 要先退出。

## 环境

- **Python 3.10+**：`pip install -r requirements.txt`。出短片要 ffmpeg。
- **Codex 命令行**（出图）：ChatGPT 桌面版自带，登录 ChatGPT 账号即可；不在默认位置时把环境变量 `CODEX` 指向它。
- **See-through**（拆层），两个网站跑的是同一个模型，至少准备一个：
  - **Hugging Face**（首选，几分钟一次，最高 1280）：不登录时每天的免费显卡时间不一定够跑一次；用户注册免费账号、建一个 Read token，放进环境变量 `HF_TOKEN` 或 `~/.cache/huggingface/token`（`hf auth login` 存的位置），每天约 1～2 次。
  - **魔搭 ModelScope**（备用，免费，最高 1536）：用户注册账号、建一个访问令牌，放进环境变量 `MODELSCOPE_API_TOKEN` 或 `~/.cache/modelscope/token`。要排队，一次可能半小时。1536 的头发轮廓和眼睛比 1280 更接近原画。
- **VTube Studio**（Steam 版）。
- **Live2D Cubism Core**（预览、自查、图标要用）：装了 Cubism Editor 时自动找到；否则下载 Cubism SDK for Native，把环境变量 `CUBISM_CORE` 指向其中的 Core 动态库。Core 按 Live2D 的许可使用，不随本 skill 分发。

macOS 上验证过；Windows 未验证。

## 已知问题

- 摄像头手势（张开手掌 = 招手、双手张开 = 摊手、竖食指 = 指向）：第一个角色能触发，第二个角色配置逐字段相同却不触发，原因未查明。快捷键 N5～N7 不受影响。
- 静止时是 See-through 画的样子，比原画软一点，睫毛、发梢描边的细节少一些。
