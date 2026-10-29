"""Package a built runtime as a VTube Studio model folder, and install it.

Adds what VTS needs on top of the MOC: physics3 (hair follow-through, a
three-point chain where a lock has a tip parameter), expressions with
hotkeys, webcam and gesture mappings, auto-breath, and an icon. Hotkey and
parameter objects follow VTS's bundled sample model (akari).

  python -m rigkit.vts CHARACTER_JSON [RUNTIME] [--install]

Installing never replaces a vtube.json VTS has saved (the user's tuning):
it only adds hotkeys and mappings the saved file lacks, after a backup.
Replacing an installed model needs VTS closed: it writes its settings back
when it exits. A new folder can be added while VTS runs.
"""
from pathlib import Path
import argparse, copy, json, os, platform, shutil, subprocess, uuid
from PIL import Image

from .config import Character


def vts_models_dir():
    if os.environ.get('VTS_MODELS'):
        return Path(os.environ['VTS_MODELS'])
    if platform.system() == 'Darwin':
        return Path.home()/'Library/Application Support/Steam/steamapps/common/VTube Studio/VTubeStudio.app/Contents/Resources/Data/StreamingAssets/Live2DModels'
    return Path(r'C:\Program Files (x86)\Steam\steamapps\common\VTube Studio\VTube Studio_Data\StreamingAssets\Live2DModels')


# (file, display name, key, parameters, fade seconds); kept only if the model has the parameters
EXPRESSIONS = [
    ['smile', '笑眼', 'N1', {'ParamEyeSmile': 1}, .25],
    ['surprise', 'O 型嘴', 'N2', {'ParamMouthVariant': 1}, .15],
    # brows only: a fixed mouth shape would override lip sync while active
    ['angry', '急眼', 'N3', {'ParamBrowForm': -1}, .25],
    ['troubled', '困扰', 'N4', {'ParamBrowForm': 1}, .25],
    # one 0..1 switch per pose, so a fade never passes through another pose
    ['wave', '招手', 'N5', {'ParamPoseWave': 1}, .15],
    ['explain', '摊手', 'N6', {'ParamPoseExplain': 1}, .15],
    ['point', '指向', 'N7', {'ParamPosePoint': 1}, .15],
    ['blush', '脸红', 'N8', {'ParamCheek': 1}, .4],
]
# Webcam hand gestures (active only while held). IDs are VTS's gestureID
# strings. A one-hand gesture waits longer than a two-hand one, so raising
# both hands does not flash the one-hand pose first.
# (name, expression file, left gesture, right gesture, combinator, seconds)
GESTURES = [
    ['招手（手势）', 'wave', '', 'HandGesture_Palm', 'OR', .8],
    ['摊手（手势）', 'explain', 'HandGesture_Palm', 'HandGesture_Palm', 'AND', .4],
    ['指向（手势）', 'point', '', 'HandGesture_Pointing', 'OR', .5],
]
# input, in range, output, out range, smoothing, breathing.
# Layer names use _L for the eye on screen left, the opposite of the Live2D
# sample convention: inputs are swapped so a wink behaves like the samples.
# Webcam eye openness rarely exceeds ~.7 with eyes normally open; an output
# range up to 1.4 makes that read as fully open (the model clamps at 1).
MAPPINGS = [
    ('FaceAngleX', (-30, 30), 'ParamAngleX', (-30, 30), 15, False),
    ('FaceAngleY', (-20, 20), 'ParamAngleY', (-30, 30), 15, False),
    ('FaceAngleZ', (-30, 30), 'ParamAngleZ', (-30, 30), 30, False),
    ('FaceAngleZ', (-30, 30), 'ParamBodyAngleZ', (-10, 10), 20, False),
    ('EyeOpenLeft', (0, 1), 'ParamEyeROpen', (0, 1.4), 10, False),
    ('EyeOpenRight', (0, 1), 'ParamEyeLOpen', (0, 1.4), 10, False),
    ('EyeRightX', (-1, 1), 'ParamEyeBallX', (1, -1), 15, False),
    ('EyeRightY', (-1, 1), 'ParamEyeBallY', (-1, 1), 15, False),
    ('MouthOpen', (0, 1), 'ParamMouthOpenY', (0, 1), 0, False),
    ('Brows', (0, 1), 'ParamBrowForm', (-.5, .5), 15, False),
    # the body follows the head at a third of its range, smoothed more so it lags
    ('FaceAngleX', (-30, 30), 'ParamBodyAngleX', (-10, 10), 25, False),
    ('FaceAngleY', (-20, 20), 'ParamBodyAngleY', (-10, 10), 25, False),
    # auto-breath: no input, VTS drives it when UseBreathing is on
    ('', (0, 1), 'ParamBreath', (0, 1), 0, True),
    ('MouthSmile', (0, 1), 'ParamMouthForm', (-1, 1), 10, False),
]
# per lock: pendulum length, mobility, delay (lower = more lag), acceleration,
# output scale, tip output scale
PHYSICS = dict(length=7., mobility=.92, delay=.85, accel=1.2, scale=3., tip_scale=3.5)
# a lock in three segments (hair.locks.<lock>.mid): a four-point pendulum, each
# point driving one segment, the lower ones relative to the one above, so a
# swing runs from the root to the tip. Per point: length, mobility, delay,
# acceleration, output scale. Set against VTube Studio's own sample models
# (their hair: delay .8-.95, mobility .9-.96, outputs well inside the
# parameter's range): an output that reaches the end of the range stops
# flat there before it swings back, and the hair looks stiff.
CHAIN = dict(length=(7., 7., 6.), mobility=(.9, .9, .88), delay=(.9, .9, .85), accel=(1.2, 1.1, 1.),
             scale=(2.8, 3., 2.))
# a dangle is a short pendulum: it settles faster than a lock of hair
DANGLE_PHYSICS = dict(length=6., mobility=.95, delay=.85, accel=1.2, scale=3.)
HEAD_INPUTS = [('ParamAngleX', 60, 'X'), ('ParamAngleZ', 100, 'Angle'), ('ParamBodyAngleZ', 40, 'Angle')]
BODY_INPUTS = [('ParamBodyAngleX', 100, 'X'), ('ParamBodyAngleZ', 100, 'Angle')]
# nodding: the hair lags and bounces up and down (ParamHairPitch). X type:
# only while the head moves, no lasting offset. Cubism negates an input
# unless reflected; Angle X and the swing point the same way on screen,
# Angle Y and the pitch opposite ways, so this one is reflected
PITCH_INPUTS = [('ParamAngleY', 70, 'X', True)]
# a short, quick pendulum: a nod of 16 degrees bounces the hair two or three
# times within half a second, at most 0.7-0.85 of the parameter's range
PITCH_PHYSICS = dict(length=6., mobility=.9, delay=.9, accel=1.2, scale=3.)


def _cfg(char):
    v = dict(char.get('vts', {}))
    v.setdefault('name', char['name'])
    v.setdefault('file_name', char.model_id)
    v.setdefault('folder', char.dir.name.replace('-', '_')+'_vts')
    return v


def mappings(params):
    return [m for m in MAPPINGS if m[2] in params]


def expressions(char, params):
    return [e for e in _cfg(char).get('expressions', EXPRESSIONS) if all(k in params for k in e[3])]


def gestures(char, files):
    return [g for g in _cfg(char).get('gestures', GESTURES) if g[1] in files]


def physics(char, params):
    def setting(pid, out, length, mobility, delay, accel, scale, inputs=HEAD_INPUTS):
        return {
            'Id': pid,
            'Input': [{'Source': {'Target': 'Parameter', 'Id': i[0]}, 'Weight': i[1], 'Type': i[2],
                       'Reflect': i[3] if len(i) > 3 else False} for i in inputs],
            'Output': [{'Destination': {'Target': 'Parameter', 'Id': out}, 'VertexIndex': 1,
                        'Scale': scale, 'Weight': 100, 'Type': 'Angle', 'Reflect': False}],
            'Vertices': [
                {'Position': {'X': 0, 'Y': 0}, 'Mobility': 1.0, 'Delay': 1.0, 'Acceleration': 1.0, 'Radius': 0.0},
                {'Position': {'X': 0, 'Y': length}, 'Mobility': mobility, 'Delay': delay,
                 'Acceleration': accel, 'Radius': length}],
            'Normalization': {'Position': {'Minimum': -10, 'Default': 0, 'Maximum': 10},
                              'Angle': {'Minimum': -10, 'Default': 0, 'Maximum': 10}}}

    def chain(pid, outs, c, inputs=HEAD_INPUTS):
        s = setting(pid, outs[0], c['length'][0], c['mobility'][0], c['delay'][0], c['accel'][0], c['scale'][0], inputs)
        y = c['length'][0]
        for k in range(1, len(c['length'])):
            y += c['length'][k]
            s['Vertices'].append({'Position': {'X': 0, 'Y': y}, 'Mobility': c['mobility'][k], 'Delay': c['delay'][k],
                                  'Acceleration': c['accel'][k], 'Radius': c['length'][k]})
            if outs[k]:
                s['Output'].append({'Destination': {'Target': 'Parameter', 'Id': outs[k]}, 'VertexIndex': k+1,
                                    'Scale': c['scale'][k], 'Weight': 100, 'Type': 'Angle', 'Reflect': False})
        return s

    sets = []
    for lock in char['hair']['locks'].values():
        if lock['param'] not in params:
            continue
        if lock.get('mid') in params:
            c = {**CHAIN, **lock.get('chain', {})}
            sets.append(chain(lock['param'].replace('Param', '', 1), [lock['param'], lock['mid'], lock.get('tip')], c))
            continue
        ph = {k: float(v) for k, v in {**PHYSICS, **lock.get('physics', {})}.items()}
        s = setting(lock['param'].replace('Param', '', 1), lock['param'], ph['length'], ph['mobility'],
                    ph['delay'], ph['accel'], ph['scale'])
        if lock.get('tip') in params:
            # three points: the first segment swings the lock, the second
            # (relative to the first) bends the tip, so the tip trails
            L = ph['length']
            s['Vertices'].append({'Position': {'X': 0, 'Y': 2*L}, 'Mobility': ph['mobility']*.97,
                                  'Delay': ph['delay']*.85, 'Acceleration': ph['accel']*1.15, 'Radius': L})
            s['Output'].append({'Destination': {'Target': 'Parameter', 'Id': lock['tip']}, 'VertexIndex': 2,
                                'Scale': ph['tip_scale'], 'Weight': 100, 'Type': 'Angle', 'Reflect': False})
        sets.append(s)
    if 'ParamHairPitch' in params and any(lk.get('pitch') for lk in char['hair']['locks'].values()):
        ph = PITCH_PHYSICS
        sets.append(setting('HairPitch', 'ParamHairPitch', ph['length'], ph['mobility'], ph['delay'],
                            ph['accel'], ph['scale'], PITCH_INPUTS))
    for d in char.get('dangles', {}).values():
        if d['param'] not in params:
            continue
        ph = {k: float(v) for k, v in {**DANGLE_PHYSICS, **d.get('physics', {})}.items()}
        sets.append(setting(d['param'].replace('Param', '', 1), d['param'], ph['length'], ph['mobility'], ph['delay'],
                            ph['accel'], ph['scale'], BODY_INPUTS if d.get('follows') == 'body' else HEAD_INPUTS))
    return {'Version': 3, 'Meta': {
        'PhysicsSettingCount': len(sets), 'TotalInputCount': sum(len(s['Input']) for s in sets),
        'TotalOutputCount': sum(len(s['Output']) for s in sets),
        'VertexCount': sum(len(s['Vertices']) for s in sets), 'Fps': 30,
        'EffectiveForces': {'Gravity': {'X': 0, 'Y': -1}, 'Wind': {'X': 0, 'Y': 0}},
        'PhysicsDictionary': [{'Id': s['Id'], 'Name': s['Id']} for s in sets]},
        'PhysicsSettings': sets}


def _param_setting(template, inp, a, b, out, c, d, smooth, breath):
    p = copy.deepcopy(template)
    p.update(Name=out, Input=inp, InputRangeLower=float(a), InputRangeUpper=float(b),
             OutputRangeLower=float(c), OutputRangeUpper=float(d), ClampInput=False,
             ClampOutput=False, UseBlinking=False, UseBreathing=breath, OutputLive2D=out,
             Smoothing=smooth, Minimized=False)
    return p


def vtube(char, template, params, exps):
    v = _cfg(char)
    data = copy.deepcopy(template)
    hotkey_template = next(h for h in template['Hotkeys'] if h['Action'] == 'ToggleExpression')
    param_template = template['ParameterSettings'][0]
    data.update(Name=v['name'], ModelID=uuid.uuid4().hex,
                FileReferences={'Icon': 'icon.png', 'Model': f"{v['file_name']}.model3.json",
                                'IdleAnimation': '', 'IdleAnimationWhenTrackingLost': ''},
                SavedActiveExpressions=[])
    data.pop('ModelSaveMetadata', None)
    data['SavedModelPosition'] = {'Position': {'x': 0., 'y': -20., 'z': 0.},
                                  'Rotation': {'x': 0., 'y': 0., 'z': 0., 'w': 1.},
                                  'Scale': {'x': 1., 'y': 1., 'z': 1.}}
    data['PhysicsSettings'].update(Use=True, UseLegacyPhysics=False)
    data['ParameterSettings'] = [_param_setting(param_template, inp, a, b, out, c, d, smooth, breath)
                                 for inp, (a, b), out, (c, d), smooth, breath in mappings(params)]
    data['Hotkeys'] = []
    for i, (file, label, key, _, _) in enumerate(exps):
        h = copy.deepcopy(hotkey_template)
        h.update(HotkeyID=uuid.uuid4().hex, Name=label, Action='ToggleExpression', File=f'{file}.exp3.json')
        h['Triggers'] = {'Trigger1': key, 'Trigger2': '', 'Trigger3': '', 'ScreenButton': i+1}
        data['Hotkeys'].append(h)
    h = copy.deepcopy(hotkey_template)
    h.update(HotkeyID=uuid.uuid4().hex, Name='清除表情', Action='RemoveAllExpressions', File='')
    h['Triggers'] = {'Trigger1': 'N0', 'Trigger2': '', 'Trigger3': '', 'ScreenButton': len(exps)+1}
    data['Hotkeys'].append(h)
    for name, file, left, right, combinator, seconds in gestures(char, {e[0] for e in exps}):
        h = copy.deepcopy(hotkey_template)
        h.update(HotkeyID=uuid.uuid4().hex, Name=name, Action='ToggleExpression', File=f'{file}.exp3.json')
        h['Triggers'] = {'Trigger1': '', 'Trigger2': '', 'Trigger3': '', 'ScreenButton': -1}
        h['HandGestureSettings'].update(GestureLeft=left, GestureRight=right, GestureCombinator=combinator,
                                        AllowMirroredGesture=True, DeactivateExpWhenGestureNotDetected=True,
                                        SecondsUntilDetection=seconds)
        data['Hotkeys'].append(h)
    return data


def package(char, runtime, out):
    from .render import render
    v = _cfg(char)
    runtime, out = Path(runtime), Path(out)
    src, name = char.model_id, v['file_name']
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(runtime/f'{src}.moc3', out/f'{name}.moc3')
    shutil.copy2(runtime/'texture_00.png', out/'texture_00.png')
    shutil.copy2(runtime/f'{src}.cdi3.json', out/f'{name}.cdi3.json')
    params = [p['Id'] for p in json.loads((runtime/f'{src}.cdi3.json').read_text())['Parameters']]
    exps = expressions(char, params)
    (out/f'{name}.physics3.json').write_text(json.dumps(physics(char, params), indent=1))
    for file, _, _, values, fade in exps:
        (out/f'{file}.exp3.json').write_text(json.dumps({
            'Type': 'Live2D Expression', 'FadeInTime': fade, 'FadeOutTime': fade,
            'Parameters': [{'Id': k, 'Value': val, 'Blend': 'Overwrite'} for k, val in values.items()]}, indent=1))
    manifest = json.loads((runtime/f'{src}.model3.json').read_text())
    manifest['FileReferences'].update(
        Moc=f'{name}.moc3', DisplayInfo=f'{name}.cdi3.json', Physics=f'{name}.physics3.json',
        Expressions=[{'Name': f, 'File': f'{f}.exp3.json'} for f, *_ in exps])
    (out/f'{name}.model3.json').write_text(json.dumps(manifest, indent=1))
    template = json.loads((vts_models_dir()/'akari_vts/akari.vtube.json').read_text())
    (out/f'{name}.vtube.json').write_text(json.dumps(vtube(char, template, params, exps), ensure_ascii=False, indent=1))
    crop = v.get('icon_crop') or _head_box(char)
    icon = render(runtime, f'{src}.model3.json', {}, None, (246, 242, 238)).crop(tuple(crop))
    icon.convert('RGB').resize((256, 256), Image.LANCZOS).save(out/'icon.png')
    return out


def _head_box(char):
    """Square around the face, with room for the hair."""
    f = char['head']['face']
    cx, cy = f['centre']
    r = 2*max(f['rx'], f['ry_up'])
    return [int(cx-r), int(cy-r), int(cx+r), int(cy+r)]


def merge_settings(saved, package_data, params):
    """Keep everything the user tuned in VTS; add only hotkeys (by name) and
    mappings (by output) the saved file lacks."""
    added = []
    have = {h['Name'] for h in saved['Hotkeys']}
    new = [h for h in package_data['Hotkeys'] if h['Name'] not in have]
    used = max([h['Triggers']['ScreenButton'] for h in saved['Hotkeys']]+[0])
    for h in new:
        if h['Triggers']['ScreenButton'] > 0:
            used += 1
            h['Triggers']['ScreenButton'] = used
    saved['Hotkeys'] += new
    added += [h['Name'] for h in new]
    outs = {p['OutputLive2D'] for p in saved['ParameterSettings']}
    template = saved['ParameterSettings'][0]
    for inp, (a, b), out, (c, d), smooth, breath in mappings(params):
        if out not in outs:
            saved['ParameterSettings'].append(_param_setting(template, inp, a, b, out, c, d, smooth, breath))
            added.append(out)
    return saved, added


def vts_running():
    try:
        r = subprocess.run(['pgrep', '-f', 'VTubeStudio'], capture_output=True, text=True)
        return bool(r.stdout.strip())
    except FileNotFoundError:
        return False


def install(char, pkg, settings_from=None):
    """Copy a packaged folder into VTS. settings_from: another VTS model's
    vtube.json whose tuning the new model starts from."""
    v = _cfg(char)
    dest = vts_models_dir()/v['folder']
    if dest.exists() and vts_running():
        raise SystemExit('VTube Studio 正在运行：退出后再安装（它退出时会把设置写回这个模型）')
    dest.mkdir(parents=True, exist_ok=True)
    name = v['file_name']
    saved_path = dest/f'{name}.vtube.json'
    params = [p['Id'] for p in json.loads((pkg/f'{name}.cdi3.json').read_text())['Parameters']]
    fresh = json.loads((pkg/f'{name}.vtube.json').read_text())
    base = None
    if saved_path.exists():
        base = json.loads(saved_path.read_text())
        shutil.copy2(saved_path, saved_path.with_suffix('.json.bak'))
    elif settings_from:
        base = json.loads(Path(settings_from).read_text())
        base.update(Name=fresh['Name'], ModelID=fresh['ModelID'], FileReferences=fresh['FileReferences'],
                    SavedActiveExpressions=[])
    for f in pkg.iterdir():
        if f.name != saved_path.name:
            shutil.copy2(f, dest/f.name)
    if base is None:
        shutil.copy2(pkg/saved_path.name, saved_path)
        return dest, ['新建设置']
    merged, added = merge_settings(base, fresh, params)
    saved_path.write_text(json.dumps(merged, ensure_ascii=False, indent=1))
    return dest, added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('character')
    ap.add_argument('runtime', nargs='?')
    ap.add_argument('--install', action='store_true')
    ap.add_argument('--settings-from', help='start from this vtube.json (another model of the same character)')
    a = ap.parse_args()
    char = Character(a.character)
    runtime = Path(a.runtime) if a.runtime else char.dir/'build/runtime'
    pkg = package(char, runtime, char.dir/'vts'/_cfg(char)['folder'])
    print(pkg)
    if a.install:
        dest, added = install(char, pkg, a.settings_from)
        print(dest, '；'.join(added) or '设置未改动')


if __name__ == '__main__':
    main()
