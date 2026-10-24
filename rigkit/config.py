"""One character = one folder with character.json.

character.json holds everything that belongs to that drawing: where the
face, eyes, mouth, neck and waist are, which layer is which, and how far
each motion goes. The code in rigkit holds only what is the same for every
character. Paths in the file are relative to the file.

The art folder has layout.json (layer name, texture, group) and parts/.
The prepared folder has meshes.json (final drawables: vertices, uvs,
triangles, draw order) and texture_00.png (the atlas the uvs point into).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class Layer:
    name: str
    group: str            # head | neck | body | pose
    role: str             # see infer_role
    side: str | None = None
    variant: str | None = None
    lock: str | None = None       # hair: the lock whose swing and weights it shares
    neck_hold: str = 'hold'       # hair: none | hold | behind_jaw
    warp: str | None = None       # deformer group under the body warp


class Character:
    def __init__(self, path):
        self.file = Path(path).resolve()
        self.dir = self.file.parent
        self.data = json.loads(self.file.read_text())
        self.art = (self.dir/self.data['art']).resolve()
        self.prepared = (self.dir/self.data.get('prepared', 'prepared')).resolve()
        self.canvas = tuple(self.data['canvas'])
        self.layout = json.loads((self.art/'layout.json').read_text())
        self.model_id = self.data['model_id']

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def art_path(self, rel):
        return self.art/rel

    def part(self, name):
        """Texture path of an original layer, from layout.json."""
        for l in self.layout['layers']:
            if l['name'] == name:
                return self.art/l['texture']
        return self.art/'parts'/f'{name}.png'

    def meshes(self):
        return json.loads((self.prepared/'meshes.json').read_text())

    def layers(self, names):
        """Layer table for the drawables in names (prepared draw order)."""
        groups = {l['name']: l['group'] for l in self.layout['layers']}
        override = self.data.get('layers', {})
        locks = self.data['hair']['locks']
        out = {}
        for nm in names:
            info = infer_role(nm, locks)
            info.update(override.get(nm, {}))
            group = info.pop('group', None) or groups.get(nm) or 'head'
            if info['role'] != 'dangle' and (info['role'] in ('body', 'pose') or group in ('body', 'pose')):
                info['role'] = group
            layer = Layer(name=nm, group=group, **info)
            if layer.warp is None:
                layer.warp = default_warp(layer)
                if layer.lock and locks[layer.lock].get('body_from'):
                    layer.warp = 'HairLong'                 # handed over to the body: not one grid with the locks
            out[nm] = layer
        return out


def _side(name):
    return name[-1] if name[-2:] in ('_L', '_R') else None


def infer_role(name, locks):
    """Role from the standard layer names (see the skill's layer list)."""
    side = _side(name)
    eye = {'Eye_Open_': 'eye_lash', 'Eye_White_': 'eye_white', 'Eye_Crease_': 'eye_crease',
           'Iris_': 'iris', 'Eye_Lower_': 'eye_lower', 'Eye_Closed_': 'eye_closed',
           'Eye_Smile_': 'eye_smile', 'Eye_Mask_': 'eye_mask'}
    for prefix, role in eye.items():
        if name.startswith(prefix):
            return dict(role=role, side=side)
    if name.startswith('Brow_'):
        return dict(role='brow', side=side, variant=name.split('_')[1])
    if name.startswith('Mouth_'):
        return dict(role='mouth', variant=name.split('_')[1])
    if name.startswith('Nose_Turn_'):
        return dict(role='nose_turn', side=side)
    if name.startswith('Nose'):
        return dict(role='nose')
    if name.startswith('Cheek_Flush_'):
        return dict(role='cheek', side=side)
    if name.startswith('Dangle_'):
        return dict(role='dangle')
    if name in locks:
        if name.startswith(('Hair_Back', 'Hair_Tail_')):
            # behind the head: swings with its own lock, never held by the neck
            return dict(role='hair', lock=name, side=side, neck_hold='none')
        return dict(role='hair', lock=name, side=side)
    if name.startswith('Hair_Back'):
        # behind the head: turns with it, no swing, not held by the neck
        return dict(role='hair', neck_hold='none', warp='Face')
    if name == 'Hair_Front':
        # along the hairline (no bangs): turns with the face it borders, no swing
        return dict(role='hair', neck_hold='none', warp='Face')
    if name.startswith('Hair_'):
        return dict(role='hair', side=side)
    if name == 'Neck':
        return dict(role='neck')
    return dict(role='face', side=side)


def default_warp(layer):
    """Deformer group: features that move as one piece get their own warp."""
    if layer.group != 'head':
        return None
    if layer.role.startswith(('eye', 'iris')):
        return 'Eye_'+layer.side
    if layer.role == 'brow':
        return 'Brow_'+layer.side
    if layer.role == 'mouth':
        return 'Mouth'
    if layer.role in ('nose', 'nose_turn'):
        return 'Nose'
    if layer.role == 'hair':
        return 'Hair'
    return 'Face'
