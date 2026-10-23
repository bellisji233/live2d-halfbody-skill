"""Build a character's model: character.json + prepared meshes -> runtime folder.

  python -m rigkit.build CHARACTER_JSON [OUT_DIR]

OUT_DIR defaults to <character folder>/build/. The runtime folder
holds <model_id>.moc3, model3/cdi3 manifests and texture_00.png, ready for
VTube Studio packaging (rigkit.vts) or the checks.
"""
from pathlib import Path
import shutil, sys, time

from .config import Character
from .rig import Rig
from . import moc


def build(char_path, out=None, log=print):
    char = Character(char_path)
    out = Path(out) if out else char.dir/'build'
    runtime = out/'runtime'
    meshes = char.meshes()['meshes']
    rig = Rig(char, [m['name'] for m in meshes])
    t = time.time()
    moc.write(rig, meshes, runtime, char.model_id, char.canvas, log=log)
    shutil.copy2(char.prepared/'texture_00.png', runtime/'texture_00.png')
    log(f'{time.time()-t:.0f} s -> {runtime}')
    return runtime


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
