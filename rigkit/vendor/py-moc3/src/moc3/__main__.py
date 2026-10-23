"""CLI entry point: python -m moc3 <command> <file.moc3>"""

from __future__ import annotations

import argparse
import json
import sys

from moc3 import Moc3, CountIdx


def cmd_info(args: argparse.Namespace) -> None:
    moc = Moc3.from_file(args.file)
    print(moc.summary())
    print()
    print(f"Part IDs ({len(moc.part_ids)}):")
    for pid in moc.part_ids:
        print(f"  {pid}")
    print()
    print(f"Parameter IDs ({len(moc.parameter_ids)}):")
    for pid in moc.parameter_ids:
        print(f"  {pid}")
    print()
    print(f"ArtMesh IDs ({len(moc.art_mesh_ids)}):")
    for mid in moc.art_mesh_ids:
        print(f"  {mid}")


def cmd_params(args: argparse.Namespace) -> None:
    moc = Moc3.from_file(args.file)
    params = []
    for i, pid in enumerate(moc.parameter_ids):
        params.append({
            "id": pid,
            "min": moc["parameter.min_values"][i],
            "max": moc["parameter.max_values"][i],
            "default": moc["parameter.default_values"][i],
        })
    if args.json:
        print(json.dumps(params, indent=2))
    else:
        for p in params:
            print(f"  {p['id']:40s}  [{p['min']:8.2f}, {p['max']:8.2f}]  default={p['default']:.2f}")


def cmd_meshes(args: argparse.Namespace) -> None:
    moc = Moc3.from_file(args.file)
    meshes = []
    for i, mid in enumerate(moc.art_mesh_ids):
        meshes.append({
            "id": mid,
            "texture": moc["art_mesh.texture_indices"][i],
            "vertices": moc["art_mesh.vertex_counts"][i],
            "indices": moc["art_mesh.position_index_counts"][i],
        })
    if args.json:
        print(json.dumps(meshes, indent=2))
    else:
        for m in meshes:
            print(f"  {m['id']:40s}  tex={m['texture']}  verts={m['vertices']:4d}  idx={m['indices']:5d}")


def cmd_verify(args: argparse.Namespace) -> None:
    """Read a moc3 file, write it back, and verify byte-identical output."""
    from pathlib import Path
    import hashlib

    original = Path(args.file).read_bytes()
    moc = Moc3.from_bytes(original)
    written = moc.to_bytes()

    h1 = hashlib.sha256(original).hexdigest()
    h2 = hashlib.sha256(written).hexdigest()

    print(f"Original: {len(original)} bytes  SHA256: {h1[:16]}...")
    print(f"Written:  {len(written)} bytes  SHA256: {h2[:16]}...")

    if original == written:
        print("PASS: byte-identical round-trip")
    else:
        print("FAIL: output differs from input")
        min_len = min(len(original), len(written))
        diffs = sum(1 for i in range(min_len) if original[i] != written[i])
        print(f"  {diffs} differing bytes out of {min_len}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="moc3",
        description="Inspect and modify Live2D Cubism .moc3 files",
    )
    sub = parser.add_subparsers(dest="command")

    p_info = sub.add_parser("info", help="Show model summary")
    p_info.add_argument("file", help="Path to .moc3 file")

    p_params = sub.add_parser("params", help="List parameters")
    p_params.add_argument("file", help="Path to .moc3 file")
    p_params.add_argument("--json", action="store_true", help="Output as JSON")

    p_meshes = sub.add_parser("meshes", help="List art meshes")
    p_meshes.add_argument("file", help="Path to .moc3 file")
    p_meshes.add_argument("--json", action="store_true", help="Output as JSON")

    p_verify = sub.add_parser("verify", help="Verify read-write round-trip")
    p_verify.add_argument("file", help="Path to .moc3 file")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    cmds = {"info": cmd_info, "params": cmd_params, "meshes": cmd_meshes, "verify": cmd_verify}
    cmds[args.command](args)


if __name__ == "__main__":
    main()
