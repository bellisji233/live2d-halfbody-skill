"""
MOC3 binary format reader/writer for Live2D Cubism models.

Format layout:
  [0..64)     Header: "MOC3" magic + version + endian flag + padding
  [64..704)   Section Offset Table: 160 x uint32 (640 bytes)
  [704..832)  Count Info Table: 23 x uint32 + padding (128 bytes)
  [832..1984) Reserved / padding
  [1984..)    Body sections (each 64-byte aligned, order matches SOT)

The body contains ~100 typed arrays (struct-of-arrays layout). Each SOT entry
points to one array. The Count Info table provides element counts for each
logical section group (parts, deformers, art meshes, parameters, etc.).

Usage:
    from moc3 import Moc3

    moc = Moc3.from_file("model.moc3")
    print(moc.summary())
    print(moc.parameter_ids)
    moc["art_mesh.texture_indices"][0] = 1
    moc.to_file("model_modified.moc3")
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAGIC = b"MOC3"
HEADER_SIZE = 64
SOT_SIZE = 640  # 160 x uint32
SOT_COUNT = 160
COUNT_INFO_SIZE = 128  # 32 x uint32 (only 23 used)
COUNT_INFO_MAX = 23
DEFAULT_OFFSET = 1984  # body starts here
ALIGN = 64


class MocVersion(IntEnum):
    V3_00 = 1
    V3_03 = 2
    V4_00 = 3
    V4_02 = 4
    V5_00 = 5


# ---------------------------------------------------------------------------
# Count indices — which counts[i] maps to which section group
# ---------------------------------------------------------------------------

class CountIdx:
    PARTS = 0
    DEFORMERS = 1
    WARP_DEFORMERS = 2
    ROTATION_DEFORMERS = 3
    ART_MESHES = 4
    PARAMETERS = 5
    PART_KEYFORMS = 6
    WARP_DEFORMER_KEYFORMS = 7
    ROTATION_DEFORMER_KEYFORMS = 8
    ART_MESH_KEYFORMS = 9
    KEYFORM_POSITIONS = 10
    KEYFORM_BINDING_INDICES = 11
    KEYFORM_BINDING_BANDS = 12
    KEYFORM_BINDINGS = 13
    KEYS = 14
    UVS = 15
    POSITION_INDICES = 16
    DRAWABLE_MASKS = 17
    DRAW_ORDER_GROUPS = 18
    DRAW_ORDER_GROUP_OBJECTS = 19
    GLUES = 20
    GLUE_INFOS = 21
    GLUE_KEYFORMS = 22


# ---------------------------------------------------------------------------
# Binary helpers
# ---------------------------------------------------------------------------

def _align_offset(pos: int, alignment: int) -> int:
    """Round up pos to next multiple of alignment."""
    if alignment <= 0:
        return pos
    rem = pos % alignment
    return pos if rem == 0 else pos + (alignment - rem)


class BinaryReader:
    """Little-endian binary reader over a bytes buffer."""

    __slots__ = ("_buf", "_pos")

    def __init__(self, data: bytes):
        self._buf = data
        self._pos = 0

    @property
    def pos(self) -> int:
        return self._pos

    @pos.setter
    def pos(self, v: int):
        self._pos = v

    @property
    def remaining(self) -> int:
        return len(self._buf) - self._pos

    def read_bytes(self, n: int) -> bytes:
        out = self._buf[self._pos : self._pos + n]
        self._pos += n
        return out

    def read_u1(self) -> int:
        v = self._buf[self._pos]
        self._pos += 1
        return v

    def read_i32(self) -> int:
        v = struct.unpack_from("<i", self._buf, self._pos)[0]
        self._pos += 4
        return v

    def read_u32(self) -> int:
        v = struct.unpack_from("<I", self._buf, self._pos)[0]
        self._pos += 4
        return v

    def read_f32(self) -> float:
        v = struct.unpack_from("<f", self._buf, self._pos)[0]
        self._pos += 4
        return v

    def read_bool(self) -> bool:
        return self.read_i32() == 1

    def read_i32_array(self, n: int) -> list[int]:
        fmt = f"<{n}i"
        vals = struct.unpack_from(fmt, self._buf, self._pos)
        self._pos += 4 * n
        return list(vals)

    def read_u32_array(self, n: int) -> list[int]:
        fmt = f"<{n}I"
        vals = struct.unpack_from(fmt, self._buf, self._pos)
        self._pos += 4 * n
        return list(vals)

    def read_f32_array(self, n: int) -> list[float]:
        fmt = f"<{n}f"
        vals = struct.unpack_from(fmt, self._buf, self._pos)
        self._pos += 4 * n
        return list(vals)

    def read_i16_array(self, n: int) -> list[int]:
        fmt = f"<{n}h"
        vals = struct.unpack_from(fmt, self._buf, self._pos)
        self._pos += 2 * n
        return list(vals)

    def read_u8_array(self, n: int) -> list[int]:
        vals = list(self._buf[self._pos : self._pos + n])
        self._pos += n
        return vals

    def read_bool_array(self, n: int) -> list[bool]:
        return [v == 1 for v in self.read_i32_array(n)]

    def read_string(self, size: int) -> str:
        raw = self._buf[self._pos : self._pos + size]
        self._pos += size
        # null-terminated within fixed-size field
        idx = raw.find(0)
        if idx >= 0:
            raw = raw[:idx]
        return raw.decode("utf-8", errors="replace")

    def read_string_array(self, n: int, field_size: int = 64) -> list[str]:
        return [self.read_string(field_size) for _ in range(n)]

    def skip(self, n: int):
        self._pos += n


class BinaryWriter:
    """Little-endian binary writer to a bytearray."""

    __slots__ = ("_buf",)

    def __init__(self):
        self._buf = bytearray()

    @property
    def pos(self) -> int:
        return len(self._buf)

    def write_bytes(self, data: bytes | bytearray):
        self._buf.extend(data)

    def write_u1(self, v: int):
        self._buf.append(v & 0xFF)

    def write_i16(self, v: int):
        self._buf.extend(struct.pack("<h", v))

    def write_i32(self, v: int):
        self._buf.extend(struct.pack("<i", v))

    def write_u32(self, v: int):
        self._buf.extend(struct.pack("<I", v))

    def write_f32(self, v: float):
        self._buf.extend(struct.pack("<f", v))

    def write_bool(self, v: bool):
        self.write_i32(1 if v else 0)

    def write_i32_array(self, vals):
        self._buf.extend(struct.pack(f"<{len(vals)}i", *vals))

    def write_u32_array(self, vals):
        self._buf.extend(struct.pack(f"<{len(vals)}I", *vals))

    def write_f32_array(self, vals):
        self._buf.extend(struct.pack(f"<{len(vals)}f", *vals))

    def write_i16_array(self, vals):
        self._buf.extend(struct.pack(f"<{len(vals)}h", *vals))

    def write_u8_array(self, vals):
        self._buf.extend(bytes(vals))

    def write_bool_array(self, vals):
        self.write_i32_array([1 if v else 0 for v in vals])

    def write_string(self, s: str, field_size: int = 64):
        encoded = s.encode("utf-8")
        if len(encoded) >= field_size:
            raise ValueError(f"String '{s}' too long for field size {field_size}")
        self._buf.extend(encoded)
        self._buf.extend(b"\x00" * (field_size - len(encoded)))

    def write_string_array(self, vals: list[str], field_size: int = 64):
        for s in vals:
            self.write_string(s, field_size)

    def pad_to(self, alignment: int):
        target = _align_offset(self.pos, alignment)
        if target > self.pos:
            self._buf.extend(b"\x00" * (target - self.pos))

    def fill(self, count: int, value: int = 0):
        self._buf.extend(bytes([value]) * count)

    def get_bytes(self) -> bytes:
        return bytes(self._buf)


# ---------------------------------------------------------------------------
# Data types — element type enum for section entries
# ---------------------------------------------------------------------------

class ElemType:
    I32 = "i32"
    F32 = "f32"
    I16 = "i16"
    U8 = "u8"
    BOOL = "bool"      # stored as i32 (0 or 1)
    STR64 = "str64"    # 64-byte fixed-width null-terminated string
    RUNTIME = "runtime"  # runtime space (zeroed, skipped on read)


# Element size in bytes
ELEM_SIZES = {
    ElemType.I32: 4,
    ElemType.F32: 4,
    ElemType.I16: 2,
    ElemType.U8: 1,
    ElemType.BOOL: 4,
    ElemType.STR64: 64,
}


# ---------------------------------------------------------------------------
# Section layout definition
# ---------------------------------------------------------------------------
# Each tuple: (name, elem_type, count_index, alignment)
# count_index: index into counts[] for element count, or "size_fn" for runtime space
# alignment: byte alignment before this section (0 = none, 64 = default for data)
#
# Order matches CMocMemoryMapperV1.initializeMemoryMap() exactly.
# The SOT stores offsets for entries starting at index 1 (index 0 is unused/count_info).

@dataclass
class SectionEntry:
    name: str
    elem_type: str
    count_idx: int  # index into counts[], -1 = special
    align: int = 0  # 0 = no alignment, >0 = pad to this before writing
    group: str = ""  # logical group name


# Alignment rules from Java decompilation:
# - RuntimeSpaceEntry: align=64
# - IdEntry (STR64): align=0
# - ArrayDataEntry (everything else): align=64
_A = 64

# fmt: off
SECTION_LAYOUT: list[SectionEntry] = [
    # -- Count Info + Canvas Info are handled specially (not in this list) --

    # EmPartSources (count_idx=0)
    SectionEntry("part.runtime_space",      ElemType.RUNTIME, CountIdx.PARTS, _A, "part"),
    SectionEntry("part.ids",                ElemType.STR64, CountIdx.PARTS, 0, "part"),
    SectionEntry("part.keyform_binding_band_indices", ElemType.I32, CountIdx.PARTS, _A, "part"),
    SectionEntry("part.keyform_begin_indices", ElemType.I32, CountIdx.PARTS, _A, "part"),
    SectionEntry("part.keyform_counts",     ElemType.I32, CountIdx.PARTS, _A, "part"),
    SectionEntry("part.visibles",           ElemType.BOOL, CountIdx.PARTS, _A, "part"),
    SectionEntry("part.enables",            ElemType.BOOL, CountIdx.PARTS, _A, "part"),
    SectionEntry("part.parent_part_indices", ElemType.I32, CountIdx.PARTS, _A, "part"),

    # EmDeformerSources (count_idx=1)
    SectionEntry("deformer.runtime_space",  ElemType.RUNTIME, CountIdx.DEFORMERS, _A, "deformer"),
    SectionEntry("deformer.ids",            ElemType.STR64, CountIdx.DEFORMERS, 0, "deformer"),
    SectionEntry("deformer.keyform_binding_band_indices", ElemType.I32, CountIdx.DEFORMERS, _A, "deformer"),
    SectionEntry("deformer.visibles",       ElemType.BOOL, CountIdx.DEFORMERS, _A, "deformer"),
    SectionEntry("deformer.enables",        ElemType.BOOL, CountIdx.DEFORMERS, _A, "deformer"),
    SectionEntry("deformer.parent_part_indices", ElemType.I32, CountIdx.DEFORMERS, _A, "deformer"),
    SectionEntry("deformer.parent_deformer_indices", ElemType.I32, CountIdx.DEFORMERS, _A, "deformer"),
    SectionEntry("deformer.types",          ElemType.I32, CountIdx.DEFORMERS, _A, "deformer"),  # EmDeformerType enum as i32
    SectionEntry("deformer.specific_indices", ElemType.I32, CountIdx.DEFORMERS, _A, "deformer"),

    # EmWarpDeformerSpecificSources (count_idx=2)
    SectionEntry("warp_deformer.keyform_binding_band_indices", ElemType.I32, CountIdx.WARP_DEFORMERS, _A, "warp_deformer"),
    SectionEntry("warp_deformer.keyform_begin_indices", ElemType.I32, CountIdx.WARP_DEFORMERS, _A, "warp_deformer"),
    SectionEntry("warp_deformer.keyform_counts", ElemType.I32, CountIdx.WARP_DEFORMERS, _A, "warp_deformer"),
    SectionEntry("warp_deformer.vertex_counts", ElemType.I32, CountIdx.WARP_DEFORMERS, _A, "warp_deformer"),
    SectionEntry("warp_deformer.rows",      ElemType.I32, CountIdx.WARP_DEFORMERS, _A, "warp_deformer"),
    SectionEntry("warp_deformer.cols",      ElemType.I32, CountIdx.WARP_DEFORMERS, _A, "warp_deformer"),

    # EmRotationDeformerSpecificSources (count_idx=3)
    SectionEntry("rotation_deformer.keyform_binding_band_indices", ElemType.I32, CountIdx.ROTATION_DEFORMERS, _A, "rotation_deformer"),
    SectionEntry("rotation_deformer.keyform_begin_indices", ElemType.I32, CountIdx.ROTATION_DEFORMERS, _A, "rotation_deformer"),
    SectionEntry("rotation_deformer.keyform_counts", ElemType.I32, CountIdx.ROTATION_DEFORMERS, _A, "rotation_deformer"),
    SectionEntry("rotation_deformer.base_angles", ElemType.F32, CountIdx.ROTATION_DEFORMERS, _A, "rotation_deformer"),

    # EmArtMeshSources (count_idx=4)
    SectionEntry("art_mesh.runtime_space_0", ElemType.RUNTIME, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.runtime_space_1", ElemType.RUNTIME, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.runtime_space_2", ElemType.RUNTIME, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.runtime_space_3", ElemType.RUNTIME, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.ids",            ElemType.STR64, CountIdx.ART_MESHES, 0, "art_mesh"),
    SectionEntry("art_mesh.keyform_binding_band_indices", ElemType.I32, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.keyform_begin_indices", ElemType.I32, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.keyform_counts", ElemType.I32, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.visibles",       ElemType.BOOL, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.enables",        ElemType.BOOL, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.parent_part_indices", ElemType.I32, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.parent_deformer_indices", ElemType.I32, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.texture_indices", ElemType.I32, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.drawable_flags", ElemType.U8, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.position_index_counts", ElemType.I32, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.uv_begin_indices", ElemType.I32, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.position_index_begin_indices", ElemType.I32, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.vertex_counts",  ElemType.I32, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.mask_begin_indices", ElemType.I32, CountIdx.ART_MESHES, _A, "art_mesh"),
    SectionEntry("art_mesh.mask_counts",    ElemType.I32, CountIdx.ART_MESHES, _A, "art_mesh"),

    # EmParameterSources (count_idx=5)
    SectionEntry("parameter.runtime_space", ElemType.RUNTIME, CountIdx.PARAMETERS, _A, "parameter"),
    SectionEntry("parameter.ids",           ElemType.STR64, CountIdx.PARAMETERS, 0, "parameter"),
    SectionEntry("parameter.max_values",    ElemType.F32, CountIdx.PARAMETERS, _A, "parameter"),
    SectionEntry("parameter.min_values",    ElemType.F32, CountIdx.PARAMETERS, _A, "parameter"),
    SectionEntry("parameter.default_values", ElemType.F32, CountIdx.PARAMETERS, _A, "parameter"),
    SectionEntry("parameter.repeats",       ElemType.BOOL, CountIdx.PARAMETERS, _A, "parameter"),
    SectionEntry("parameter.decimal_places", ElemType.I32, CountIdx.PARAMETERS, _A, "parameter"),
    SectionEntry("parameter.keyform_binding_begin_indices", ElemType.I32, CountIdx.PARAMETERS, _A, "parameter"),
    SectionEntry("parameter.keyform_binding_counts", ElemType.I32, CountIdx.PARAMETERS, _A, "parameter"),

    # EmPartKeyformSources (count_idx=6)
    SectionEntry("part_keyform.draw_orders", ElemType.F32, CountIdx.PART_KEYFORMS, _A, "part_keyform"),

    # EmWarpDeformerKeyformSources (count_idx=7)
    SectionEntry("warp_deformer_keyform.opacities", ElemType.F32, CountIdx.WARP_DEFORMER_KEYFORMS, _A, "warp_deformer_keyform"),
    SectionEntry("warp_deformer_keyform.keyform_position_begin_indices", ElemType.I32, CountIdx.WARP_DEFORMER_KEYFORMS, _A, "warp_deformer_keyform"),

    # EmRotationDeformerKeyformSources (count_idx=8)
    SectionEntry("rotation_deformer_keyform.opacities", ElemType.F32, CountIdx.ROTATION_DEFORMER_KEYFORMS, _A, "rotation_deformer_keyform"),
    SectionEntry("rotation_deformer_keyform.angles", ElemType.F32, CountIdx.ROTATION_DEFORMER_KEYFORMS, _A, "rotation_deformer_keyform"),
    SectionEntry("rotation_deformer_keyform.origin_xs", ElemType.F32, CountIdx.ROTATION_DEFORMER_KEYFORMS, _A, "rotation_deformer_keyform"),
    SectionEntry("rotation_deformer_keyform.origin_ys", ElemType.F32, CountIdx.ROTATION_DEFORMER_KEYFORMS, _A, "rotation_deformer_keyform"),
    SectionEntry("rotation_deformer_keyform.scales", ElemType.F32, CountIdx.ROTATION_DEFORMER_KEYFORMS, _A, "rotation_deformer_keyform"),
    SectionEntry("rotation_deformer_keyform.reflect_xs", ElemType.BOOL, CountIdx.ROTATION_DEFORMER_KEYFORMS, _A, "rotation_deformer_keyform"),
    SectionEntry("rotation_deformer_keyform.reflect_ys", ElemType.BOOL, CountIdx.ROTATION_DEFORMER_KEYFORMS, _A, "rotation_deformer_keyform"),

    # EmArtMeshKeyformSources (count_idx=9)
    SectionEntry("art_mesh_keyform.opacities", ElemType.F32, CountIdx.ART_MESH_KEYFORMS, _A, "art_mesh_keyform"),
    SectionEntry("art_mesh_keyform.draw_orders", ElemType.F32, CountIdx.ART_MESH_KEYFORMS, _A, "art_mesh_keyform"),
    SectionEntry("art_mesh_keyform.keyform_position_begin_indices", ElemType.I32, CountIdx.ART_MESH_KEYFORMS, _A, "art_mesh_keyform"),

    # EmKeyformPositionSources (count_idx=10)
    SectionEntry("keyform_position.xys",    ElemType.F32, CountIdx.KEYFORM_POSITIONS, _A, "keyform_position"),

    # EmKeyformBindingIndexSources (count_idx=11)
    SectionEntry("keyform_binding_index.indices", ElemType.I32, CountIdx.KEYFORM_BINDING_INDICES, _A, "keyform_binding_index"),

    # EmKeyformBindingBandSources (count_idx=12)
    SectionEntry("keyform_binding_band.begin_indices", ElemType.I32, CountIdx.KEYFORM_BINDING_BANDS, _A, "keyform_binding_band"),
    SectionEntry("keyform_binding_band.counts", ElemType.I32, CountIdx.KEYFORM_BINDING_BANDS, _A, "keyform_binding_band"),

    # EmKeyformBindingSources (count_idx=13)
    SectionEntry("keyform_binding.keys_begin_indices", ElemType.I32, CountIdx.KEYFORM_BINDINGS, _A, "keyform_binding"),
    SectionEntry("keyform_binding.keys_counts", ElemType.I32, CountIdx.KEYFORM_BINDINGS, _A, "keyform_binding"),

    # EmKeysSources (count_idx=14)
    SectionEntry("keys.values",             ElemType.F32, CountIdx.KEYS, _A, "keys"),

    # EmUvSources (count_idx=15)
    SectionEntry("uv.xys",                 ElemType.F32, CountIdx.UVS, _A, "uv"),

    # EmPositionIndexSources (count_idx=16)
    SectionEntry("position_index.indices",  ElemType.I16, CountIdx.POSITION_INDICES, _A, "position_index"),

    # EmDrawableMaskSources (count_idx=17)
    SectionEntry("drawable_mask.art_mesh_indices", ElemType.I32, CountIdx.DRAWABLE_MASKS, _A, "drawable_mask"),

    # EmDrawOrderGroupSources (count_idx=18)
    SectionEntry("draw_order_group.object_begin_indices", ElemType.I32, CountIdx.DRAW_ORDER_GROUPS, _A, "draw_order_group"),
    SectionEntry("draw_order_group.object_counts", ElemType.I32, CountIdx.DRAW_ORDER_GROUPS, _A, "draw_order_group"),
    SectionEntry("draw_order_group.object_total_counts", ElemType.I32, CountIdx.DRAW_ORDER_GROUPS, _A, "draw_order_group"),
    SectionEntry("draw_order_group.min_draw_orders", ElemType.I32, CountIdx.DRAW_ORDER_GROUPS, _A, "draw_order_group"),
    SectionEntry("draw_order_group.max_draw_orders", ElemType.I32, CountIdx.DRAW_ORDER_GROUPS, _A, "draw_order_group"),

    # EmDrawOrderGroupObjectSources (count_idx=19)
    SectionEntry("draw_order_group_object.types", ElemType.I32, CountIdx.DRAW_ORDER_GROUP_OBJECTS, _A, "draw_order_group_object"),  # enum as i32
    SectionEntry("draw_order_group_object.indices", ElemType.I32, CountIdx.DRAW_ORDER_GROUP_OBJECTS, _A, "draw_order_group_object"),
    SectionEntry("draw_order_group_object.group_indices", ElemType.I32, CountIdx.DRAW_ORDER_GROUP_OBJECTS, _A, "draw_order_group_object"),

    # EmGlueSources (count_idx=20)
    SectionEntry("glue.runtime_space",      ElemType.RUNTIME, CountIdx.GLUES, _A, "glue"),
    SectionEntry("glue.ids",                ElemType.STR64, CountIdx.GLUES, 0, "glue"),
    SectionEntry("glue.keyform_binding_band_indices", ElemType.I32, CountIdx.GLUES, _A, "glue"),
    SectionEntry("glue.keyform_begin_indices", ElemType.I32, CountIdx.GLUES, _A, "glue"),
    SectionEntry("glue.keyform_counts",     ElemType.I32, CountIdx.GLUES, _A, "glue"),
    SectionEntry("glue.art_mesh_index_as",  ElemType.I32, CountIdx.GLUES, _A, "glue"),
    SectionEntry("glue.art_mesh_index_bs",  ElemType.I32, CountIdx.GLUES, _A, "glue"),
    SectionEntry("glue.info_begin_indices", ElemType.I32, CountIdx.GLUES, _A, "glue"),
    SectionEntry("glue.info_counts",        ElemType.I32, CountIdx.GLUES, _A, "glue"),

    # EmGlueInfoSources (count_idx=21)
    SectionEntry("glue_info.weights",       ElemType.F32, CountIdx.GLUE_INFOS, _A, "glue_info"),
    SectionEntry("glue_info.position_indices", ElemType.I16, CountIdx.GLUE_INFOS, _A, "glue_info"),

    # EmGlueKeyformSources (count_idx=22)
    SectionEntry("glue_keyform.intensities", ElemType.F32, CountIdx.GLUE_KEYFORMS, _A, "glue_keyform"),
]

# V3.03+ additional section
ADDITIONAL_V303 = SectionEntry(
    "additional.quad_transforms", ElemType.BOOL, -1, 0, "additional"
)
# fmt: on

RUNTIME_UNIT_SIZE = 8  # bytes per runtime space element


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

@dataclass
class Moc3Header:
    version: int = MocVersion.V3_00
    endian: int = 0  # 0 = little-endian

    def to_bytes(self) -> bytes:
        w = BinaryWriter()
        w.write_bytes(MAGIC)
        w.write_u1(self.version)
        w.write_u1(self.endian)
        # pad to HEADER_SIZE
        w.fill(HEADER_SIZE - w.pos)
        return w.get_bytes()

    @classmethod
    def from_reader(cls, r: BinaryReader) -> Moc3Header:
        magic = r.read_bytes(4)
        if magic != MAGIC:
            raise ValueError(f"Not a MOC3 file (magic: {magic!r})")
        version = r.read_u1()
        endian = r.read_u1()
        # skip rest of header
        r.skip(HEADER_SIZE - 6)
        return cls(version=version, endian=endian)


# ---------------------------------------------------------------------------
# Canvas info
# ---------------------------------------------------------------------------

@dataclass
class CanvasInfo:
    pixels_per_unit: float = 1.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    canvas_width: float = 0.0
    canvas_height: float = 0.0
    canvas_flag: int = 0

    BODY_SIZE = 64  # padding after the 5 floats + 1 byte


# ---------------------------------------------------------------------------
# Main Moc3 class
# ---------------------------------------------------------------------------

class Moc3:
    """In-memory representation of a .moc3 file."""

    def __init__(self):
        self.header = Moc3Header()
        self.counts: list[int] = [0] * COUNT_INFO_MAX
        self.canvas = CanvasInfo()
        # Section data: name -> list of values
        self._sections: dict[str, list] = {}
        # Original SOT offsets (for debugging)
        self._sot_offsets: list[int] = []
        # Effective section layout (depends on version)
        self._layout: list[SectionEntry] = []

    def _build_layout(self) -> list[SectionEntry]:
        layout = list(SECTION_LAYOUT)
        if self.header.version >= MocVersion.V3_03:
            layout.append(ADDITIONAL_V303)
        return layout

    def get(self, name: str) -> list:
        """Get section data by name (e.g., 'art_mesh.ids')."""
        return self._sections[name]

    def set(self, name: str, data: list):
        """Set section data by name."""
        self._sections[name] = data

    def __getitem__(self, name: str) -> list:
        return self._sections[name]

    def __setitem__(self, name: str, data: list):
        self._sections[name] = data

    # -- Convenience accessors for common sections --

    @property
    def part_ids(self) -> list[str]:
        return self._sections.get("part.ids", [])

    @property
    def deformer_ids(self) -> list[str]:
        return self._sections.get("deformer.ids", [])

    @property
    def art_mesh_ids(self) -> list[str]:
        return self._sections.get("art_mesh.ids", [])

    @property
    def parameter_ids(self) -> list[str]:
        return self._sections.get("parameter.ids", [])

    # -- Read --

    @classmethod
    def from_file(cls, path: str | Path) -> Moc3:
        data = Path(path).read_bytes()
        return cls.from_bytes(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> Moc3:
        moc = cls()
        r = BinaryReader(data)

        # Header
        moc.header = Moc3Header.from_reader(r)

        # Section Offset Table
        moc._sot_offsets = r.read_u32_array(SOT_COUNT)

        # Seek to DEFAULT_OFFSET for body
        r.pos = DEFAULT_OFFSET

        # Build layout for this version
        moc._layout = moc._build_layout()

        # Count Info (first thing after DEFAULT_OFFSET in the read path)
        # But wait — the Java code positions at DEFAULT_OFFSET and then reads
        # count info + canvas info as the first two entries.
        # The SOT offsets[1..N] tell us where subsequent sections start.
        # offsets[0] is the count info offset (= DEFAULT_OFFSET usually).

        # Read Count Info
        count_data = r.read_i32_array(COUNT_INFO_MAX)
        moc.counts = count_data
        # Skip padding to fill COUNT_INFO_SIZE
        r.skip(COUNT_INFO_SIZE - COUNT_INFO_MAX * 4)

        # Read Canvas Info
        moc.canvas.pixels_per_unit = r.read_f32()
        moc.canvas.origin_x = r.read_f32()
        moc.canvas.origin_y = r.read_f32()
        moc.canvas.canvas_width = r.read_f32()
        moc.canvas.canvas_height = r.read_f32()
        moc.canvas.canvas_flag = r.read_u1()
        r.skip(CanvasInfo.BODY_SIZE - (5 * 4 + 1))

        # Read body sections
        # SOT layout: offsets[0]=countInfo, offsets[1]=canvasInfo,
        # offsets[2]=layout[0], offsets[3]=layout[1], ...
        # Java code uses offsets[i+1] as "next section start" after reading entry[i].
        # Our layout[i] = Java entry[i+2], so we seek to offsets[i+2] before reading.
        for i, entry in enumerate(moc._layout):
            sot_idx = i + 2  # this section's SOT slot
            if sot_idx < len(moc._sot_offsets):
                target = moc._sot_offsets[sot_idx]
                if target > 0 and r.pos < target:
                    r.pos = target

            count = _get_section_count(moc, entry)
            if count == 0:
                moc._sections[entry.name] = []
                continue

            moc._sections[entry.name] = _read_section(r, entry, count)

        return moc

    # -- Write --

    def to_bytes(self) -> bytes:
        self._layout = self._build_layout()

        # Phase 1: Write body sections, record offsets
        # Java write order: countInfo, canvasInfo, then layout entries
        body_writer = BinaryWriter()
        sot_entries: list[int] = []  # will become SOT offsets[0..N]

        # CountInfo — SOT[0]
        sot_entries.append(DEFAULT_OFFSET + body_writer.pos)
        body_writer.write_i32_array(self.counts)
        body_writer.fill(COUNT_INFO_SIZE - body_writer.pos)

        # CanvasInfo — SOT[1]
        sot_entries.append(DEFAULT_OFFSET + body_writer.pos)
        canvas_start = body_writer.pos
        body_writer.write_f32(self.canvas.pixels_per_unit)
        body_writer.write_f32(self.canvas.origin_x)
        body_writer.write_f32(self.canvas.origin_y)
        body_writer.write_f32(self.canvas.canvas_width)
        body_writer.write_f32(self.canvas.canvas_height)
        body_writer.write_u1(self.canvas.canvas_flag)
        body_writer.fill(CanvasInfo.BODY_SIZE - (body_writer.pos - canvas_start))

        # Body sections — SOT[2..]
        for entry in self._layout:
            # Align if needed
            if entry.align > 0:
                body_writer.pad_to(entry.align)

            sot_entries.append(DEFAULT_OFFSET + body_writer.pos)

            data = self._sections.get(entry.name, [])
            count = len(data) if entry.elem_type != ElemType.RUNTIME else _get_section_count(self, entry)
            _write_section(body_writer, entry, data, count)

        # Phase 2: Build header + SOT + padding, then append body
        out = BinaryWriter()

        # Header
        out.write_bytes(self.header.to_bytes())

        # SOT: pad to SOT_COUNT entries
        while len(sot_entries) < SOT_COUNT:
            sot_entries.append(0)
        out.write_u32_array(sot_entries[:SOT_COUNT])

        # Pad to DEFAULT_OFFSET
        out.fill(DEFAULT_OFFSET - out.pos)
        assert out.pos == DEFAULT_OFFSET

        # Append body
        out.write_bytes(body_writer.get_bytes())

        # Final 64-byte alignment
        out.pad_to(ALIGN)

        return out.get_bytes()

    def to_file(self, path: str | Path):
        Path(path).write_bytes(self.to_bytes())

    def summary(self) -> str:
        """Return a human-readable summary of the moc3 structure."""
        lines = [
            f"MOC3 v{self.header.version} ({'LE' if self.header.endian == 0 else 'BE'})",
            f"Canvas: {self.canvas.canvas_width}x{self.canvas.canvas_height} "
            f"ppu={self.canvas.pixels_per_unit}",
            f"Parts: {self.counts[CountIdx.PARTS]}",
            f"Deformers: {self.counts[CountIdx.DEFORMERS]}",
            f"  Warp: {self.counts[CountIdx.WARP_DEFORMERS]}",
            f"  Rotation: {self.counts[CountIdx.ROTATION_DEFORMERS]}",
            f"Art Meshes: {self.counts[CountIdx.ART_MESHES]}",
            f"Parameters: {self.counts[CountIdx.PARAMETERS]}",
            f"Keyform Positions: {self.counts[CountIdx.KEYFORM_POSITIONS]}",
            f"UVs: {self.counts[CountIdx.UVS]}",
            f"Position Indices: {self.counts[CountIdx.POSITION_INDICES]}",
            f"Glues: {self.counts[CountIdx.GLUES]}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section read/write helpers
# ---------------------------------------------------------------------------

def _get_section_count(moc: Moc3, entry: SectionEntry) -> int:
    """Get element count for a section entry."""
    if entry.count_idx == -1:
        # Additional sections: count = remaining bytes / elem_size
        # For the BOOL additional, use all deformers count (warp+rotation)
        # Actually in the Java code, count=-1 means read until EOF
        # For write, we use the length of the data list
        return len(moc._sections.get(entry.name, []))
    return moc.counts[entry.count_idx]


def _read_section(r: BinaryReader, entry: SectionEntry, count: int) -> list:
    """Read a typed array section."""
    if entry.elem_type == ElemType.RUNTIME:
        # Runtime space: skip count * RUNTIME_UNIT_SIZE bytes
        # But in the Java reader, it actually reads strings for debugging
        # We'll just skip
        r.skip(count * RUNTIME_UNIT_SIZE)
        return []
    elif entry.elem_type == ElemType.I32:
        return r.read_i32_array(count)
    elif entry.elem_type == ElemType.F32:
        return r.read_f32_array(count)
    elif entry.elem_type == ElemType.I16:
        return r.read_i16_array(count)
    elif entry.elem_type == ElemType.U8:
        return r.read_u8_array(count)
    elif entry.elem_type == ElemType.BOOL:
        return r.read_bool_array(count)
    elif entry.elem_type == ElemType.STR64:
        return r.read_string_array(count, 64)
    else:
        raise ValueError(f"Unknown element type: {entry.elem_type}")


def _write_section(w: BinaryWriter, entry: SectionEntry, data: list, count: int):
    """Write a typed array section."""
    if entry.elem_type == ElemType.RUNTIME:
        w.fill(count * RUNTIME_UNIT_SIZE)
    elif entry.elem_type == ElemType.I32:
        w.write_i32_array(data)
    elif entry.elem_type == ElemType.F32:
        w.write_f32_array(data)
    elif entry.elem_type == ElemType.I16:
        w.write_i16_array(data)
    elif entry.elem_type == ElemType.U8:
        w.write_u8_array(data)
    elif entry.elem_type == ElemType.BOOL:
        w.write_bool_array(data)
    elif entry.elem_type == ElemType.STR64:
        w.write_string_array(data, 64)
    else:
        raise ValueError(f"Unknown element type: {entry.elem_type}")


