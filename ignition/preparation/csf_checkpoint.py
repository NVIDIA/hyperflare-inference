# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Read lossless CSF scale containers without restoring expanded checkpoints.

A container keeps the source checkpoint's metadata in `metadata/` and its shards
in `tensors/`. Each compressed block-scale tensor is stored as a fixed plane and
an exception plane. Every other tensor keeps its source name and bytes.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from .checkpoint import CheckpointError, IndexedCheckpoint, _parse_json, _read_header


@dataclass(frozen=True, slots=True)
class CsfFormat:
    """One container schema: its codec, scale naming, and packed weight geometry."""

    quantization: str
    codec: str
    # Suffix of a logical block-scale tensor beside its packed `.weight`.
    scale_suffix: str
    weight_dtypes: tuple[str, ...]
    # Packed weight bytes per stored scale element in a row, and bits per encoded scale.
    packed_columns_per_scale: int
    scale_bits: int

    # The fixed plane holds one slab per sixteen scale rows: sixteen row bases followed by the
    # rows' packed codes, `16 * (1 + ceil(columns * scale_bits / 8))` bytes.
    def slab_bytes(self, columns: int) -> int:
        return 16 * (1 + (columns * self.scale_bits + 7) // 8)


FORMATS = MappingProxyType({
    "lil-mxfp4-csf-checkpoint/1": CsfFormat(
        "mxfp4", "row-base-offset1-u24-exceptions/1", ".scale", ("I8", "U8"), 16, 1),
    "lil-nvfp4-csf-checkpoint/1": CsfFormat(
        "nvfp4", "byte-window4-fixed-stream-u24-exceptions/1", ".weight_scale", ("U8",), 8, 4),
})


@dataclass(frozen=True, slots=True)
class CsfCheckpoint(IndexedCheckpoint):
    compressed_scales: frozenset[str]

    @property
    def metadata_path(self) -> Path:
        return self.snapshot / "metadata"

    @property
    def index_path(self) -> None:
        # The source index describes logical scales; stored headers describe their encoded planes.
        return None


def inspect_csf(snapshot: Path, description: dict) -> CsfCheckpoint:
    container = FORMATS[description["schema"]]
    label = f"{container.quantization.upper()}-CSF"
    if description.get("codec") != container.codec:
        raise CheckpointError(f"unsupported {label} scale codec")
    metadata = snapshot / "metadata"
    hashes = description.get("metadata_sha256")
    if not isinstance(hashes, dict):
        raise CheckpointError(f"{label} metadata fingerprints are required")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        raw = (metadata / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != hashes.get(name):
            raise CheckpointError(f"{label} metadata differs from its manifest: {name}")
    index = _parse_json((metadata / "model.safetensors.index.json").read_bytes(), "source index")
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise CheckpointError(f"{label} requires a nonempty source weight map")
    files = set()
    for name, file in mapping.items():
        if not isinstance(name, str) or not name or not isinstance(file, str):
            raise CheckpointError(f"invalid {label} source index entry")
        path = PurePosixPath(file)
        if path.is_absolute() or ".." in path.parts or "\\" in file or not file.endswith(".safetensors"):
            raise CheckpointError(f"{label} shard path must stay inside the tensor directory")
        files.add(file)
    inventory = description.get("shards")
    if (not isinstance(inventory, list) or len(inventory) != len(files)
            or any(not isinstance(item, dict) or not isinstance(item.get("file"), str) for item in inventory)):
        raise CheckpointError(f"{label} shard inventory differs from its source index")
    declared = {item["file"]: item for item in inventory}
    if set(declared) != files:
        raise CheckpointError(f"{label} shard names differ from its source index")
    tensors, shards, logical, compressed = {}, [], set(), set()
    plane = f".{container.quantization}_csf_"
    fixed_suffix, exceptions_suffix = plane + "fixed", plane + "exceptions"
    for file in sorted(files):
        relative = "tensors/" + file
        record, entries = _read_header(snapshot / relative, relative)
        if record.file_size != declared[file].get("target_file_bytes"):
            raise CheckpointError(f"{label} shard size differs from its manifest: {file}")
        for name, tensor in entries.items():
            if name in tensors:
                raise CheckpointError(f"duplicate {label} tensor: {name}")
            encoded = name.endswith((fixed_suffix, exceptions_suffix))
            original = name.rsplit(plane, 1)[0] if encoded else name
            if mapping.get(original) != file:
                raise CheckpointError(f"{label} tensor is outside its source inventory: {name}")
            logical.add(original)
        tensors.update(entries)
        shards.append(record)
    # A scale's planes and its packed weight may lie in different shards.
    for name, tensor in tensors.items():
        if not name.endswith(fixed_suffix):
            continue
        original = name.removesuffix(fixed_suffix)
        other = tensors.get(original + exceptions_suffix)
        weight = tensors.get(original.removesuffix(container.scale_suffix) + ".weight")
        if (not original.endswith(container.scale_suffix) or original in tensors or other is None
                or weight is None or weight.dtype not in container.weight_dtypes or len(weight.shape) != 2):
            raise CheckpointError(f"{label} scale needs a unique packed weight and encoded pair: {original}")
        rows, packed_columns = weight.shape
        if rows <= 0 or rows % 32 or packed_columns <= 0 or packed_columns % 16:
            raise CheckpointError(f"invalid {label} matrix geometry: {original}")
        columns = packed_columns // container.packed_columns_per_scale
        expected = rows // 16 * container.slab_bytes(columns)
        if tensor.dtype != "U8" or tensor.shape != (expected,) or other.dtype != "U32" or len(other.shape) != 1:
            raise CheckpointError(f"invalid {label} scale plane geometry: {original}")
        compressed.add(original)
    if logical != mapping.keys() or len(compressed) != description.get("compressed_scale_count"):
        raise CheckpointError(f"{label} tensor coverage differs from its declared source inventory")
    if any(name.endswith(exceptions_suffix) and name.removesuffix(exceptions_suffix) not in compressed for name in tensors):
        raise CheckpointError(f"{label} exception plane lacks its fixed plane")
    if sum(shard.file_size for shard in shards) != description.get("compressed_weight_file_bytes"):
        raise CheckpointError(f"{label} stored bytes differ from its shard inventory")
    return CsfCheckpoint(snapshot, MappingProxyType(tensors), tuple(shards), frozenset(compressed))
