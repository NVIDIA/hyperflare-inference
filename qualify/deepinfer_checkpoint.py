# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Check real CSF tensor coverage with DeepInfer's unchanged parameter mapping.

Run with the isolated DeepInfer preparation environment. This CPU-only check
constructs the complete model on meta, including vision and draft layers. It
does not validate GPU arithmetic or qualify serving.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from ignition.recipes.deepseek41.config import ModelConfig
from ignition.recipes.deepseek41.reference_checkpoint import ReferenceCheckpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from deepinfer.hugging_face.weights import build_coverage, _TensorReader
    from deepinfer.hugging_face.names import hf_layer_module_prefix
    from deepinfer.hugging_face import cover_tied_parameters, model_params, validate_hf_shapes
    from deepinfer.model_executor.models.v4_modules.builder import V41Builder
    from deepinfer.model_executor.parallel_utils.parallel_state import PG

    checkpoint = ReferenceCheckpoint(args.checkpoint)
    raw = json.loads((checkpoint.source.metadata_path / "config.json").read_text())
    model = ModelConfig.from_dict(raw)
    reference_config = checkpoint.model_config()
    logical_map = {name: checkpoint.source.tensors[
        name + ".mxfp4_csf_fixed" if name in checkpoint.compressed_scales else name].shard
        for name in checkpoint.tensors}
    coverage = build_coverage(logical_map, gpt_cfg=reference_config.gpt)
    PG.initialize_local()
    with torch.device("meta"):
        reference_model = V41Builder(reference_config).build_model(vision_device="meta")
    params = model_params(reference_model)
    if not all(param.is_meta for param in params.values()):
        raise AssertionError("reference parameter construction allocated physical weights")
    coverage = cover_tied_parameters(coverage, set(params), n_layer=model.layers)
    if missing := set(params) - coverage.model_names():
        raise AssertionError(f"checkpoint does not cover reference parameters: {sorted(missing)}")
    validate_hf_shapes(params, coverage, hf_dir=checkpoint.source.snapshot)
    expert_operands = 0
    for target, experts in coverage.expert_tensors.items():
        param = params[target]
        count = param.n_experts
        input_width, remainder = divmod(param.shape[0], count)
        if remainder or len(experts) != count or input_width % 32:
            raise AssertionError(f"reference expert layout differs: {target}")
        for names in experts:
            rows = 0
            for name in names:
                weight = checkpoint.tensors[name]
                scale = checkpoint.tensors[name.removesuffix(".weight") + ".scale"]
                if (len(weight.shape) != 2 or weight.dtype not in ("I8", "U8")
                        or weight.shape[1] != input_width // 2
                        or scale.dtype != "F8_E8M0"
                        or scale.shape != (weight.shape[0], input_width // 32)):
                    raise AssertionError(f"reference expert operand differs: {name}")
                rows += weight.shape[0]
                expert_operands += 1
            if rows != param.shape[1]:
                raise AssertionError(f"reference expert concatenation differs: {target}")
    if "deep_ep" in sys.modules or torch.distributed.is_initialized():
        raise AssertionError("single-device construction loaded a distributed runtime")

    # Flare's native delivery receives canonical table rows. A different
    # compressed vocabulary or hash layout would silently fetch other weights.
    from transformers import PreTrainedTokenizerFast
    from b12x.sequence.engram import build_compressed_token_map, build_geometry
    from deepinfer.model_executor.models.v4_modules.engram_metadata import CompressedVocabulary, NgramHashMapping

    tokenizer_path = checkpoint.source.metadata_path / "tokenizer.json"
    lookup, vocabulary = build_compressed_token_map(PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path)))
    reference_vocabulary = CompressedVocabulary(str(tokenizer_path))
    if lookup != reference_vocabulary.old2new or vocabulary != len(reference_vocabulary):
        raise AssertionError("Flare and reference Engram token maps differ")
    geometry = build_geometry(layer_ids=tuple(model.text["engram_layer_ids"]),
        base_table_size=model.text["engram_vocab_size"], compressed_vocab_size=vocabulary)
    reference_hash = NgramHashMapping(reference_config.gpt.engram, device="cpu")
    for actual, expected, label in (
        (reference_hash.ngram_vocab_sizes.flatten(1), geometry.primes, "primes"),
        (reference_hash.ngram_offsets, geometry.offsets, "offsets"),
        (reference_hash.ngram_multipliers, geometry.multipliers, "multipliers"),
    ):
        if actual.tolist() != [list(row) for row in expected]:
            raise AssertionError(f"Flare and reference Engram {label} differ")
    expected_layers = [(f"layers.{layer}", model.text["n_routed_experts"]) for layer in range(model.layers)]
    expected_layers += [(f"mtp.{layer}", model.text["dspark_n_routed_experts"]) for layer in range(model.draft.layers)]
    checks = []
    physical_map = {name: metadata.shard for name, metadata in checkpoint.source.tensors.items()}
    with _TensorReader(checkpoint.source.snapshot, physical_map) as native:
        for prefix, count in expected_layers:
            module = hf_layer_module_prefix(prefix, n_layer=model.layers)
            for which, projections in (("linear1", ("w1", "w3")), ("linear2", ("w2",))):
                actual = coverage.expert_tensors[module + f"experts.{which}.weight"]
                expected = tuple(tuple(f"{prefix}.ffn.experts.{expert}.{projection}.weight"
                                       for projection in projections) for expert in range(count))
                if actual != expected:
                    raise AssertionError(f"canonical expert coverage differs: {prefix}.{which}")
            # Inspect the first and last expert's projection operands in every
            # target/draft layer. Loading remains bounded to one operand pair.
            for expert in (0, count - 1):
                for projection in ("w1", "w3", "w2"):
                    name = f"{prefix}.ffn.experts.{expert}.{projection}"
                    packed = checkpoint.read(name + ".weight")
                    reference = native.read(name + ".weight", torch.device("cpu"))
                    if not torch.equal(packed, reference):
                        raise AssertionError(f"packed checkpoint bytes differ: {name}")
                    scale = checkpoint.read(name + ".scale")
                    if scale.shape != (packed.shape[0], packed.shape[1] // 16):
                        raise AssertionError(f"decoded scale geometry differs: {name}")
                    checks.append(dict(projection=name, packed_bytes=packed.nbytes,
                                       decoded_scale_bytes=scale.nbytes))
                    del packed, reference, scale
    if torch.cuda.is_initialized():
        raise AssertionError("checkpoint preflight initialized CUDA")
    result = dict(scope=__doc__, checkpoint=str(checkpoint.source.snapshot),
        architecture=reference_config.architecture,
        physical_tensors=len(checkpoint.source.tensors), logical_tensors=len(checkpoint.tensors),
        compressed_scale_tensors=len(checkpoint.compressed_scales),
        expert_parameters=len(coverage.expert_tensors), canonical_experts=sum(n for _, n in expected_layers),
        validated_expert_operand_headers=expert_operands,
        ordinary_parameters=len(coverage.tensors), engram_tables=len(coverage.engram_tables),
        model_parameters=len(params),
        vision_parameters=sum(name.startswith("vision_pipeline.") for name in params),
        engram_token_map_entries=len(lookup), engram_compressed_vocabulary=vocabulary,
        engram_hash_geometry_equal=True,
        distributed_initialized=False, deep_ep_imported=False,
        sampled_operands=checks, cuda_initialized=False, completed=True, serving_qualified=False)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("sampled_operands", "scope")}, indent=2))


if __name__ == "__main__":
    main()
