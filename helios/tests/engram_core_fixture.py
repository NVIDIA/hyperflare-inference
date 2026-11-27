# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Export a compact b12x Engram graph and file-backed table for native replay tests."""

from dataclasses import dataclass
from pathlib import Path
from signing import publisher
from fixture_frontend import server_config
import json
import sys

import torch
from ignition.preparation.module_images import ModuleImages
from ignition.preparation.core_storage import CoreStorage
from ignition.preparation.runtime_core import FileSource, export_model


@dataclass(frozen=True)
class _FixtureTables:
    geometry: object
    token_map: tuple[int, ...]
    sources: dict
    inputs: dict
    mailboxes: dict
    table_world: int

    def buffers(self):
        return {
            **{name: tensor for layer, inputs in self.inputs.items() for name, tensor in inputs.buffers(layer).items()},
            **{f"engram_{layer}_requests": mailbox.device_view for layer, mailbox in self.mailboxes.items()},
            **next(iter(self.inputs.values())).timings.buffers(),
        }

    def events(self):
        return {f"engram_{layer}_ready": inputs.ready.cuda_event for layer, inputs in self.inputs.items()}

    def export_metadata(self):
        tables = []
        for ordinal, layer in enumerate(self.geometry.layer_ids):
            head_offsets = self.geometry.offsets[ordinal]
            total_rows = self.geometry.num_embeddings[ordinal]
            head_spans = [
                (head_offsets[head], head_offsets[head + 1] if head + 1 < 24 else total_rows)
                for head in range(24)
            ]
            table = dict(
                layer=layer,
                rows=total_rows,
                embedding_width=256,
                scale_block=32,
                encoding="FP8_E4M3_E8M0",
                weight_buffer=f"engram_{layer}_weights",
                scale_buffer=f"engram_{layer}_scales",
                hash_buffer=f"engram_{layer}_hashes",
                ready_event=f"engram_{layer}_ready",
                request_buffer=f"engram_{layer}_requests",
            )
            offsets = [0] * 24
            first = 0
            shards = []
            for rank in range(self.table_world):
                rank_heads = tuple(range(rank, 24, self.table_world))
                source_rows = [head_spans[head] for head in rank_heads]
                local = 0
                for head, (start, end) in zip(rank_heads, source_rows, strict=True):
                    offsets[head] = first + local
                    local += end - start
                planes = dict(rank=rank, first_row=first, rows=local)
                for name, source, width in zip(("weights", "scales"), self.sources[layer], (256, 8), strict=True):
                    planes[name] = [
                        dict(path=str(source.path), offset=source.offset + start * width, nbytes=(end - start) * width)
                        for start, end in source_rows
                    ]
                if self.table_world == 1:
                    table.update(weights=planes["weights"], scales=planes["scales"])
                else:
                    shards.append(planes)
                first += local
            table["hash"] = dict(
                method="XOR_MODULO",
                orders=[order for order in (2, 3, 4) for _ in range(8)],
                multipliers=list(self.geometry.multipliers[ordinal]),
                primes=list(self.geometry.primes[ordinal]),
                offsets=offsets,
            )
            if shards:
                table["shards"] = shards
            tables.append(table)
        return dict(
            token_map=list(self.token_map),
            pad_token_id=2,
            compressed_vocabulary=self.geometry.compressed_vocab_size,
            tables=tables,
        )


@torch.no_grad()
def prepare(root, shard_size=64 << 30, table_world=1):
    root.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    with ModuleImages() as images, CoreStorage() as storage:
        from b12x.preparation import PreparationSession
        from b12x.sequence import engram
        from ignition.components.engram import EngramInputs, PreparedEngramLookup
        from ignition.components.ngram_delivery import publish_requests
        from ignition.components.readiness import ReadinessTimings
        from ignition.preparation.execution import PreparedExecution, GraphSignature

        geometry = engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
        rows = geometry.num_embeddings[0]
        weight = (torch.arange(rows * 256).remainder(15).float() - 7).to(torch.float8_e4m3fn)
        scales = torch.full((rows * 8,), 127, dtype=torch.uint8)
        sources = []
        for name, tensor in [('weights', weight), ('scales', scales)]:
            path = root / name
            path.write_bytes(tensor.view(torch.uint8).numpy().tobytes())
            sources.append(FileSource(path, 0, tensor.nbytes))
        device = torch.device('cuda:0')
        # Preparation can initialize library storage that the serving graph does not reference.
        warmup = torch.ones((32, 32), device=device)
        torch.mm(warmup, warmup)
        torch.cuda.synchronize()
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        inputs = EngramInputs.allocate(4, device, timings=ReadinessTimings.allocate(frame, 1))
        output = torch.empty((4, 6144), dtype=torch.bfloat16, device=device)
        count = torch.full((1,), 4, dtype=torch.int32, device=device)
        source = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        transfer_source = torch.zeros(1 << 20, dtype=torch.uint8, device=device)
        transfer_target = torch.empty_like(transfer_source)
        mailbox = storage.host_allocate("engram_1_requests", (4 * 24 + 1,), torch.int64)
        mailbox.host_view.zero_()
        records = torch.zeros((2, 4, 32), dtype=torch.int64, device=device)
        record = torch.zeros((4, 32), dtype=torch.int64, device=device)
        with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
            lookup = PreparedEngramLookup(inputs, output, count, layer=1, token_map=tuple(range(32)),
                geometry=geometry, session=session, name='engram')

            def run():
                transfer_target.copy_(transfer_source)
                source.add_(1)
                publish_requests(inputs.hashes, mailbox.device_view, sequence)
                lookup(source)
                record.zero_()
                record[0, 0].fill_(1)
                record[0, 1].copy_(sequence[0])
                record[0, 2].copy_(output.float().square().sum().to(torch.int64))
                record[0, 4].fill_(-1)
                record[0, 5].fill_(1)
                record[0, 12].fill_(-1)
                record[0, 15].fill_(1)
                record[0, 16:].fill_(-1)
                records.index_copy_(0, frame.long(), record.unsqueeze(0))

            run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with session.capture(), torch.cuda.graph(graph, pool=storage.graph_pool.id):
                run()
            graph.instantiate()
            # Session cleanup unloads compiled kernels; retain it until export has read the graph.
            try:
                tables = _FixtureTables(geometry, tuple(range(32)), {1: tuple(sources)}, {1: inputs}, {1: mailbox},
                    table_world)
                reference_rows = list(range(rows))
                (root / 'reference_rows.json').write_text(json.dumps(reference_rows))
                buffers = dict(tables.buffers(), output=output, transfer_source=transfer_source, frame_index=frame,
                               submission_sequence=sequence, completion_records=records)
                prepared = PreparedExecution(device=0, graphs={'decode': graph}, buffers=buffers,
                    signatures={'decode': GraphSignature('decode', 4, 1, ('engram_1_hashes', 'transfer_source'))},
                    max_active=4, resources=(storage, lookup), export_events=tables.events())
                prepared.server_config = server_config(root)
                prepared.core_data = tables
                if table_world == 1:
                    export_model(prepared, images, storage, root / 'core', options={}, publisher=publisher(),
                                 )
                else:
                    from ignition.preparation import runtime_core_pb2 as proto
                    from ignition.preparation.core_file import read_index
                    from signing import publish_legacy as publish
                    from ignition.preparation.protection import resource_roles

                    staged = root / 'table-source'
                    export_model(prepared, images, storage, staged, options={})
                    index, start = read_index(staged)
                    # The replay test binds one graph to each partition separately. Empty
                    # peer bundles provide storage owners without allocating more GPUs.
                    index.ranks.extend(proto.BundleIndex() for _ in range(table_world - 1))
                    payload = sorted((span.offset, staged, start + span.offset, span.nbytes)
                                     for _, _, span in resource_roles(index))
                    publish(root / 'core', index, payload, publisher=publisher(), shard_size=shard_size)
                    staged.unlink()
            finally:
                graph.reset()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--shard-size", type=int, default=64 << 30)
    parser.add_argument("--table-world", type=int, choices=(1, 4), default=1)
    args = parser.parse_args()
    prepare(args.directory, args.shard_size, args.table_world)
