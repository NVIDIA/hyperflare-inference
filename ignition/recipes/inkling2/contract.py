# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Paged attention, complete continuation rows and Inkling 2 frontend bindings."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p
from .frontend import frontend
from .pool import PAGE_TOKENS


def prepare(signatures, buffers, pool, *, vocabulary, image_slots=4, vision=True, drafting=True):
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()} if kind != 'vision'}
    inputs['vision'] = c.vision_inputs({field for s in signatures.values() if s.kind == 'vision' for field in s.inputs})
    continuation = c.state_view('state_pool', pool.layout.row_bytes)
    consumed = pool.layout.region('consumed_tokens')
    cursor = (dict(state=c.state_view('state_pool', pool.layout.row_bytes, offset=consumed.offset,
        nbytes=consumed.nbytes), lag_tokens=0) if drafting else None)
    regions = [dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0)]
    regions.extend(c.history_region(buffers, f'cache.{name}', tensor, p.PAGED_KEY_VALUE)
                   for name, tensor in pool.pages.items())
    return c.model_contract(vocabulary=vocabulary, page_tokens=PAGE_TOKENS,
        page_tables=[dict(buffer='page_tables', page_offset=0)], regions=regions,
        graphs=c.graph_contracts(signatures, inputs, continuation=continuation,
            pending=c.state_view('request_pending_token', 8), cursor=cursor),
        vision=(dict(feature_slots=image_slots, features_per_request=16384, max_images=4,
                     patches_per_feature=4, patch_capacities=[64, 256, 1024, 4096, 16384, 65536])
                if vision else None), frontend=frontend(vision=vision))
