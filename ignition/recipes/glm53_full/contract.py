# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Graph inputs, continuation storage, page regions, and frontend syntax that Helios binds."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p

from .attention import PAGE_TOKENS

# The checkpoint template writes no media token: it replaces a media block with a text notice. Request
# text that spells `<|image|>` or `<|video|>` tokenizes to that token, as the checkpoint tokenizer does,
# and the text model embeds it like any other token, so the contract rejects no input token.
FRONTEND = c.frontend(tools=p.GLM_ARGUMENT_TAGS, assistant_marker="<|assistant|>",
    message_end_markers=["<|user|>", "<|system|>", "<|observation|>"],
    prefix_boundary=c.prefix_boundary(["system"], append_role="user", minimum_messages=0,
        required_start="[gMASK]<sop><|system|>", required_end="<|user|>"))


def prepare(signatures, buffers, pool, *, vocabulary, draft_lag=None):
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()}}
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    consumed = pool.layout.region("consumed_tokens")
    cursor = None if draft_lag is None else dict(state=c.state_view("state_pool", pool.layout.row_bytes,
        offset=consumed.offset, nbytes=consumed.nbytes), lag_tokens=draft_lag)
    pending = c.state_view("request_pending_token", 8)
    regions = [c.history_region(buffers, f"cache.{name}", tensor, p.PAGED_LATENT)
               for name, tensor in pool.pages.items()]
    return c.model_contract(vocabulary=vocabulary, page_tokens=PAGE_TOKENS,
        page_tables=[dict(buffer="page_tables", page_offset=0)],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0), *regions],
        graphs=c.graph_contracts(signatures, inputs, continuation=continuation, pending=pending, cursor=cursor),
        vision=None, frontend=FRONTEND)
