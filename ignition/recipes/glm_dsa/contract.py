# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared graph inputs, continuation storage, paged attention bindings, and frontend syntax."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p

# The checkpoint is text-only: its template answers media parts with a reminder and writes no media
# token, so the special media delimiters never reach the model. The `<|image|>` and `<|video|>`
# placeholders are ordinary added tokens that the template does not name: request text that spells
# one tokenizes to it, as it does in Transformers.
FRONTEND = c.frontend(tools=p.GLM_ARGUMENT_TAGS, assistant_marker="<|assistant|>",
    rejected_tokens=range(154830, 154836),
    message_end_markers=["<|user|>", "<|system|>", "<|observation|>"],
    prefix_boundary=c.prefix_boundary(["system"], append_role="user", minimum_messages=0,
        required_start="[gMASK]<sop><|system|>", required_end="<|user|>"))


def prepare(signatures, buffers, pool, *, vocabulary, page_tokens, draft_lag=None):
    """Declare the catalog's inputs and state.

    With `draft_lag`, speculative graphs declare the draft cache's cursor: the cache holds that
    many fewer positions than the request has consumed.
    """
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()}}
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    pending = c.state_view("request_pending_token", 8)
    consumed = pool.layout.region("consumed_tokens")
    cursor = None if draft_lag is None else dict(state=c.state_view("state_pool", pool.layout.row_bytes,
        offset=consumed.offset, nbytes=consumed.nbytes), lag_tokens=draft_lag)
    regions = [c.history_region(buffers, f"cache.{name}", tensor, p.PAGED_LATENT)
               for name, tensor in pool.pages.items()]
    return c.model_contract(vocabulary=vocabulary, page_tokens=page_tokens,
        page_tables=[dict(buffer="page_tables", page_offset=0)],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0), *regions],
        graphs=c.graph_contracts(signatures, inputs, continuation=continuation, pending=pending, cursor=cursor),
        frontend=FRONTEND)
