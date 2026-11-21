# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Head-sharded GLM latent attention over the records a DSA indexer selects.

Every rank holds identical low-rank projections and attends with its own
heads. `ignition.reference.glm_dsa` states the arithmetic: content queries are
absorbed into latent space, rotary pairs stay interleaved, and attended latents
expand into per-head values. Without context parallelism a rank also holds
every latent record and index key. With it, `dcp` places both: decode merges
the ranks' selections, exchanges query heads, and combines partial results, and
prefill gathers index keys and the selected records into slabs.
"""

from typing import NamedTuple

import torch
from b12x.attention import sparse_mla
from b12x.attention.dsa_indexer.fused_indexer import fused_indexer_scratch_max_rows
from flashinfer.triton.norm import rms_norm

from ignition.components.collectives import PreparedAllGather, PreparedLseReduceScatter
from ignition.components.fp8_indexer import PreparedFP8Indexer, write_fp8_index_keys
from ignition.components.linear import BatchedUnquantizedProjection, prepare_projection
from ignition.components.sparse_mla import PreparedSparseMLA
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.tuning import register_finalizer

from .config import (INDEX_DIM, INDEX_HEADS, INDEX_RECORD_BYTES, INDEX_TOPK, KERNEL_PAGE_RECORDS, LATENT_RANK,
                     LATENT_RECORD_BYTES, PAGE_RECORDS, ROTARY_DIM)
from .dcp import ContextShards, global_records, pull_keys, pull_marked
from .records import rotary_angles, rotate, write_records

# Prefill selection scratch grows with query rows, so long chunks select in slices.
SELECT_ROWS = 512
QUERY_WIDTH = LATENT_RANK + ROTARY_DIM


class Selection(NamedTuple):
    """Records each row attends, compacted to the front, and each row's record count.

    Records are local physical records, or global positions when context-parallel prefill attends its slab.
    """

    records: torch.Tensor
    counts: torch.Tensor


class PreparedIndexer:
    """One layer's index keys and per-row selection of the 2048 highest-scoring earlier positions.

    Without `selects`, the indexer only writes keys: its layer attends a selection made elsewhere or
    only records its positions.

    With `context`, a rank stores the keys of the positions it owns. Decode selects among them, and
    the context ranks merge their selections. Prefill selects from the caller's key slab, which holds
    one key per global position: this layer's keys of the chunk, and every earlier position's, copied
    from the ranks that own them through `peers`.

    `first` is the first cache slot a row may attend.
    """

    def __init__(self, module, weights, source, query_low, weight_input, metadata, cache, *, shards, mode,
                 session, name, workspace, examples, selects=True, context=None, peers=None, first=0):
        rows, device = source.shape[0], source.device
        if (module.n_heads, module.head_dim, module.index_topk) != (INDEX_HEADS, INDEX_DIM, INDEX_TOPK):
            raise ValueError("GLM MoE DSA indexing requires 32 heads of 128 channels and 2048 selected positions")
        self.cache, self.mode, self.eps = cache, mode, module.k_norm.eps
        self.shards, self.peers, self.first = shards, peers, first
        self.norm = weights["indexer.k_norm.weight"], weights["indexer.k_norm.bias"]
        self.query, self.selectors, self.bounds, self.gather, self.scores = None, {}, (), None, {}
        # Context-parallel prefill selects from a slab, and every other selection reads this rank's pool.
        self.slab_pages = context.slab_positions // KERNEL_PAGE_RECORDS if context is not None and mode == "prefill" else 0
        self.merges = context is not None and mode == "decode"
        if not selects:
            return

        def tensor(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "glm_dsa.index." + role, shape, dtype, device)

        query = tensor("query", (rows, INDEX_HEADS * INDEX_DIM))
        self.query = prepare_projection(query_low, weights["indexer.wq_b"], query, session=session,
                                        name=name + ".query", workspace=workspace)
        if self.slab_pages:
            cache = tensor("key_slab", (self.slab_pages, KERNEL_PAGE_RECORDS * INDEX_RECORD_BYTES), torch.uint8)
            width = self.slab_pages
        else:
            width = metadata.page_table.shape[1] * (PAGE_RECORDS // KERNEL_PAGE_RECORDS)
        # b12x selects for every row of a decode call in one launch through this many rows, and
        # launches twice for each row of a longer call, so longer graphs select in slices.
        step = SELECT_ROWS if mode == "prefill" else fused_indexer_scratch_max_rows(topk=INDEX_TOPK,
                                                                                    num_heads=INDEX_HEADS)
        self.bounds = tuple((start, min(start + step, rows)) for start in range(0, rows, step))
        for start, end in self.bounds:
            count = end - start
            if count in self.selectors:
                continue
            # Prefill rows of one request share its page table, and decode rows carry one each.
            pages = tensor("pages", (1 if mode == "prefill" else count, width), torch.int32)
            if self.merges:
                self.scores[count] = tensor("scores", (count, INDEX_TOPK), torch.float32)
            self.selectors[count] = PreparedFP8Indexer(
                query[:count].view(count, INDEX_HEADS, INDEX_DIM), weight_input[:count], cache,
                pages.expand(count, -1) if mode == "prefill" else pages, tensor("lengths", (count,), torch.int32),
                None, tensor("selected", (count, INDEX_TOPK), torch.int32),
                mode="prefill" if mode == "prefill" else "decode", session=session, name=f"{name}.select{count}",
                workspace=workspace, topk=INDEX_TOPK, prefill_records_per_call=131072, scores=self.scores.get(count),
                hadamard=False, key_power_of_two=False, examples=examples, example_role="glm_dsa.index.fp8")
        if self.merges:
            # A candidate is eight bytes, gathered as four BF16 columns.
            packed = tensor("candidates", (rows, INDEX_TOPK * 4))
            self.gather = PreparedAllGather(packed, tensor("gathered", (rows, INDEX_TOPK * 4 * shards.world)),
                runtime=context.candidates, session=session, coordinator=context.coordinator,
                name=name + ".candidates").functional(session=session)
        self.query.functional(session=session)
        for selector in self.selectors.values():
            # Every call supplies its own slab. The declaration example must not outlive tuning.
            selector.functional(session=session, dynamic_cache=bool(self.slab_pages))
        register_finalizer(session, self.scores.clear)

    def _key(self, key_input, cos, sin):
        weight, bias = self.norm
        key = torch.nn.functional.layer_norm(key_input.float(), (INDEX_DIM,), weight.float(), bias.float(), self.eps)
        key = key.to(key_input.dtype)
        return torch.cat((rotate(key[:, :ROTARY_DIM], cos, sin), key[:, ROTARY_DIM:]), -1)

    def _store(self, key, metadata):
        write_fp8_index_keys(key, self.shards.record_slots(metadata.positions, metadata.token_page_table),
                             self.cache, hadamard=False, power_of_two=False)

    def write_keys(self, key_input, metadata, cos, sin):
        """Write the index keys of the rows' positions that this rank owns."""
        self._store(self._key(key_input, cos, sin), metadata)

    def __call__(self, query_low, key_input, weight_input, metadata, cos, sin, *, slabs=None):
        """Write the rows' index keys and return the selection each row attends."""
        m, rows, shards = metadata, query_low.shape[0], self.shards
        if bool(self.slab_pages) != (slabs is not None):
            raise ValueError("context-parallel prefill selects from the caller's key slab")
        query = self.query(query_low).view(rows, INDEX_HEADS, INDEX_DIM)
        query = torch.cat((rotate(query[..., :ROTARY_DIM], cos, sin), query[..., ROTARY_DIM:]), -1)
        key = self._key(key_input, cos, sin)
        self._store(key, m)
        lengths = (m.positions + 1).clamp_min(0)
        if slabs is None:
            cache = None
            lengths = shards.local_tokens(lengths)
            pages = shards.index_pages(m.page_table if self.mode == "prefill" else m.token_page_table)
        else:
            # Every rank derives the same keys for the chunk's positions, so none of them cross ranks.
            # Positions before the chunk were written by earlier graphs on the ranks that own them.
            write_fp8_index_keys(key, m.positions.long(), slabs.keys, hadamard=False, power_of_two=False)
            pull_keys(self.peers, m.page_table[0], m.positions[:1].long(), slabs.keys, shards)
            cache = slabs.keys
            # The selector copies every page it is given before it scores, so the pages after the
            # chunk's last position stay out of the table.
            pages = torch.arange(self.slab_pages, dtype=torch.int32, device=query.device)
            pages = torch.where(pages <= m.positions.max() // KERNEL_PAGE_RECORDS, pages, -1)[None]
        lengths = lengths.to(torch.int32)
        selected, scores = [], []
        for start, end in self.bounds:
            count = end - start
            result = self.selectors[count](query[start:end], weight_input[start:end],
                pages.expand(count, -1) if self.mode == "prefill" else pages[start:end], lengths[start:end], None,
                cache=cache)
            if self.merges:
                # A selector run on its declaration operands leaves the scores in them.
                result, score = result if isinstance(result, tuple) else (result, self.scores[count])
                scores.append(score)
            selected.append(result)
        selected = selected[0] if len(selected) == 1 else torch.cat(selected)
        if slabs is not None:
            records, counts = global_records(selected, m.positions, first=self.first, marks=slabs.marks)
            return Selection(records, counts)
        if not self.merges:
            return Selection(*shards.local_records(selected, m.positions, m.token_page_table, first=self.first))
        packed = shards.pack_candidates(selected, scores[0] if len(scores) == 1 else torch.cat(scores))
        gathered = self.gather(packed.view(torch.bfloat16)).view(torch.int64)
        return Selection(*shards.merged_records(gathered.view(rows, shards.world, INDEX_TOPK), m.positions,
                                                m.token_page_table, first=self.first))


class PreparedAttention:
    """One rank's attention before the row-parallel output reduction.

    The fused input projection yields the low-rank query, the latent and rotary
    key of each position's cache record, and, for a layer that owns an indexer,
    its index key and head weights. A layer without an indexer attends the
    selection passed to it.

    With `context`, the rank stores only the latent records and index keys it
    owns, and `layer` names the layer's page pools in the context: a decoder
    layer index, or "draft". Decode gathers every context rank's absorbed
    heads, attends its own records for all of them, and combines the ranks'
    partial results by their log-sum-exp. Prefill attends the caller's record
    slab, which holds one record per global position: this layer's records of
    the chunk, and the earlier positions the rows selected, copied from the
    ranks that own them. A layer that selects also takes the caller's key slab
    and reads the ranks' key pools.

    `selects=False` prepares a layer that owns an indexer without its
    selection: the layer writes its index keys and attends the selection
    passed to it. `cache_only` prepares only what writes a position's cache
    record and index key, and `write` runs it.

    The draft layer stores the record of sequence position q in cache slot
    q + 1, beside the target's records of the token that record depends on, so
    it passes `rotary_offset=-1` and `first_record=1`: metadata positions name
    cache slots, rotation uses the slot less one, and slot zero, which holds no
    sequence position, is never attended.
    """

    def __init__(self, module, weights, source, metadata, cache, *, index_cache, frequencies, mode, session,
                 name, workspace, examples, context=None, layer=None, selects=None, cache_only=False,
                 rotary_offset=0, first_record=0):
        rows, hidden = source.shape
        device, heads = source.device, weights["q_b_proj"].shape[0] // (module.qk_nope_head_dim + ROTARY_DIM)
        content, value, rank = module.qk_nope_head_dim, module.v_head_dim, module.q_lora_rank
        if (module.kv_lora_rank, module.qk_rope_head_dim) != (LATENT_RANK, ROTARY_DIM):
            raise ValueError("canonical MLA geometry differs from the GLM MoE DSA latent record")
        if cache.shape[1:] != (KERNEL_PAGE_RECORDS, LATENT_RECORD_BYTES) or cache.dtype != torch.uint8:
            raise ValueError("GLM MoE DSA latent cache requires 64-record pages of 656-byte records")
        if mode not in ("decode", "prefill"):
            raise ValueError("GLM MoE DSA attention prepares decode and prefill")
        indexed = module.indexer is not None
        if indexed != (index_cache is not None) or indexed != ("indexer.wq_b" in weights):
            raise ValueError("an indexer layer requires its index weights and key cache")
        shards = ContextShards() if context is None else context.shards
        peers = None if context is None else context.peers[layer]
        index_peers = context.index_peers[layer] if context is not None and indexed else None
        self.cache, self.mode, self.frequencies = cache, mode, frequencies
        self.heads, self.content, self.value, self.rank = heads, content, value, rank
        self.context, self.shards, self.peers = context, shards, peers
        self.norms = weights["q_a_layernorm.weight"], weights["kv_a_layernorm.weight"]
        self.eps = module.q_a_layernorm.variance_epsilon, module.kv_a_layernorm.variance_epsilon
        self.selects = indexed if selects is None else selects
        if self.selects and not indexed:
            raise ValueError("only a layer that owns an indexer selects positions")
        self.rotary_offset, self.first_record = rotary_offset, first_record

        def tensor(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "glm_dsa.attention." + role, shape, dtype, device)

        width = weights["input_proj"].shape[0]
        projected = tensor("projected" + ("_indexed" if indexed else ""), (rows, width))
        query_low = tensor("query_low", (rows, rank))
        base = rank + QUERY_WIDTH
        self.indexer = self.gather = self.combine = self.attention = None

        def indexer(selects):
            return PreparedIndexer(module.indexer, weights, source, query_low, projected[:, base + INDEX_DIM:],
                metadata, index_cache, shards=shards, mode=mode, session=session, name=name + ".indexer",
                workspace=workspace, examples=examples, selects=selects, context=context, peers=index_peers,
                first=first_record)

        if cache_only:
            self.projections = {"input_proj": prepare_projection(source, weights["input_proj"], projected,
                session=session, name=name + ".input_proj", workspace=workspace)}
            if indexed:
                self.indexer = indexer(False)
            self.projections["input_proj"].functional(session=session)
            return
        query = tensor("query", (rows, heads * (content + ROTARY_DIM)))
        absorbed = tensor("absorbed", (rows, heads, QUERY_WIDTH))
        attended = tensor("attended", (rows, heads, LATENT_RANK))
        expanded = tensor("expanded", (rows, heads * value))
        self.projections = {key: prepare_projection(x, weights[key], y, session=session, name=f"{name}.{key}",
                                                    workspace=workspace)
            for key, x, y in (("input_proj", source, projected), ("q_b_proj", query_low, query),
                              ("o_proj", expanded, tensor("output", (rows, hidden))))}
        # Per head, the key expansion absorbs content queries and the value expansion follows attention.
        # Both multiply [heads, rows, width] stacks: cuBLAS reads far past a stack whose head matrices
        # interleave, as they do in the [rows, heads, width] buffers the surrounding operators use.
        latent = weights["kv_b_proj"].view(heads, content + value, LATENT_RANK)
        self.absorb = BatchedUnquantizedProjection(tensor("content_by_head", (heads, rows, content)),
            latent[:, :content], tensor("absorbed_by_head", (heads, rows, LATENT_RANK)), major="n",
            session=session, name=name + ".absorb")
        self.expand = BatchedUnquantizedProjection(tensor("attended_by_head", (heads, rows, LATENT_RANK)),
            latent[:, content:], tensor("expanded_by_head", (heads, rows, value)), major="k", session=session,
            name=name + ".expand")
        if indexed:
            self.indexer = indexer(self.selects)
        selected = tensor("selected", (rows, INDEX_TOPK), torch.int32)
        visible, counts = tensor("visible", (rows,), torch.int32), tensor("counts", (rows,), torch.int32)
        geometry = dict(device=device, max_q_rows=rows, max_width=INDEX_TOPK, softmax_scale=module.scaling,
            max_batch=rows, kv_dtype=torch.uint8, head_dim=QUERY_WIDTH, v_head_dim=LATENT_RANK,
            page_size=KERNEL_PAGE_RECORDS, cache_record_bytes=LATENT_RECORD_BYTES)
        if context is None:
            caps = sparse_mla.Caps(num_q_heads=heads, max_kv_rows=cache.shape[0] * KERNEL_PAGE_RECORDS,
                                   mode="decode" if mode == "decode" else "extend", **geometry)
            self.attention = PreparedSparseMLA(absorbed, cache, selected, visible, counts, attended, caps=caps,
                session=session, name=name + ".attention", workspace=workspace)
        elif mode == "decode":
            total = heads * shards.world
            gathered = tensor("gathered", (rows, total, QUERY_WIDTH))
            partial = tensor("partial", (rows, total, LATENT_RANK))
            lse = tensor("lse", (rows, total), torch.float32)
            self.gather = PreparedAllGather(absorbed.view(rows, -1), gathered.view(rows, -1), runtime=context.heads,
                session=session, coordinator=context.coordinator, name=name + ".gather_heads")
            caps = sparse_mla.Caps(num_q_heads=total, max_kv_rows=cache.shape[0] * KERNEL_PAGE_RECORDS,
                                   mode="decode", return_lse=True, lse_scale="natural", **geometry)
            self.attention = PreparedSparseMLA(gathered, cache, selected, visible, counts, partial, caps=caps,
                session=session, name=name + ".attention", workspace=workspace, lse=lse)
            self.combine = PreparedLseReduceScatter(partial, lse, attended, runtime=context.heads, session=session,
                coordinator=context.coordinator, name=name + ".combine")
        else:
            slab = tensor("slab", (context.slab_positions, LATENT_RECORD_BYTES), torch.uint8)
            caps = sparse_mla.Caps(num_q_heads=heads, max_kv_rows=context.slab_positions, mode="extend", **geometry)
            self.attention = PreparedSparseMLA(absorbed, slab.view(-1, KERNEL_PAGE_RECORDS, LATENT_RECORD_BYTES),
                selected, visible, counts, attended, caps=caps, session=session, name=name + ".attention",
                workspace=workspace)
        for projection in self.projections.values():
            projection.functional(session=session)
        self.absorb.functional(session=session)
        self.expand.functional(session=session)
        self.attention.functional(session=session)
        if self.gather is not None:
            self.gather.functional(session=session)
            self.combine.functional(session=session)
        if context is not None and mode == "prefill":
            # Every call supplies its own slab. The declaration example must not outlive tuning.
            register_finalizer(session, self._release_slab)

    def _release_slab(self):
        self.attention.inputs["kv_cache"] = None

    @staticmethod
    def _by_head(value):
        """[rows, heads, width] as a [heads, rows, width] stack whose head matrices do not interleave.

        One row's heads already lie in that order, and more rows are gathered head by head.
        """
        return value.transpose(0, 1) if value.shape[0] == 1 else value.transpose(0, 1).contiguous()

    def _record(self, source, metadata):
        """Project the layer input, write each position's latent record, and return what attention reuses."""
        m, rows, rank = metadata, source.shape[0], self.rank
        projected = self.projections["input_proj"](source)
        query_low = torch.empty((rows, rank), dtype=source.dtype, device=source.device)
        latent = torch.empty((rows, LATENT_RANK), dtype=source.dtype, device=source.device)
        rms_norm(projected[:, :rank], self.norms[0], query_low, self.eps[0], round_before_weight=True)
        rms_norm(projected[:, rank:rank + LATENT_RANK], self.norms[1], latent, self.eps[1], round_before_weight=True)
        cos, sin = rotary_angles(m.positions + self.rotary_offset, self.frequencies)
        key_rotary = rotate(projected[:, rank + LATENT_RANK:rank + QUERY_WIDTH], cos, sin)
        write_records(latent, key_rotary, self.shards.record_slots(m.positions, m.token_page_table), self.cache)
        return projected, query_low, latent, key_rotary, cos, sin

    def write(self, source, metadata):
        """Write the rows' cache records and index keys without attending."""
        projected, _, _, _, cos, sin = self._record(source, metadata)
        if self.indexer is not None:
            base = self.rank + QUERY_WIDTH
            self.indexer.write_keys(projected[:, base:base + INDEX_DIM], metadata, cos, sin)

    def __call__(self, source, metadata, selection=None, *, slabs=None):
        m, rows = metadata, source.shape[0]
        if self.attention is None:
            raise ValueError("a cache-only layer records positions without attending")
        if (selection is None) != self.selects:
            raise ValueError("a layer that selects no positions attends the selection passed to it")
        gathers = self.context is not None and self.mode == "prefill"
        if gathers != (slabs is not None):
            raise ValueError("context-parallel prefill attends the caller's slabs")
        shards, rank = self.shards, self.rank
        projected, query_low, latent, key_rotary, cos, sin = self._record(source, m)
        if gathers:
            # Every rank derives the same records for the chunk's positions, so none of them cross ranks.
            write_records(latent, key_rotary, m.positions.long(), slabs.records)
        query = self.projections["q_b_proj"](query_low).view(rows, self.heads, self.content + ROTARY_DIM)
        latent_query = self.absorb(self._by_head(query[:, :, :self.content]))
        absorbed = torch.cat((latent_query.transpose(0, 1), rotate(query[:, :, self.content:], cos, sin)), -1)
        base = rank + QUERY_WIDTH
        if self.selects:
            selection = self.indexer(query_low, projected[:, base:base + INDEX_DIM], projected[:, base + INDEX_DIM:],
                                     m, cos, sin, slabs=slabs)
        elif self.indexer is not None:
            self.indexer.write_keys(projected[:, base:base + INDEX_DIM], m, cos, sin)
        lengths = (m.positions + 1).clamp_min(0)
        if self.context is None:
            attended = self.attention(absorbed, selection.records, lengths.to(torch.int32), selection.counts,
                                      cache=self.cache)
        elif self.mode == "decode":
            total = self.heads * shards.world
            gathered = self.gather(absorbed.view(rows, -1)).view(rows, total, QUERY_WIDTH)
            partial, lse = self.attention(gathered, selection.records, shards.local_tokens(lengths).to(torch.int32),
                                          selection.counts, cache=self.cache)
            attended = self.combine(partial, lse)
        else:
            # Positions before the chunk were written by earlier graphs on the ranks that own them.
            pull_marked(self.peers, m.page_table[0], slabs.marks, m.positions[:1].long(), slabs.records, shards)
            attended = self.attention(absorbed, selection.records, lengths.to(torch.int32), selection.counts,
                cache=slabs.records.view(-1, KERNEL_PAGE_RECORDS, LATENT_RECORD_BYTES))
        expanded = self.expand(self._by_head(attended)).transpose(0, 1).reshape(rows, self.heads * self.value)
        return self.projections["o_proj"](expanded), selection
