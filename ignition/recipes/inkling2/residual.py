# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared attention-residual mixtures over normalized source values, fused with their RMSNorm.

One program per row mixes the sources in FP32 and normalizes the mixture without storing it: the mean
square, the product with the reciprocal root and then with the norm weight are FP32, and only the
result rounds to BF16. That is the arithmetic of the mixture followed by `PreparedRMSNorm`, whose
tuned plans differ from one another only in the summation order of the mean square, so the fused
result differs from any of them by at most one BF16 unit in a few elements per million.
"""

import torch
import triton
import triton.language as tl

from ignition.preparation.examples import declaration_tensor
from ignition.preparation.triton import launch_triton, prepare_triton


@triton.jit
def _mixture(Source0, Source1, Source2, Source3, Source4, Source5, Source6, Source7, Source8,
             Queries, Output0, Output1, Output2, Addend, Norm0, Norm1, Norm2, STRIDES: tl.constexpr,
             HIDDEN: tl.constexpr, SOURCES: tl.constexpr, STREAMS: tl.constexpr, ADD: tl.constexpr,
             ADD_STRIDE: tl.constexpr, EPS: tl.constexpr, NORMALIZE: tl.constexpr, BLOCK: tl.constexpr):
    Sources = (Source0, Source1, Source2, Source3, Source4, Source5, Source6, Source7, Source8)
    Outputs = (Output0, Output1, Output2)
    Norms = (Norm0, Norm1, Norm2)
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    valid = columns < HIDDEN
    for stream in tl.static_range(STREAMS):
        query = tl.load(Queries + stream * HIDDEN + columns, valid, other=0.)
        maximum = tl.full((), -float('inf'), tl.float32)
        denominator = tl.full((), 0., tl.float32)
        mixture = tl.full((BLOCK,), 0., tl.float32)
        for source in tl.static_range(SOURCES):
            value = tl.load(Sources[source] + row * STRIDES[source] + columns, valid, other=0.).to(tl.float32)
            if ADD and source == SOURCES - 1:
                value += tl.load(Addend + row * ADD_STRIDE + columns, valid, other=0.).to(tl.float32)
            inverse = tl.rsqrt(tl.sum(value * value, 0) / HIDDEN + EPS)
            score = tl.sum(value * query, 0) * inverse
            peak = tl.maximum(maximum, score)
            rescale = tl.exp(maximum - peak)
            probability = tl.exp(score - peak)
            denominator = denominator * rescale + probability
            mixture = mixture * rescale + value * (inverse * probability)
            maximum = peak
        mixed = mixture / denominator
        if NORMALIZE:
            # The mixture is scaled before the weight is applied, as the unfused RMSNorm does.
            scale = tl.rsqrt(tl.sum(mixed * mixed, 0) / HIDDEN + EPS)
            weight = tl.load(Norms[stream] + columns, valid, other=0.).to(tl.float32)
            tl.store(Outputs[stream] + row * HIDDEN + columns, (mixed * scale * weight).to(tl.bfloat16), valid)
        else:
            tl.store(Outputs[stream] + row * HIDDEN + columns, mixed, valid)


class PreparedResidualMix:
    """Normalized attention-residual mixtures of one to nine BF16 sources for one to three output streams.

    Calling the object returns each stream's BF16 `RMSNorm(mixture) * norm` from one kernel launch.
    `mix` returns the FP32 mixtures alone, which the fused kernel never stores; tests compare the two.
    """

    def __init__(self, source, queries, norms, *, sources, eps, session, name, addend=False, examples=None):
        del session, name
        if not 1 <= sources <= 9 or not 1 <= len(norms) <= 3:
            raise ValueError('attention-residual mixtures require one to nine sources and one to three output streams')
        if queries.dtype != torch.float32 or queries.shape != (len(norms), source.shape[1]):
            raise ValueError('attention-residual queries must contain one FP32 row per output stream')
        if any(norm.shape != source.shape[1:] or not norm.is_contiguous()
               or norm.dtype not in (torch.bfloat16, torch.float32) for norm in norms):
            raise ValueError('attention-residual norm weights must be contiguous BF16 or FP32 rows of the hidden width')
        self.queries, self.norms, self.sources = queries, tuple(norms), sources
        self.eps, self.streams, self.addend = eps, len(norms), addend
        self.tokens, self.hidden = source.shape
        self.options = dict(HIDDEN=self.hidden, SOURCES=sources, STREAMS=self.streams, ADD=addend,
            ADD_STRIDE=source.stride(0), EPS=eps, BLOCK=triton.next_power_of_2(self.hidden),
            STRIDES=(source.stride(0),) * sources, num_warps=8, enable_fp_fusion=False)
        declarations = {} if examples is None else examples
        outputs = tuple(declaration_tensor(declarations, f"inkling.residual.output{index}",
            tuple(source.shape), torch.bfloat16, source.device) for index in range(len(norms)))
        self.compiled = prepare_triton(_mixture, self._arguments((source,) * sources, outputs, source),
                                       (self.tokens, 1, 1), NORMALIZE=True, **self.options).compiled
        self.unnormalized = None

    def _arguments(self, sources, outputs, addend):
        return (sources + (sources[0],) * (9 - self.sources) + (self.queries,) + outputs
                + (outputs[0],) * (3 - self.streams) + (addend,)
                + self.norms + (self.norms[0],) * (3 - self.streams))

    def _check(self, sources, addend):
        if len(sources) != self.sources:
            raise ValueError('attention-residual source count differs from the prepared layer')
        if (addend is not None) != self.addend:
            raise ValueError('attention-residual addend differs from the prepared layer')

    def mix(self, sources, *, addend=None):
        """The FP32 mixtures without normalization."""
        self._check(sources, addend)
        output = tuple(torch.empty((self.tokens, self.hidden), dtype=torch.float32,
            device=sources[0].device) for _ in range(self.streams))
        arguments = self._arguments(sources, output, sources[0] if addend is None else addend)
        if self.unnormalized is None:
            self.unnormalized = prepare_triton(_mixture, arguments, (self.tokens, 1, 1), NORMALIZE=False,
                                               **self.options).compiled
        launch_triton(_mixture, self.unnormalized, arguments, (self.tokens, 1, 1), NORMALIZE=False, **self.options)
        return output

    def __call__(self, sources, *, addend=None):
        self._check(sources, addend)
        output = tuple(torch.empty((self.tokens, self.hidden), dtype=torch.bfloat16,
            device=sources[0].device) for _ in range(self.streams))
        launch_triton(_mixture, self.compiled, self._arguments(sources, output, sources[0] if addend is None else addend),
                      (self.tokens, 1, 1), NORMALIZE=True, **self.options)
        return output
