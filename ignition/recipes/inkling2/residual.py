# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared FP32 attention-residual mixtures over normalized source values."""

import torch
import triton
import triton.language as tl

from ignition.components.rms_norm import PreparedRMSNorm
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.triton import launch_triton, prepare_triton


@triton.jit
def _mixture(Source0, Source1, Source2, Source3, Source4, Source5, Source6, Source7, Source8,
             Queries, Output0, Output1, Output2, Addend, STRIDES: tl.constexpr, HIDDEN: tl.constexpr,
             SOURCES: tl.constexpr, STREAMS: tl.constexpr, ADD: tl.constexpr, ADD_STRIDE: tl.constexpr,
             EPS: tl.constexpr, BLOCK: tl.constexpr):
    Sources = (Source0, Source1, Source2, Source3, Source4, Source5, Source6, Source7, Source8)
    Outputs = (Output0, Output1, Output2)
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
        tl.store(Outputs[stream] + row * HIDDEN + columns, mixture / denominator, valid)


class PreparedResidualMix:
    def __init__(self, source, queries, norms, *, sources, eps, session, name, addend=False, examples=None):
        if not 1 <= sources <= 9 or not 1 <= len(norms) <= 3:
            raise ValueError('attention-residual mixtures require one to nine sources and one to three output streams')
        if queries.dtype != torch.float32 or queries.shape != (len(norms), source.shape[1]):
            raise ValueError('attention-residual queries must contain one FP32 row per output stream')
        self.queries, self.norms, self.sources = queries, norms, sources
        self.eps, self.streams, self.addend = eps, len(norms), addend
        self.tokens, self.hidden = source.shape
        self.options = dict(HIDDEN=self.hidden, SOURCES=sources, STREAMS=self.streams, ADD=addend,
            ADD_STRIDE=source.stride(0), EPS=eps, BLOCK=triton.next_power_of_2(self.hidden),
            STRIDES=(source.stride(0),) * sources, num_warps=8, enable_fp_fusion=False)
        declarations = {} if examples is None else examples
        mixtures = tuple(declaration_tensor(declarations, f"inkling.residual.mixture{index}",
            tuple(source.shape), torch.float32, source.device) for index in range(len(norms)))
        outputs = tuple(declaration_tensor(declarations, f"inkling.residual.output{index}",
            tuple(source.shape), torch.bfloat16, source.device) for index in range(len(norms)))
        arguments = ((source,) * 9 + (queries,) + mixtures
                     + (mixtures[0],) * (3 - self.streams) + (source,))
        self.compiled = prepare_triton(_mixture, arguments, (self.tokens, 1, 1), **self.options).compiled
        self.normalizers = tuple(PreparedRMSNorm(value, norm, output,
            eps=eps, session=session, name=f'{name}.norm{index}').functional()
            for index, (value, norm, output) in enumerate(zip(mixtures, norms, outputs, strict=True)))

    def mix(self, sources, *, addend=None):
        if len(sources) != self.sources:
            raise ValueError('attention-residual source count differs from the prepared layer')
        output = tuple(torch.empty((self.tokens, self.hidden), dtype=torch.float32,
            device=sources[0].device) for _ in range(self.streams))
        if (addend is not None) != self.addend:
            raise ValueError('attention-residual addend differs from the prepared layer')
        options = self.options
        arguments = (sources + (sources[0],) * (9 - self.sources) + (self.queries,) + output
                     + (output[0],) * (3 - self.streams) + (sources[0] if addend is None else addend,))
        launch_triton(_mixture, self.compiled,
            arguments, (self.tokens, 1, 1), **options)
        return output

    def __call__(self, sources, *, addend=None):
        return tuple(norm(value) for norm, value in zip(self.normalizers,
            self.mix(sources, addend=addend), strict=True))
