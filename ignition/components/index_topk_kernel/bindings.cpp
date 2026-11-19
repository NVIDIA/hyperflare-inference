// SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
// SPDX-License-Identifier: Apache-2.0

#include <torch/csrc/stable/library.h>
#include "csrc/ops.h"

STABLE_TORCH_LIBRARY(flare_reference_topk, m) {
    m.def("prefill(Tensor logits, Tensor starts, Tensor ends, Tensor(a!) output, int rows, int stride0, int stride1, int topk) -> ()");
    m.def("persistent(Tensor logits, Tensor lengths, Tensor(a!) output, Tensor(b!) workspace, int topk, int max_seq_len) -> ()");
    m.def("cooperative(Tensor logits, Tensor lengths, Tensor(a!) output, Tensor(b!) workspace, int topk, int max_seq_len) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(flare_reference_topk, CUDA, m) {
    m.impl("prefill", TORCH_BOX(&top_k_per_row_prefill));
    m.impl("persistent", TORCH_BOX(&persistent_topk));
    m.impl("cooperative", TORCH_BOX(&cooperative_topk));
}
