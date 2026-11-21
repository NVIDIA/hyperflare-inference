# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of which hidden state and cache slot each GLM MoE DSA draft row uses."""

from types import SimpleNamespace

import torch

from ignition.components.paged_batch import Metadata
from ignition.recipes.glm_dsa.mtp import DraftState, PreparedFeedback, PreparedProposals


def _feedback(metadata, hidden, saved):
    operation, seen = PreparedFeedback.__new__(PreparedFeedback), {}
    operation.layer = SimpleNamespace(cache=lambda previous, m: seen.update(previous=previous, metadata=m))
    operation((hidden, saved), metadata.lengths, metadata)
    return seen["previous"]


def test_feedback_joins_each_row_with_the_hidden_state_before_it():
    # Two requests verify three and two positions, and a third is inactive.
    starts, lengths = torch.tensor([10, 7, 0]), torch.tensor([3, 2, 0], dtype=torch.int32)
    tokens = torch.arange(9).view(3, 3) + 100
    metadata = Metadata.pack(starts, lengths, torch.tensor([4, 5, 0]), torch.zeros((3, 2), dtype=torch.int32),
                             tokens, torch.tensor([3], dtype=torch.int32), token_capacity=9, sample_count=9)
    hidden = torch.arange(9.)[:, None].expand(9, 4) + 1
    saved = -(torch.arange(3.)[:, None].expand(3, 4) + 1)
    previous = _feedback(metadata, hidden, saved)
    assert metadata.positions.tolist() == [10, 11, 12, 7, 8, -1, -1, -1, -1]
    # A request's first row joins its saved state, and each later row joins the row before it.
    assert previous[:5, 0].tolist() == [-1., 1., 2., -2., 4.]


def test_feedback_joins_a_sequence_start_with_a_zero_state():
    metadata = Metadata.pack(torch.tensor([0]), torch.tensor([4], dtype=torch.int32), torch.tensor([3]),
                             torch.zeros((1, 2), dtype=torch.int32), torch.arange(4).view(1, 4),
                             torch.tensor([1], dtype=torch.int32), token_capacity=4)
    hidden = torch.arange(4.)[:, None].expand(4, 4) + 1
    previous = _feedback(metadata, hidden, torch.full((1, 4), 9.))
    assert previous[:, 0].tolist() == [0., 1., 2., 3.]


def test_proposals_write_one_slot_per_proposing_request():
    batch, capacity = 3, 4
    proposals = PreparedProposals.__new__(PreparedProposals)
    proposals.batch, proposals.capacity = batch, capacity
    tables = torch.arange(capacity * 2, dtype=torch.int32).view(capacity, 2)
    proposals.inputs = SimpleNamespace(row_slot_ids=torch.tensor([2, 0, -1, -1], dtype=torch.int32),
                                       page_tables=tables, live_rows=torch.tensor([2], dtype=torch.int32))
    received = []

    def rest(previous, metadata, selection):
        received.append(metadata)
        return previous, None, selection

    proposals.rest = rest
    state = DraftState(torch.zeros(batch, 4), None)
    proposals.step(state, 2, torch.tensor([31, 32, 33]), torch.tensor([20, 5, 0]), torch.tensor([6, 7, 0]),
                   torch.tensor([True, True, False]), torch.tensor([True, False, False]))
    metadata, = received
    # The pending token occupies slot `before`, and proposal step i writes slot `before + i`.
    assert metadata.positions.tolist() == [22, -1, -1] and metadata.token_ids.tolist() == [31, 0, 0]
    assert metadata.page_table.tolist() == [[4, 5], [-1, -1], [-1, -1]] and metadata.state_rows.tolist() == [6, 0, 0]
    assert metadata.live_counts.tolist() == [3, 2] and metadata.lengths.tolist() == [1, 0, 0]
