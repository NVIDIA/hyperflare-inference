# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Resolved noise tables preserve upstream values without assuming solver policy."""

from dataclasses import FrozenInstanceError

import pytest

from ignition.preparation.noise_schedule import PreparedNoiseSchedule


def test_flow_table_preserves_model_time_separately_from_sigma():
    schedule = PreparedNoiseSchedule(timesteps=(1000, 750, 200), sigmas=(1, .75, .2, 0))
    assert schedule.rows == ((1000., 1., .75), (750., .75, .2), (200., .2, 0.))


def test_sigma_scale_and_partial_endpoint_are_not_normalized():
    schedule = PreparedNoiseSchedule(timesteps=(999, 600), sigmas=(14.6, 2.4, .03))
    assert schedule.rows == ((999., 14.6, 2.4), (600., 2.4, .03))


def test_repeated_evaluations_and_increasing_intervals_remain_explicit():
    schedule = PreparedNoiseSchedule(timesteps=(10, 5, 5, 8), sigmas=(1, .5, .5, .8, 0))
    assert schedule.rows == ((10., 1., .5), (5., .5, .5), (5., .5, .8), (8., .8, 0.))


def test_scheduler_mutation_cannot_change_a_preparation_snapshot():
    timesteps, sigmas = [1000, 500], [1, .5, 0]
    schedule = PreparedNoiseSchedule(timesteps=timesteps, sigmas=sigmas)
    timesteps.clear()
    sigmas[:] = [float("nan")]
    assert schedule.rows == ((1000., 1., .5), (500., .5, 0.))
    with pytest.raises(FrozenInstanceError):
        schedule.sigmas = (0,)


def test_integer_timesteps_do_not_round_through_float_storage():
    timestep = 2**53 + 1
    schedule = PreparedNoiseSchedule(timesteps=(timestep,), sigmas=(1, 0))
    assert schedule.rows[0][0] == timestep
    assert type(schedule.rows[0][0]) is int


@pytest.mark.parametrize("timesteps,sigmas", [
    ((), (0,)), ((1,), (1,)), ((1,), (1, .5, 0)),
    ((1,), (1, -.1)), ((float("nan"),), (1, 0)),
    ((float("inf"),), (1, 0)), ((1,), (float("inf"), 0)),
    ((1,), (1, float("nan"))), ((True,), (1, 0)),
    ((1,), (True, 0)), (("1",), (1, 0)),
])
def test_invalid_tables_fail_before_capture(timesteps, sigmas):
    with pytest.raises(ValueError):
        PreparedNoiseSchedule(timesteps=timesteps, sigmas=sigmas)
