# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Immutable preparation snapshots of resolved sigma-based noise schedules."""

from dataclasses import dataclass
from math import isfinite


def _finite_values(values, name):
    result = []
    for value in values:
        if type(value) not in (int, float) or type(value) is float and not isfinite(value):
            raise ValueError(f"{name} must contain finite numbers")
        result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class PreparedNoiseSchedule:
    """Keep model timesteps and sigma intervals after canonical schedule setup.

    Recipes resolve their upstream scheduler during preparation, then copy its
    timesteps and sigmas into this table. Each row contains a model timestep,
    its sigma, and the following sigma. Row count does not specify model-forward
    count or solver order. Repeated timesteps, increasing intervals, and nonzero
    terminal sigmas retain their supplied values.

    This table neither constructs a schedule nor implements a solver. Captured
    forwards own prediction conversion, guidance, solver history, and stochastic
    updates. Discrete token diffusion need not have a sigma-based schedule.
    Captured recipe bindings determine tensor dtype; this snapshot holds host
    scalar values without conversion or quantization.
    """

    timesteps: tuple[int | float, ...]
    sigmas: tuple[int | float, ...]

    def __post_init__(self):
        timesteps = _finite_values(self.timesteps, "timesteps")
        sigmas = _finite_values(self.sigmas, "sigmas")
        if not timesteps:
            raise ValueError("a prepared noise schedule requires at least one timestep")
        if len(sigmas) != len(timesteps) + 1:
            raise ValueError("sigmas must include one terminal value after the timesteps")
        if any(sigma < 0 for sigma in sigmas):
            raise ValueError("sigmas must be nonnegative")
        object.__setattr__(self, "timesteps", timesteps)
        object.__setattr__(self, "sigmas", sigmas)

    @property
    def rows(self) -> tuple[tuple[int | float, int | float, int | float], ...]:
        """Return unchanged model timesteps and their adjacent sigma values."""
        return tuple(zip(self.timesteps, self.sigmas[:-1], self.sigmas[1:], strict=True))
