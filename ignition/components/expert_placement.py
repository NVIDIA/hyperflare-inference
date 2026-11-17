# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical expert coverage for explicit HBM and mapped-host weight rows."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertPlacement:
    """Physical row order; an HBM copy overrides the same expert's host copy."""

    experts: int
    host: tuple[int, ...]
    hbm: tuple[int, ...]

    def __post_init__(self):
        if (type(self.experts) is not int or self.experts <= 0
                or not isinstance(self.host, tuple) or not isinstance(self.hbm, tuple)
                or not self.hbm
                or any(type(i) is not int or not 0 <= i < self.experts for i in self.host + self.hbm)
                or any(len(set(rows)) != len(rows) for rows in (self.host, self.hbm))
                or set(self.host + self.hbm) != set(range(self.experts))):
            raise ValueError("expert placement requires complete canonical coverage and unique rows in each tier, with nonempty HBM")

    @property
    def physical_rows(self):
        return len(self.host) + len(self.hbm)

    def to_dict(self):
        return dict(experts=self.experts, host=list(self.host), hbm=list(self.hbm))
