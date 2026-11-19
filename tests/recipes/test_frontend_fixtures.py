# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend contract fixtures that the native checkpoint tests load equal the recipes' own."""

import importlib
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parents[2] / "helios/tests/support/contracts"


@pytest.mark.parametrize("recipe", ["step5", "nemotron_h", "gemma4"])
def test_native_frontend_fixture_matches_the_recipe(recipe):
    declared = importlib.import_module(f"ignition.recipes.{recipe}.contract").FRONTEND
    assert json.loads((FIXTURES / f"{recipe}_frontend.json").read_text()) == declared
