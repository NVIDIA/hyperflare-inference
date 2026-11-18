# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Guard serialization failures identify the source without weakening checks."""

import pytest
from torch._dynamo.exc import PackageError
from torch._dynamo.guards import CheckFunctionManager, GuardBuilder
from torch._dynamo.source import LocalSource
from torch._guards import Guard

from ignition.preparation.aot import _describe_guard_failure


def test_rejected_guard_note_preserves_the_upstream_error():
    guard = Guard(LocalSource("forward"), GuardBuilder.ID_MATCH)
    guard.code_list = ["check_object_id(forward, 123)"]
    with pytest.raises(PackageError, match="ID_MATCH guard cannot be serialized") as caught:
        CheckFunctionManager.serialize_guards(None, None, [guard], None)
    error = caught.value
    _describe_guard_failure(error)
    assert len(error.__notes__) == 1
    assert "L['forward']" in error.__notes__[0]
    assert "create_fn=ID_MATCH" in error.__notes__[0]
    assert "check_object_id(forward, 123)" in error.__notes__[0]
    assert str(error) == "ID_MATCH guard cannot be serialized."


def test_rejected_guard_diagnostic_is_bounded():
    guard = Guard(LocalSource("x" * 10000), GuardBuilder.ID_MATCH)
    guard.code_list = ["y" * 10000] * 20
    with pytest.raises(PackageError) as caught:
        CheckFunctionManager.serialize_guards(None, None, [guard], None)
    _describe_guard_failure(caught.value)
    assert len(caught.value.__notes__[0]) < 2100
