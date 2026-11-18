# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native tensor requests preserve the exported allocation bounds."""

from types import SimpleNamespace
import argparse
import sys

import pytest

from ignition.recipes.sdxl.prepare import (
    DecodeForward, GRAPH_ORDER, PreparationNamespace, canonical_reference,
    graph_order, positive_steps, remove_output_capture_hooks, tensor_view,
)


def tensor(address=1040, *, contiguous=True):
    return SimpleNamespace(data_ptr=lambda: address, numel=lambda: 8,
                           element_size=lambda: 2, shape=(2, 4), dtype="torch.float16",
                           is_contiguous=lambda: contiguous)


def test_request_uses_exported_allocation_and_byte_offset():
    allocation = SimpleNamespace(name="retained-device-0", address=1024, nbytes=64)
    assert tensor_view("latent", tensor(), [allocation]) == dict(
        name="latent", allocation="retained-device-0", offset=16, dtype="F16", shape=[2, 4])


@pytest.mark.parametrize("address", [1016, 1080])
def test_request_rejects_tensor_crossing_allocation_boundary(address):
    allocation = SimpleNamespace(name="storage", address=1024, nbytes=64)
    with pytest.raises(ValueError, match="expected one exported allocation"):
        tensor_view("latent", tensor(address), [allocation])


def test_request_rejects_ambiguous_allocation_ownership():
    allocations = [SimpleNamespace(name=name, address=1024, nbytes=64) for name in ("first", "second")]
    with pytest.raises(ValueError, match="expected one exported allocation"):
        tensor_view("latent", tensor(), allocations)


def test_request_rejects_noncontiguous_tensor():
    allocation = SimpleNamespace(name="storage", address=1024, nbytes=64)
    with pytest.raises(ValueError, match="nonempty and contiguous"):
        tensor_view("latent", tensor(contiguous=False), [allocation])


def test_prepare_removes_only_transformers_output_observation_hooks():
    def capture(module, args, output):
        return None

    capture.__module__ = "transformers.utils.output_capturing"
    capture.__qualname__ = "install_output_capuring_hook.<locals>.output_capturing_hook"

    def application_hook(module, args, output):
        return output

    layer = SimpleNamespace(_forward_hooks={1: capture, 2: application_hook},
                            _forward_hooks_with_kwargs={1: True, 2: True},
                            _forward_hooks_always_called={1: True})
    model = SimpleNamespace(modules=lambda: [layer])
    assert remove_output_capture_hooks(model) == 1
    assert layer._forward_hooks == {2: application_hook}
    assert layer._forward_hooks_with_kwargs == {2: True}
    assert layer._forward_hooks_always_called == {}
    assert remove_output_capture_hooks(model) == 0


@pytest.mark.parametrize("steps", [4, 30])
def test_schedule_changes_graph_order_cache_identity_and_decoder_input(steps):
    order = graph_order(steps)
    assert order[:2] == ("text", "init")
    assert order[2:-1] == tuple(f"step_{index}" for index in range(steps))
    assert order[-1] == "decode"
    assert len(order) == steps + 3
    buffers = {f"latent_after_step_{index}": object() for index in range(steps)}
    buffers["decoded_image"] = object()
    vae = SimpleNamespace(post_quant_conv=SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype="F32")])),
                          config=SimpleNamespace(scaling_factor=0.13025))
    decoder = DecodeForward(SimpleNamespace(vae=vae), buffers, steps)
    assert decoder.input is buffers[f"latent_after_step_{steps - 1}"]
    assert decoder.output is buffers["decoded_image"]
    assert PreparationNamespace("revision", steps).to_dict()["steps"] == steps
    assert PreparationNamespace("revision", 4).to_dict() != PreparationNamespace("revision", 30).to_dict()


def test_four_step_defaults_are_preserved():
    assert graph_order() == GRAPH_ORDER == ("text", "init", "step_0", "step_1", "step_2", "step_3", "decode")
    assert PreparationNamespace("revision").to_dict()["steps"] == 4
    assert positive_steps("30") == 30


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_cli_rejects_invalid_step_counts(value):
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        positive_steps(value)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_graph_order_rejects_invalid_step_counts(value):
    with pytest.raises(ValueError, match="positive integer"):
        graph_order(value)


@pytest.mark.parametrize("steps", [4, 30])
def test_canonical_pipeline_receives_requested_step_count(monkeypatch, steps):
    class Pipeline:
        def __init__(self):
            self.prepare_latents = object()
            self.vae = SimpleNamespace(decode=object())
            self.image_processor = SimpleNamespace(postprocess=object())

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(images="raw image")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    pipe = Pipeline()
    original = pipe.image_processor.postprocess
    assert canonical_reference(pipe, "prompt", {"initial_noise": "noise"}, capture=False, steps=steps) == "raw image"
    assert pipe.kwargs["num_inference_steps"] == steps
    assert pipe.kwargs["latents"] == "noise"
    assert pipe.image_processor.postprocess is original
