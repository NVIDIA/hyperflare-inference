# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiler fixes install independently in every preparation process."""

import subprocess
import sys
import textwrap

import pytest

from ignition import runtime_patches
from ignition.errors import PreparationError


_FACTOR = 3
_DEFAULT = object()


def _example(value, default=_DEFAULT, *, offset=2):
    def scale():
        return value * _FACTOR
    return scale() + offset


def test_checked_replacement_retains_live_globals_defaults_and_nested_functions(monkeypatch):
    replacement = runtime_patches._replace(_example, (("value * _FACTOR", "value * _FACTOR + 1"),))
    assert replacement(4) == 15
    assert _example(4) == 14
    monkeypatch.setattr(sys.modules[__name__], "_FACTOR", 5)
    assert replacement(4, offset=7) == 28
    assert replacement.__globals__ is _example.__globals__
    assert replacement.__defaults__ is _example.__defaults__
    assert replacement.__kwdefaults__ is _example.__kwdefaults__


@pytest.mark.parametrize("before", ["missing context", "value"])
def test_missing_or_ambiguous_context_is_rejected(before):
    with pytest.raises(PreparationError, match="context is missing or ambiguous"):
        runtime_patches._replace(_example, ((before, "replacement"),))


def test_incompatible_triton_version_is_rejected_before_installation(monkeypatch):
    monkeypatch.setattr(runtime_patches, "_TRITON_APPLIED", False)
    monkeypatch.setattr(runtime_patches, "version", lambda name: "3.7.0")
    with pytest.raises(PreparationError, match="require triton==3.8.0"):
        runtime_patches.apply_triton()
    assert not runtime_patches._TRITON_APPLIED


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_missing_triton_is_allowed_only_on_windows(monkeypatch, platform):
    from types import SimpleNamespace

    def missing(name):
        raise runtime_patches.PackageNotFoundError(name)

    monkeypatch.setattr(runtime_patches, "_TRITON_APPLIED", False)
    monkeypatch.setattr(runtime_patches, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(runtime_patches, "version", missing)
    if platform == "win32":
        runtime_patches.apply_triton()
    else:
        with pytest.raises(PreparationError, match="require the preparation dependencies"):
            runtime_patches.apply_triton()
    assert not runtime_patches._TRITON_APPLIED


_LAYOUT_HELPERS = r'''
import torch
import torch._refs as refs
import torch._prims as prims
import torch._prims_common as common
import torch._subclasses.fake_tensor as fake_tensor
import torch._subclasses.fake_impls as fake_impls
import torch._library.custom_ops as custom_ops
from torch._subclasses.fake_tensor import FakeTensorMode
from ignition import runtime_patches

def _layout_targets():
    targets = [(refs, "_reshape_view_helper_core_alg"),
               (FakeTensorMode, "_dispatch_impl"), (FakeTensorMode, "_cache_key")]
    for name in ("add", "sub", "mul", "div"):
        packet = getattr(torch.ops.prims, name)
        targets.extend(((packet, "prim_meta_impl"), (packet.default, "prim_meta_impl"),
                        (custom_ops.OPDEFS["prims::" + name], "_abstract_fn")))
    return [(owner, name, getattr(owner, name)) for owner, name in targets]

def _layout_untouched():
    targets = [(common, "compute_elementwise_output_logical_to_physical_perm"),
               (prims, "_prim_elementwise_meta"), (refs, "_reshape_view_helper"),
               (fake_impls, "_compute_stride")]
    for name in ("pow", "gt", "remainder", "neg"):
        packet = getattr(torch.ops.prims, name)
        targets.extend(((packet, "prim_meta_impl"), (packet.default, "prim_meta_impl"),
                        (custom_ops.OPDEFS["prims::" + name], "_abstract_fn")))
    return [(owner, name, getattr(owner, name)) for owner, name in targets]

def _assert_retained(targets):
    assert all(getattr(owner, name) is value for owner, name, value in targets)

def _metadata(value):
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
    def concrete(item):
        if isinstance(item, torch.SymInt):
            assert not free_unbacked_symbols(item)
            assert type(item.node.hint) is int
            return item.node.hint
        return int(item)
    return (tuple(concrete(item) for item in value.shape),
            tuple(concrete(item) for item in value.stride()), value.dtype,
            concrete(value.storage_offset()))

def _check_layout_behavior():
    from torch.fx.experimental.proxy_tensor import make_fx
    source = torch.arange(60, dtype=torch.float32).as_strided((1, 4, 3, 5), (4, 1, 20, 4))
    projection = torch.arange(280, dtype=torch.float32).view(1, 35, 8)
    for value, function in ((source, lambda item: item / 1.0),
                            (projection, lambda item: item.transpose(1, 2).view(1, 8, 5, 7))):
        expected = _metadata(function(value))
        meta = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device="meta")
        assert _metadata(function(meta)) == expected
        mode = FakeTensorMode()
        fake = mode.from_tensor(value)
        with mode:
            assert _metadata(function(fake)) == expected
        graph = make_fx(function, tracing_mode="symbolic")(value)
        result = next(node for node in graph.graph.nodes if node.op == "output").args[0]
        assert _metadata(result.meta["val"]) == expected
    assert not torch.cuda.is_initialized()
'''


def _run(source, *, layout=False):
    script = textwrap.dedent(source)
    if layout:
        script = "_LAYOUT_HELPERS = " + repr(_LAYOUT_HELPERS) + "\n" + _LAYOUT_HELPERS + script
    subprocess.run([sys.executable, "-c", script], check=True, timeout=180)


def test_apply_is_idempotent_and_keeps_dependency_sources_unchanged():
    _run('''
        from pathlib import Path
        import torch
        from torch._dynamo import output_graph, symbolic_convert
        from torch._inductor import memory, scheduler, pattern_matcher
        from torch._inductor.fx_passes import post_grad
        from torch._inductor.runtime import static_triton_launcher, triton_heuristics
        from triton.runtime import build
        from ignition import runtime_patches

        modules = (output_graph, symbolic_convert, memory, scheduler, pattern_matcher, post_grad,
                   static_triton_launcher, triton_heuristics, build, refs, prims, common,
                   fake_tensor, fake_impls, custom_ops)
        before = {Path(module.__file__): Path(module.__file__).read_bytes() for module in modules}
        untouched = _layout_untouched()
        generic_matcher = pattern_matcher.PatternMatcherPass.apply
        torch.compiler.config.cache_key_tag = "caller-tag"
        runtime_patches.apply()
        targets = (memory.assign_memory_planning_info_for_scheduler_nodes,
                   post_grad.decompose_auto_functionalized, scheduler.BaseSchedulerNode.decide_inplace_update,
                   output_graph.OutputGraph.check_event_record_after_input_mutation,
                   output_graph.OutputGraph.check_input_mutation_on_current_stream)
        layout_targets = _layout_targets()
        for name in ("add", "sub", "mul", "div"):
            packet = getattr(torch.ops.prims, name)
            assert packet.prim_meta_impl is packet.default.prim_meta_impl
            assert packet.prim_meta_impl is custom_ops.OPDEFS["prims::" + name]._abstract_fn
        _check_layout_behavior()
        expected_tag = "caller-tag;" + runtime_patches.identity()
        assert torch.compiler.config.cache_key_tag == expected_tag
        runtime_patches.apply()
        assert targets == (memory.assign_memory_planning_info_for_scheduler_nodes,
                           post_grad.decompose_auto_functionalized, scheduler.BaseSchedulerNode.decide_inplace_update,
                           output_graph.OutputGraph.check_event_record_after_input_mutation,
                           output_graph.OutputGraph.check_input_mutation_on_current_stream)
        assert torch.compiler.config.cache_key_tag == expected_tag
        _assert_retained(layout_targets)
        _assert_retained(untouched)
        _check_layout_behavior()
        assert pattern_matcher.PatternMatcherPass.apply is generic_matcher
        assert all(path.read_bytes() == content for path, content in before.items())
        assert not torch.cuda.is_initialized()
    ''', layout=True)


def test_incompatible_version_and_context_publish_no_partial_patch():
    _run('''
        import torch
        from torch._dynamo.output_graph import OutputGraph
        from torch._inductor import scheduler
        from torch._inductor.fx_passes import post_grad
        from ignition import runtime_patches
        from ignition.errors import PreparationError

        original = (OutputGraph.check_event_record_after_input_mutation,
                    post_grad.decompose_auto_functionalized, scheduler.BaseSchedulerNode.decide_inplace_update)
        layout_targets = _layout_targets()
        untouched = _layout_untouched()
        torch.compiler.config.cache_key_tag = "caller-tag"
        version = torch.__version__
        torch.__version__ = "unsupported"
        try:
            runtime_patches.apply()
        except PreparationError as error:
            assert "require torch==" in str(error)
        else:
            raise AssertionError("unsupported Torch accepted")
        torch.__version__ = version
        from ignition.runtime_patches import scheduler as patches
        builder = patches.replacements
        def incompatible():
            raise PreparationError("missing scheduler context")
        patches.replacements = incompatible
        try:
            runtime_patches.apply()
        except PreparationError as error:
            assert "missing scheduler context" in str(error)
        else:
            raise AssertionError("incompatible source accepted")
        assert original == (OutputGraph.check_event_record_after_input_mutation,
                            post_grad.decompose_auto_functionalized, scheduler.BaseSchedulerNode.decide_inplace_update)
        _assert_retained(layout_targets)
        _assert_retained(untouched)
        assert torch.compiler.config.cache_key_tag == "caller-tag"
        assert not runtime_patches._APPLIED
        patches.replacements = builder
        runtime_patches.apply()
        _check_layout_behavior()
    ''', layout=True)


@pytest.mark.parametrize("kind", ["SPAWN", "FORK"])
def test_inductor_worker_installs_matching_patch_identity_before_first_job(kind):
    _run('''
        import torch
        from torch._inductor.compile_worker.subproc_pool import SubprocKind, SubprocPool
        from ignition import runtime_patches

        runtime_patches.apply()
        pool = SubprocPool(1, kind=SubprocKind.KIND)
        try:
            worker_tag = pool.submit(eval, "__import__('torch').compiler.config.cache_key_tag").result(timeout=30)
            assert worker_tag == torch.compiler.config.cache_key_tag
            pool.submit(exec, _LAYOUT_HELPERS + "\\n_check_layout_behavior()\\n", {}).result(timeout=30)
            pool.submit(runtime_patches.apply).result(timeout=30)
            repeated = pool.submit(eval, "__import__('torch').compiler.config.cache_key_tag").result(timeout=30)
            assert repeated == worker_tag
        finally:
            pool.shutdown()
    '''.replace("SubprocKind.KIND", f"SubprocKind.{kind}"), layout=True)


def test_triton_bootstrap_does_not_import_torch():
    _run('''
        import sys
        from ignition import runtime_patches
        runtime_patches.apply_triton()
        from triton.runtime.build import platform_key
        assert platform_key.__module__ == "ignition.runtime_patches.triton"
        assert "torch" not in sys.modules
    ''')


def test_direct_spawned_inductor_worker_installs_before_first_job():
    _run('''
        import multiprocessing
        import os
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        from ignition import runtime_patches
        from torch._inductor import async_compile

        runtime_patches.apply()
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
                initializer=partial(async_compile._async_compile_initializer, os.getpid())) as pool:
            tag = pool.submit(eval, "__import__('torch').compiler.config.cache_key_tag").result(timeout=120)
            assert tag == runtime_patches.identity()
            pool.submit(exec, _LAYOUT_HELPERS + "\\n_check_layout_behavior()\\n", {}).result(timeout=120)
    ''', layout=True)


def test_b12x_offline_worker_installs_triton_fix_before_first_job():
    _run('''
        from ignition import runtime_patches
        from b12x._lib.compile_pool import CompilePool

        runtime_patches.apply()
        with CompilePool(device_ordinal=0, compute_capability=(12, 0), device_uuid="runtime-patch-test",
                         product_name="NVIDIA RTX PRO 6000 Blackwell", sm_count=188,
                         max_shared_memory_per_block=101376, max_shared_memory_per_multiprocessor=102400,
                         workers=1) as pool:
            result = pool._pool.apply_async(eval, ("(__import__('triton.runtime.build', fromlist=['platform_key'])."
                "platform_key.__module__, __import__('torch').cuda.is_initialized())",)).get(timeout=120)
            assert result == ("ignition.runtime_patches.triton", False)
    ''')



def test_partial_installation_restores_metadata_attributes_and_can_retry():
    _run('''
        from ignition.runtime_patches import tensor_layout
        from ignition.errors import PreparationError

        originals = _layout_targets()
        untouched = _layout_untouched()
        torch.compiler.config.cache_key_tag = "caller-tag"
        builder = tensor_layout.replacements
        built = builder()
        _assert_retained(originals)
        _assert_retained(untouched)

        class RejectOnce:
            def __setattr__(self, name, value):
                if name == "trigger" and value is marker:
                    _assert_retained(untouched)
                    assert all(getattr(owner, attribute) is not original
                               for owner, attribute, original in originals)
                    raise PreparationError("injected publication failure")
                object.__setattr__(self, name, value)

        marker = object()
        rejected = RejectOnce()
        rejected.trigger = None
        tensor_layout.replacements = lambda: (*built, (rejected, "trigger", marker))
        try:
            runtime_patches.apply()
        except PreparationError as error:
            assert str(error) == "injected publication failure"
        else:
            raise AssertionError("injected setter failure was not reached")
        finally:
            tensor_layout.replacements = builder
        _assert_retained(originals)
        _assert_retained(untouched)
        assert rejected.trigger is None
        assert torch.compiler.config.cache_key_tag == "caller-tag"
        assert not runtime_patches._APPLIED
        runtime_patches.apply()
        _check_layout_behavior()
        assert torch.compiler.config.cache_key_tag == "caller-tag;" + runtime_patches.identity()
    ''', layout=True)


def test_primed_cpu_fake_caches_survive_patch_installation():
    # CPU fakes populate both metadata caches before installation without
    # initializing CUDA. CUDA metadata is checked separately after apply().
    subprocess.run([sys.executable, "-c", textwrap.dedent('''
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv, free_unbacked_symbols
        from torch.fx.experimental.proxy_tensor import make_fx
        from ignition import runtime_patches

        def metadata(value):
            def concrete(item):
                if isinstance(item, torch.SymInt):
                    assert not free_unbacked_symbols(item)
                    assert type(item.node.hint) is int
                    return item.node.hint
                return int(item)
            return (tuple(concrete(x) for x in value.shape),
                    tuple(concrete(x) for x in value.stride()), value.dtype,
                    concrete(value.storage_offset()))

        source = torch.ones(60).as_strided((1, 4, 3, 5), (4, 1, 20, 4))
        expected = metadata(source / 1.0)
        view_source = torch.arange(280, dtype=torch.float32).view(1, 35, 8).transpose(1, 2)
        view_expected = metadata(view_source.view(1, 8, 5, 7))

        def unrelated_metadata():
            # Preserve existing behavior without claiming these unrelated
            # singleton-layout cases agree with native execution.
            result = []
            for function in (lambda value: torch.gt(value, 2.5),
                             lambda value: torch.remainder(value, 2.5),
                             lambda value: torch.ops.prims.gt.default(value, 2.5),
                             lambda value: torch.ops.prims.remainder.default(value, 2.5)):
                meta = torch.empty_strided(source.shape, source.stride(), device="meta")
                mode = FakeTensorMode()
                fake = mode.from_tensor(source)
                with mode:
                    fake_result = metadata(function(fake))
                traced = make_fx(function, tracing_mode="symbolic")(source)
                output = next(node for node in traced.graph.nodes if node.op == "output").args[0]
                result.append((metadata(function(meta)), fake_result, metadata(output.meta["val"])))
            return result

        unrelated_before = unrelated_metadata()
        probes = []
        for symbolic in (False, True):
            mode = FakeTensorMode(shape_env=ShapeEnv(), static_shapes=False) if symbolic else FakeTensorMode()
            fake = mode.from_tensor(source, static_shapes=not symbolic)
            fake_view = mode.from_tensor(view_source, static_shapes=not symbolic)
            cache = mode.shape_env.fake_tensor_cache if symbolic else FakeTensorMode.cache
            with mode:
                assert metadata(fake / 1.0) == metadata(fake / 1.0)
                assert metadata(fake_view.view(1, 8, 5, 7)) == metadata(fake_view.view(1, 8, 5, 7))
            retained = dict(cache)
            assert any(key.key[0] is torch.ops.aten.div.Tensor for key in retained)
            assert any(key.key[0] is torch.ops.aten.view.default for key in retained)
            probes.append((mode, fake, fake_view, cache, retained))
        assert not torch.cuda.is_initialized()
        runtime_patches.apply()
        identity = torch.compiler.config.cache_key_tag
        assert unrelated_metadata() == unrelated_before
        for mode, fake, fake_view, cache, retained in probes:
            with mode:
                assert metadata(fake / 1.0) == expected
                hits = FakeTensorMode.cache_hits
                assert metadata(fake / 1.0) == expected
                assert FakeTensorMode.cache_hits > hits
                assert metadata(fake_view.view(1, 8, 5, 7)) == view_expected
                hits = FakeTensorMode.cache_hits
                assert metadata(fake_view.view(1, 8, 5, 7)) == view_expected
                assert FakeTensorMode.cache_hits > hits
            assert all(cache.get(key) is value for key, value in retained.items())
            for operation in (torch.ops.aten.div.Tensor, torch.ops.aten.view.default):
                corrected = [key for key in cache if key not in retained and key.key[0] is operation]
                assert corrected
                assert all(key.key[1] == runtime_patches.identity() for key in corrected)
        runtime_patches.apply()
        assert torch.compiler.config.cache_key_tag == identity
        assert not torch.cuda.is_initialized()
    ''')], check=True, timeout=180)


def test_zero_scalar_invalid_and_unbacked_views_preserve_behavior():
    _run(r'''
        def normalize_expression(value, symbols):
            if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                value = value.node.expr
            if hasattr(value, "xreplace"):
                value = value.xreplace(symbols)
            return str(value)

        def symbolic_state(environment, symbols):
            render = lambda value: normalize_expression(value, symbols)
            return dict(
                guards=[render(item.expr) for item in environment.guards],
                deferred_runtime_asserts=sorted(
                    (render(key), [render(item.expr) for item in items])
                    for key, items in environment.deferred_runtime_asserts.items()),
                ranges=sorted((render(key), render(value.lower), render(value.upper))
                              for key, value in environment.var_to_range.items()),
                replacements=sorted((render(key), render(value))
                                    for key, value in environment.replacements.items()),
                divisible=sorted(render(item) for item in environment.divisible),
            )

        def symbolic_metadata(value, symbols):
            render = lambda item: normalize_expression(item, symbols)
            return dict(shape=[render(item) for item in value.shape],
                        stride=[render(item) for item in value.stride()],
                        storage_offset=render(value.storage_offset()), dtype=str(value.dtype),
                        device=str(value.device))

        def unbacked_case(name, path):
            import sympy
            import torch._refs as refs
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.fx.experimental.symbolic_shapes import ShapeEnv

            environment = ShapeEnv()
            mode = FakeTensorMode(shape_env=environment, static_shapes=False)
            symbol = environment.create_unbacked_symint()
            symbols = {symbol.node.expr: sympy.Symbol("unbacked")}
            with mode:
                # Apply equal size constraints without assigning a concrete symbol hint.
                torch._check_is_size(symbol)
                torch._check(symbol >= 2)
                if name == "contiguous_dimension":
                    shape, strides, target = (symbol, 8), (8, 1), (symbol, 2, 4)
                elif name == "gapped_dimension":
                    shape, strides, target = (symbol, 8), (16, 2), (symbol, 2, 4)
                else:
                    shape, strides, target = (2, 8), (symbol, 1), (16,)
                # Construction failures are structural failures, not preservation passes.
                source = torch.empty_strided(shape, strides, dtype=torch.float32, device="cpu")
                source_metadata = symbolic_metadata(source, symbols)
                before = symbolic_state(environment, symbols)
                try:
                    if path.startswith("core_"):
                        result = refs._reshape_view_helper_core_alg(source, target,
                                                                    allow_copy=path == "core_reshape")
                    elif path.startswith("reference_"):
                        result = getattr(refs, path.removeprefix("reference_"))(source, target)
                    else:
                        result = getattr(source, path.removeprefix("fake_"))(target)
                    outcome = dict(metadata=symbolic_metadata(result, symbols),
                                   aliases_input=result.untyped_storage()._cdata == source.untyped_storage()._cdata)
                except (ValueError, RuntimeError) as error:
                    outcome = dict(error_type=type(error).__module__ + "." + type(error).__qualname__,
                                   message=str(error))
                after = symbolic_state(environment, symbols)
                unchanged = symbolic_metadata(source, symbols) == source_metadata
            return dict(case=name, path=path, input=source_metadata,
                        target=[normalize_expression(item, symbols) for item in target],
                        before=before, outcome=outcome, after=after,
                        source_metadata_unchanged=unchanged)

        def unbacked_identity(record):
            # Generated source filenames are not part of the contract. Error
            # classes, guards and deferred assertions must remain identical.
            return {**record, "outcome": {key: value for key, value in record["outcome"].items()
                                          if key != "message"}}


        def edge_cases():
            records = []
            cases = [((0,3),(91,2),0,(0,3)), ((0,3),(91,2),0,(3,0)),
                   ((),(),7,(1,1)), ((),(),7,()), ((1,1),(97,3),7,()),
                   ((2,3),(3,1),0,(5,))]
            for shape, strides, offset, target in cases:
                extent = offset + 1 + sum((size - 1) * stride for size, stride in zip(shape, strides) if size)
                backing = torch.arange(max(extent, offset + 1), dtype=torch.float32)
                source = backing.as_strided(shape, strides, offset)
                saved = backing.clone()
                for operation in ("view", "reshape"):
                    try:
                        result = getattr(refs, operation)(source, target)
                        records.append((_metadata(result),
                            result.untyped_storage()._cdata == source.untyped_storage()._cdata,
                            result.contiguous().reshape(-1).view(torch.uint8).tolist()))
                    except (ValueError, RuntimeError) as error:
                        records.append((type(error).__module__, type(error).__qualname__))
                    assert torch.equal(backing.view(torch.uint8), saved.view(torch.uint8))
            return records


        def unbacked_cases():
            records = [unbacked_case(case, path)
                     for case in ("contiguous_dimension","gapped_dimension","unknown_stride")
                     for path in ("core_view","core_reshape","reference_view","reference_reshape",
                                  "fake_view","fake_reshape")]
            assert all(record["source_metadata_unchanged"] for record in records)
            assert any("error_type" in record["outcome"] for record in records)
            for prefix in ("core_","reference_","fake_"):
                assert any(record["path"].startswith(prefix) and "error_type" not in record["outcome"]
                           for record in records)
            return [unbacked_identity(record) for record in records]

        before_edges, before_unbacked = edge_cases(), unbacked_cases()
        assert not torch.cuda.is_initialized()
        runtime_patches.apply()
        assert edge_cases() == before_edges
        assert unbacked_cases() == before_unbacked
        assert not torch.cuda.is_initialized()
    ''', layout=True)
