import ctypes
import errno
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_jit():
    repository = Path(__file__).resolve().parents[2]
    path = repository / "symmetrix" / "source" / "symmetrix" / "jit.py"
    spec = importlib.util.spec_from_file_location("symmetrix_jit_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def jit():
    return _load_jit()


@pytest.fixture(scope="module")
def cxx():
    command = os.environ.get("CXX", "c++").split()[0]
    compiler = shutil.which(command)
    if compiler is None:
        pytest.skip("a C++ compiler is required for Execution JIT cache tests")
    return compiler


def test_cuda_r1_edge_launch_policy_is_validated_and_tunable(jit, monkeypatch):
    assert jit.execution_cuda_r1_edge_launch_policy() == {
        "strategy": "wave",
        "logical_subgroup_width": 32,
        "edge_threads_per_block": 32,
        "persistent_blocks_per_compute_unit": 8,
    }
    assert jit.execution_cuda_r1_edge_launch_policy(
        default_strategy="wave",
        default_logical_subgroup_width=32,
        default_edge_threads_per_block=32,
        default_persistent_blocks_per_compute_unit=32,
    ) == {
        "strategy": "wave",
        "logical_subgroup_width": 32,
        "edge_threads_per_block": 32,
        "persistent_blocks_per_compute_unit": 32,
    }
    monkeypatch.setenv(jit.CUDA_R1_EDGE_STRATEGY_ENVIRONMENT_VARIABLE, "serial")
    assert jit.execution_cuda_r1_edge_launch_policy(
        default_strategy="wave",
        default_logical_subgroup_width=32,
        default_edge_threads_per_block=32,
        default_persistent_blocks_per_compute_unit=32,
    ) == {
        "strategy": "serial",
        "logical_subgroup_width": 1,
        "edge_threads_per_block": 256,
        "persistent_blocks_per_compute_unit": 1,
    }
    monkeypatch.delenv(jit.CUDA_R1_EDGE_STRATEGY_ENVIRONMENT_VARIABLE)
    monkeypatch.setenv(jit.CUDA_R1_EDGE_STRATEGY_ENVIRONMENT_VARIABLE, "wave")
    monkeypatch.setenv(jit.CUDA_R1_EDGE_LOGICAL_WIDTH_ENVIRONMENT_VARIABLE, "4")
    monkeypatch.setenv(jit.CUDA_R1_EDGE_THREADS_ENVIRONMENT_VARIABLE, "128")
    monkeypatch.setenv(jit.CUDA_R1_EDGE_BLOCKS_PER_SM_ENVIRONMENT_VARIABLE, "4")
    assert jit.execution_cuda_r1_edge_launch_policy() == {
        "strategy": "wave",
        "logical_subgroup_width": 4,
        "edge_threads_per_block": 128,
        "persistent_blocks_per_compute_unit": 4,
    }

    monkeypatch.setenv(jit.CUDA_R1_EDGE_LOGICAL_WIDTH_ENVIRONMENT_VARIABLE, "3")
    with pytest.raises(ValueError, match="power of two"):
        jit.execution_cuda_r1_edge_launch_policy()
    monkeypatch.setenv(jit.CUDA_R1_EDGE_LOGICAL_WIDTH_ENVIRONMENT_VARIABLE, "4")
    monkeypatch.setenv(jit.CUDA_R1_EDGE_THREADS_ENVIRONMENT_VARIABLE, "96")
    with pytest.raises(ValueError, match="power of two"):
        jit.execution_cuda_r1_edge_launch_policy()
    monkeypatch.setenv(jit.CUDA_R1_EDGE_THREADS_ENVIRONMENT_VARIABLE, "128")
    monkeypatch.setenv(jit.CUDA_R1_EDGE_BLOCKS_PER_SM_ENVIRONMENT_VARIABLE, "0")
    with pytest.raises(ValueError, match=r"\[1, 32\]"):
        jit.execution_cuda_r1_edge_launch_policy()


def _source(value=42):
    return f'extern "C" int symmetrix_jit_probe() {{ return {value}; }}\n'


def _prepare(jit, tmp_path, cxx, source=None, **kwargs):
    return jit.prepare_jit_artifact(
        _source() if source is None else source,
        abi={"symmetrix": "test-abi-v1"},
        build={"backend": "standalone-test", "precision": "float32"},
        cpu={"machine": "test-cpu", "features": ["sse2"]},
        cache_root=tmp_path / "cache",
        cxx=cxx,
        **kwargs,
    )


def _fake_nvcc(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    compiler = tmp_path / "nvcc"
    compiler.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "import sys\n"
        "log = os.environ.get('FAKE_NVCC_LOG')\n"
        "if log:\n"
        "    with Path(log).open('a', encoding='utf-8') as stream:\n"
        "        stream.write(json.dumps({\n"
        "            'argv': sys.argv[1:],\n"
        "            'prepend': os.environ.get('NVCC_PREPEND_FLAGS'),\n"
        "            'append': os.environ.get('NVCC_APPEND_FLAGS'),\n"
        "        }) + '\\n')\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('Cuda compilation tools, release 13.3, V13.3.73')\n"
        "    raise SystemExit(0)\n"
        "if os.environ.get('FAKE_NVCC_FAIL') == '1':\n"
        "    print('synthetic nvcc compile failure', file=sys.stderr)\n"
        "    raise SystemExit(7)\n"
        "output = Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "output.write_bytes(b'fake CUDA shared library')\n",
        encoding="utf-8",
    )
    compiler.chmod(0o755)
    return compiler


def _prepare_cuda(jit, tmp_path, nvcc, **kwargs):
    compute_capability = kwargs.pop(
        "compute_capability",
        {
            "major": 12,
            "minor": 0,
            "compute_capability": "12.0",
            "multiprocessor_count": 108,
            "runtime_version": 13030,
            "driver_version": 13030,
        },
    )
    return jit.prepare_execution_cuda_jit_artifact(
        'extern "C" __global__ void kernel() {}\n',
        abi={"tag": "symmetrix.jit.cuda-plugin/1", "version": 1},
        build={"generator": "cuda-test-v1", "contract": "contract-a"},
        compute_capability=compute_capability,
        cache_root=tmp_path / "cuda-cache",
        nvcc=nvcc,
        **kwargs,
    )


def _prepare_nvrtc(jit, tmp_path, compile_source, **kwargs):
    return jit.prepare_nvrtc_jit_artifact(
        'extern "C" __global__ void kernel() {}\n',
        abi={"tag": "symmetrix.jit.cuda-module/1", "version": 1},
        build={"generator": "nvrtc-test-v1", "contract": "contract-a"},
        compute_capability={
            "major": 12,
            "minor": 0,
            "compute_capability": "12.0",
            "runtime_version": 13030,
        },
        cache_root=tmp_path / "nvrtc-cache",
        nvrtc_information={
            "available": True,
            "library": "libnvrtc.so.13",
            "major": 13,
            "minor": 3,
            "supported_architectures": [80, 86, 89, 90, 100, 103, 120],
        },
        compile_source=compile_source,
        **kwargs,
    )


def test_nvrtc_build_publish_validate_and_reuse(jit, tmp_path):
    calls = []

    def compile_source(source, options):
        calls.append((source, tuple(options)))
        return b"synthetic exact-sm cubin", "synthetic NVRTC log"

    built = _prepare_nvrtc(jit, tmp_path, compile_source)

    assert built.status == "built"
    assert built.artifact_path.suffix == ".cubin"
    assert {path.name for path in built.artifact_path.parent.iterdir()} == {
        "manifest.json",
        "module.cu",
        "factorized_cuda_module_gen11.cubin",
    }
    manifest = jit.validate_jit_manifest(
        built.artifact_path.parent, expected_key=built.cache_key
    )
    assert manifest["source"] == "module.cu"
    assert manifest["key_inputs"]["compiler"] == {
        "kind": "nvrtc",
        "version": [13, 3],
        "library": "libnvrtc.so.13",
    }
    assert manifest["key_inputs"]["build"] == {
        "schema": jit.CUDA_MODULE_BUILD_SCHEMA,
        "version": jit.CUDA_MODULE_BUILD_VERSION,
        "backend": "nvrtc",
        "translation_unit": "module.cu",
        "compute_capability": [12, 0],
        "architecture": "sm_120",
        "request": {"generator": "nvrtc-test-v1", "contract": "contract-a"},
    }
    assert manifest["compiler_options"] == [
        "--std=c++20",
        "--gpu-architecture=sm_120",
    ]
    assert len(calls) == 1
    assert any("synthetic NVRTC log" in value for value in built.diagnostics)

    cached = _prepare_nvrtc(jit, tmp_path, compile_source)
    assert cached.status == "cached"
    assert cached.cache_key == built.cache_key
    assert cached.artifact_path == built.artifact_path
    assert len(calls) == 1


def test_nvrtc_rejects_unsupported_exact_architecture(jit, tmp_path):
    result = jit.prepare_nvrtc_jit_artifact(
        'extern "C" __global__ void kernel() {}\n',
        abi={"tag": "test", "version": 1},
        build={"generator": "test"},
        compute_capability="sm_120",
        cache_root=tmp_path / "cache",
        nvrtc_information={
            "available": True,
            "library": "libnvrtc.so.12",
            "major": 12,
            "minor": 8,
            "supported_architectures": [80, 86, 89, 90],
        },
        compile_source=lambda source, options: (b"unexpected", ""),
    )
    assert result.status == "fallback"
    assert "does not support sm_120" in result.reason


def test_cuda_jit_backend_selection_prefers_nvrtc_and_preserves_forced_nvcc(
    jit, monkeypatch
):
    monkeypatch.setattr(
        jit,
        "jit_nvrtc_information",
        lambda: {"available": True, "major": 13, "minor": 3},
    )
    assert jit.select_execution_cuda_jit_backend("automatic") == "nvrtc"
    assert jit.select_execution_cuda_jit_backend("nvrtc") == "nvrtc"
    with pytest.warns(FutureWarning, match="runtime nvcc"):
        assert jit.select_execution_cuda_jit_backend("nvcc") == "nvcc"


def test_cuda_jit_backend_selection_never_falls_back_to_nvcc(jit, monkeypatch):
    monkeypatch.setattr(
        jit,
        "jit_nvrtc_information",
        lambda: {"available": False, "reason": "runtime library absent"},
    )
    with pytest.raises(jit.JitError, match="no AOT compiler fallback"):
        jit.select_execution_cuda_jit_backend("automatic")
    with pytest.raises(jit.JitError, match="runtime library absent"):
        jit.select_execution_cuda_jit_backend("nvrtc")


def test_compiler_registry_declares_no_automatic_aot_fallbacks(jit):
    assert jit.execution_compiler_registry() == {
        "nvcc": {
            "backend": "cuda",
            "fallback": None,
            "source_kind": "cuda",
            "artifact_kind": "shared-library",
        },
        "nvrtc": {
            "backend": "cuda",
            "fallback": None,
            "source_kind": "cuda",
            "artifact_kind": "cubin",
        },
        "hipcc": {
            "backend": "hip",
            "fallback": None,
            "source_kind": "hip",
            "artifact_kind": "shared-library",
        },
        "hiprtc": {
            "backend": "hip",
            "fallback": None,
            "source_kind": "hip",
            "artifact_kind": "code-object",
        },
    }


def test_generic_nvrtc_request_uses_native_option_mapping(jit, tmp_path):
    calls = []

    def compile_source(source, options):
        calls.append((source, tuple(options)))
        return b"generic cubin", ""

    request = jit.CompilerRequest(
        kind="nvrtc",
        build={"generator": "registry-test-v1"},
        target={"compute_capability": "sm_120"},
        source_kind="cuda",
        artifact_kind="cubin",
        information={
            "available": True,
            "library": "libnvrtc.so.13",
            "major": 13,
            "minor": 1,
            "supported_architectures": [120],
        },
        compile_source=compile_source,
    )
    result = jit.prepare_execution_compiler_artifact(
        'extern "C" __global__ void kernel() {}\n',
        abi={"tag": "registry-test", "version": 1},
        request=request,
        cache_root=tmp_path / "registry-cache",
    )

    assert result.status == "built"
    assert calls[0][1] == ("--std=c++20", "--gpu-architecture=sm_120")
    assert "-O3" not in calls[0][1]


def test_generic_compiler_request_rejects_cross_backend_artifact(jit, tmp_path):
    request = jit.CompilerRequest(
        kind="nvrtc",
        build={},
        target={"compute_capability": "sm_120"},
        source_kind="cuda",
        artifact_kind="code-object",
    )
    with pytest.raises(ValueError, match="artifact kind must be cubin"):
        jit.prepare_execution_compiler_artifact(
            "source",
            abi={},
            request=request,
            cache_root=tmp_path,
        )


def test_packaged_nvrtc_library_is_exposed_to_native_loader(jit, tmp_path, monkeypatch):
    package = tmp_path / "nvidia" / "cuda_nvrtc"
    library = package / "lib" / "libnvrtc.so.13"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"synthetic NVRTC library")
    monkeypatch.delenv(jit.NVRTC_LIBRARY_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(
                origin=str(package / "__init__.py"),
                submodule_search_locations=[str(package)],
            )
            if name == "nvidia.cuda_nvrtc"
            else None
        ),
    )

    assert jit._discover_packaged_nvrtc_library() == str(library.resolve())
    assert os.environ[jit.NVRTC_LIBRARY_ENVIRONMENT_VARIABLE] == str(library.resolve())


def test_cuda13_packaged_nvrtc_library_uses_unified_namespace(
    jit, tmp_path, monkeypatch
):
    package = tmp_path / "nvidia" / "cu13"
    library = package / "lib" / "libnvrtc.so.13"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"synthetic CUDA 13 NVRTC library")
    monkeypatch.delenv(jit.NVRTC_LIBRARY_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(
                origin=str(package / "__init__.py"),
                submodule_search_locations=[str(package)],
            )
            if name == "nvidia.cu13"
            else None
        ),
    )

    assert jit._discover_packaged_nvrtc_library("cuda13") == str(library.resolve())
    assert os.environ[jit.NVRTC_LIBRARY_ENVIRONMENT_VARIABLE] == str(library.resolve())


def test_cache_key_is_canonical_and_covers_all_inputs(jit):
    common = {
        "source": _source(),
        "abi": {"z": 2, "a": 1},
        "build": {"backend": "host"},
        "compiler": {"version": "test", "command": ["c++"]},
        "cpu": {"flags": ["avx2", "sse2"]},
    }
    first = jit.jit_cache_key(**common)
    reordered = jit.jit_cache_key(**{**common, "abi": {"a": 1, "z": 2}})
    assert first == reordered
    assert first == ("18053499c74e0e7f9a481bb581ca2bcd8b63be62631928a60c1968ee89d6a8ef")
    assert len(first) == 64

    variants = [
        {"source": _source(43)},
        {"abi": {"a": 1, "z": 3}},
        {"build": {"backend": "cuda"}},
        {"compiler": {"version": "other", "command": ["c++"]}},
        {"cpu": {"flags": ["sse2"]}},
        {"cxx_flags": ("-DNDEBUG",)},
        {"artifact_name": "other_plugin"},
    ]
    for update in variants:
        assert jit.jit_cache_key(**{**common, **update}) != first


def test_gpu_codegen_backend_schema_and_schedule_do_not_collide(jit):
    common = {
        "source": _source(),
        "abi": {"tag": "device-v1"},
        "compiler": {"kind": "synthetic", "version": 1},
        "cpu": None,
    }
    identity = {
        "schema_version": 1,
        "program": {"schema_version": 1},
        "target": {"schema_version": 1, "backend": "cuda"},
        "schedule": {"schema_version": 1, "edge_strategy": "serial"},
        "launch_plan": {"schema_version": 1},
    }

    def cache_key(gpu_codegen):
        return jit.jit_cache_key(
            **common,
            build={"generator": "r1-device-v1", "gpu_codegen": gpu_codegen},
        )

    baseline = cache_key(identity)
    variants = []
    for section, field, value in (
        (None, "schema_version", 2),
        ("target", "backend", "hip"),
        ("schedule", "edge_strategy", "wave"),
    ):
        changed = json.loads(json.dumps(identity))
        destination = changed if section is None else changed[section]
        destination[field] = value
        variants.append(cache_key(changed))
    assert len({baseline, *variants}) == 4


def test_cache_root_precedence(jit, monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    environment = tmp_path / "environment"
    xdg = tmp_path / "xdg"
    monkeypatch.setenv(jit.CACHE_ENVIRONMENT_VARIABLE, str(environment))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    assert jit.jit_cache_root(explicit) == explicit
    assert jit.jit_cache_root() == environment
    monkeypatch.delenv(jit.CACHE_ENVIRONMENT_VARIABLE)
    assert jit.jit_cache_root() == xdg / "symmetrix" / "execution-jit"


def test_host_jit_default_policy_prefers_accepted_native_target(jit, monkeypatch):
    probes = []

    def probe(command, flags, **kwargs):
        probes.append((tuple(command), tuple(flags), kwargs))
        return True, ""

    monkeypatch.delenv(jit.HOST_TARGET_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.delenv(jit.HOST_FLAGS_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.delenv(jit.HOST_FP_MODE_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(jit, "_probe_host_compiler_flags", probe)

    policy = jit.resolve_host_jit_policy(("/usr/bin/c++",))

    assert policy.requested == "automatic"
    assert policy.selected == "native"
    assert policy.flags == ("-ffast-math", "-march=native")
    assert probes == [(("/usr/bin/c++",), ("-ffast-math", "-march=native"), {})]
    assert "automatic -> native" in policy.diagnostics[0]


def test_host_jit_ieee_mode_disables_fast_math_and_contraction(jit, monkeypatch):
    probes = []
    monkeypatch.setenv(jit.HOST_FP_MODE_ENVIRONMENT_VARIABLE, "ieee")
    monkeypatch.setattr(
        jit,
        "_probe_host_compiler_flags",
        lambda command, flags: (probes.append(tuple(flags)) or True, ""),
    )

    policy = jit.resolve_host_jit_policy(("c++",), target="native")

    assert policy.flags == ("-fno-fast-math", "-ffp-contract=off", "-march=native")
    assert probes == [policy.flags]
    assert any("floating-point mode: ieee" in item for item in policy.diagnostics)


def test_host_jit_rejects_unknown_fp_mode(jit, monkeypatch):
    monkeypatch.setenv(jit.HOST_FP_MODE_ENVIRONMENT_VARIABLE, "approximate")

    with pytest.raises(jit.JitError, match="must be 'fast' or 'ieee'"):
        jit.resolve_host_jit_policy(("c++",), target="native")


def _x86_64_v3_identity(jit, *, without=()):
    features = {
        next(iter(aliases)) for aliases in jit._X86_64_V3_FEATURES.values()
    } - set(without)
    return {"machine": "x86_64", "cpuinfo": {"flags": sorted(features)}}


def test_host_jit_rejected_native_target_selects_x86_64_v3(jit, monkeypatch):
    probes = []

    def probe(command, flags):
        probes.append(tuple(flags))
        if "-march=native" in flags:
            return False, "unsupported CPU target"
        return True, ""

    monkeypatch.setattr(jit, "_probe_host_compiler_flags", probe)
    monkeypatch.setattr(jit, "detect_cpu_identity", lambda: _x86_64_v3_identity(jit))

    policy = jit.resolve_host_jit_policy(("c++",), target="automatic")

    assert policy.selected == "portable"
    assert policy.flags == ("-ffast-math", "-march=x86-64-v3")
    assert probes == [
        ("-ffast-math", "-march=native"),
        ("-ffast-math", "-march=x86-64-v3"),
    ]
    assert any("unsupported CPU target" in item for item in policy.diagnostics)
    assert any("x86-64-v3" in item for item in policy.diagnostics)


def test_host_jit_portable_policy_requires_x86_64_v3(jit, monkeypatch):
    probes = []
    monkeypatch.setattr(jit, "detect_cpu_identity", lambda: _x86_64_v3_identity(jit))
    monkeypatch.setattr(
        jit,
        "_probe_host_compiler_flags",
        lambda command, flags: (probes.append(tuple(flags)) or True, ""),
    )

    policy = jit.resolve_host_jit_policy(("c++",), target="portable")

    assert policy.selected == "portable"
    assert policy.flags == ("-ffast-math", "-march=x86-64-v3")
    assert probes == [("-ffast-math", "-march=x86-64-v3")]


def test_host_jit_portable_policy_rejects_missing_cpu_feature(jit, monkeypatch):
    monkeypatch.setattr(
        jit,
        "detect_cpu_identity",
        lambda: _x86_64_v3_identity(jit, without=("avx2",)),
    )
    monkeypatch.setattr(
        jit,
        "_probe_host_compiler_flags",
        lambda *args, **kwargs: pytest.fail("CPU rejection must precede probing"),
    )

    with pytest.raises(jit.JitError, match="missing CPU features: avx2"):
        jit.resolve_host_jit_policy(("c++",), target="portable")


def test_host_jit_automatic_policy_fails_when_both_targets_are_rejected(
    jit, monkeypatch
):
    monkeypatch.setattr(jit, "detect_cpu_identity", lambda: _x86_64_v3_identity(jit))
    monkeypatch.setattr(
        jit,
        "_probe_host_compiler_flags",
        lambda command, flags: (False, f"rejected {flags[-1]}"),
    )

    with pytest.raises(jit.JitError, match="rejected both.*x86-64-v3"):
        jit.resolve_host_jit_policy(("c++",), target="automatic")


def test_host_jit_explicit_target_and_flags_are_safe_argv(jit, monkeypatch):
    probes = []
    monkeypatch.setattr(
        jit,
        "_probe_host_compiler_flags",
        lambda command, flags: (probes.append(tuple(flags)) or True, ""),
    )
    target_policy = jit.resolve_host_jit_policy(("c++",), target="x86-64")
    assert target_policy.selected == "target:x86-64"
    assert target_policy.flags == ("-ffast-math", "-march=x86-64")
    assert probes == [("-ffast-math", "-march=x86-64")]

    monkeypatch.setattr(
        jit.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("explicit flags must not invoke a shell"),
    )
    explicit = jit.resolve_host_jit_policy(
        ("c++",), flags='-march=x86-64 -mavx2 -DNAME="two words"'
    )
    assert explicit.selected == "explicit-flags"
    assert explicit.flags == (
        "-march=x86-64",
        "-mavx2",
        "-DNAME=two words",
    )


def test_host_jit_explicit_native_target_fails_closed(jit, monkeypatch):
    monkeypatch.setattr(
        jit,
        "_probe_host_compiler_flags",
        lambda command, flags: (False, "native target rejected"),
    )
    with pytest.raises(jit.JitError, match="native target probe failed"):
        jit.resolve_host_jit_policy(("c++",), target="native")


def test_host_jit_effective_flags_and_policy_are_cached(jit, tmp_path, cxx):
    common = {
        "abi": {"symmetrix": "host-policy-test-v1"},
        "build": {"backend": "host-policy-test", "precision": "float64"},
        "cpu": {"machine": "test-cpu"},
        "cache_root": tmp_path / "host-policy-cache",
        "cxx": cxx,
        "cxx_flags": None,
    }
    first = jit.prepare_jit_artifact(
        _source(), host_flags=("-ffast-math", "-DHOST_POLICY=1"), **common
    )
    second = jit.prepare_jit_artifact(
        _source(), host_flags=("-ffast-math", "-DHOST_POLICY=2"), **common
    )

    assert first.status == second.status == "built"
    assert first.cache_key != second.cache_key
    assert first.manifest["key_inputs"]["cxx_flags"] == [
        "-ffast-math",
        "-DHOST_POLICY=1",
    ]
    assert first.manifest["key_inputs"]["build"]["host_jit_policy"] == {
        "schema": "symmetrix.jit.host-policy/1",
        "requested": "automatic",
        "selected": "explicit-flags",
        "flags": ["-ffast-math", "-DHOST_POLICY=1"],
    }
    assert any("host JIT policy: explicit-flags" in item for item in first.diagnostics)
    assert any("-DHOST_POLICY=1" in item for item in first.diagnostics)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ((8, 9), (8, 9)),
        ([12, 0], (12, 0)),
        ("12.0", (12, 0)),
        ("sm_120", (12, 0)),
        ("compute_89", (8, 9)),
        ({"major": 9, "minor": 0}, (9, 0)),
        ({"compute_capability": "12.0"}, (12, 0)),
        ({"compute_capability_code": 120}, (12, 0)),
        (
            {"compute_capability": "12.0", "compute_capability_code": 120},
            (12, 0),
        ),
    ],
)
def test_cuda_compute_capability_normalization(jit, value, expected):
    assert jit._normalize_compute_capability(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        None,
        "120",
        "sm_9",
        (12,),
        (0, 0),
        (12, 10),
        (True, 0),
        {},
        {"major": 12, "minor": 0, "compute_capability": "8.9"},
        {"compute_capability": "8.9", "compute_capability_code": 120},
        {"compute_capability_code": True},
    ],
)
def test_cuda_compute_capability_rejects_ambiguous_or_invalid_values(jit, value):
    with pytest.raises(ValueError):
        jit._normalize_compute_capability(value)


def test_cuda_compiler_discovery_precedence(jit, monkeypatch, tmp_path):
    explicit = _fake_nvcc(tmp_path / "explicit")
    environment = _fake_nvcc(tmp_path / "environment")
    cudacxx = _fake_nvcc(tmp_path / "cudacxx")
    monkeypatch.setenv(jit.CUDA_COMPILER_ENVIRONMENT_VARIABLE, str(environment))
    monkeypatch.setenv("CUDACXX", str(cudacxx))

    assert jit._discover_nvcc_command(explicit) == (str(explicit),)
    assert jit._discover_nvcc_command(None) == (str(environment),)
    monkeypatch.delenv(jit.CUDA_COMPILER_ENVIRONMENT_VARIABLE)
    assert jit._discover_nvcc_command(None) == (str(cudacxx),)


def test_cuda_build_publish_validate_and_reuse(jit, monkeypatch, tmp_path):
    compiler = _fake_nvcc(tmp_path)
    log_path = tmp_path / "nvcc.jsonl"
    monkeypatch.setenv("FAKE_NVCC_LOG", str(log_path))
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "--use_fast_math")
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "-arch=sm_80")

    built = _prepare_cuda(jit, tmp_path, compiler)

    assert built.status == "built"
    assert built.available and not built.cache_hit
    assert built.artifact_path.is_file()
    assert {path.name for path in built.artifact_path.parent.iterdir()} == {
        "manifest.json",
        "plugin.cu",
        "factorized_cuda_plugin_gen11.so",
    }
    manifest = jit.validate_jit_manifest(
        built.artifact_path.parent, expected_key=built.cache_key
    )
    assert manifest == built.manifest
    assert manifest["source"] == "plugin.cu"
    assert manifest["key_inputs"]["compiler"]["kind"] == "nvcc"
    assert manifest["key_inputs"]["build"] == {
        "schema": jit.CUDA_BUILD_SCHEMA,
        "version": jit.CUDA_BUILD_VERSION,
        "backend": "cuda",
        "translation_unit": "plugin.cu",
        "compute_capability": [12, 0],
        "architecture": "sm_120",
        "request": {"generator": "cuda-test-v1", "contract": "contract-a"},
    }
    target = manifest["key_inputs"]["cpu"]
    assert target["backend"] == "cuda"
    assert target["compute_capability"] == [12, 0]
    assert target["architecture"] == "sm_120"
    assert target["runtime"] == {"runtime_version": 13030}
    options = manifest["compiler_options"]
    assert "-Xcompiler=-fPIC" in options
    assert "-fPIC" not in options
    assert "-gencode=arch=compute_120,code=sm_120" in options
    assert manifest["key_inputs"]["cxx_flags"] == options
    assert any(
        "ignored ambient NVCC_PREPEND_FLAGS" in item for item in built.diagnostics
    )
    assert any(
        "ignored ambient NVCC_APPEND_FLAGS" in item for item in built.diagnostics
    )

    invocations = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(invocations) == 2
    assert invocations[0]["argv"] == ["--version"]
    assert Path(invocations[1]["argv"][-3]).name == "plugin.cu"
    assert invocations[1]["argv"][-2] == "-o"
    assert Path(invocations[1]["argv"][-1]).name == "factorized_cuda_plugin_gen11.so"
    assert all(call["prepend"] is None for call in invocations)
    assert all(call["append"] is None for call in invocations)

    cached = _prepare_cuda(jit, tmp_path, compiler)
    assert cached.status == "cached"
    assert cached.cache_key == built.cache_key
    assert cached.artifact_path == built.artifact_path
    invocations = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(invocations) == 3
    assert invocations[-1]["argv"] == ["--version"]

    same_binary_target = _prepare_cuda(
        jit,
        tmp_path,
        compiler,
        compute_capability={
            "major": 12,
            "minor": 0,
            "compute_capability": "12.0",
            "device_name": "another GPU",
            "device_ordinal": 3,
            "multiprocessor_count": 999,
            "warp_width": 32,
            "runtime_version": 13030,
            "driver_version": 99999,
        },
    )
    assert same_binary_target.status == "cached"
    assert same_binary_target.cache_key == built.cache_key


def test_cuda_cache_identity_covers_target_contract_and_options(jit, tmp_path):
    compiler = _fake_nvcc(tmp_path)
    first = _prepare_cuda(jit, tmp_path, compiler)
    architecture = jit.prepare_execution_cuda_jit_artifact(
        'extern "C" __global__ void kernel() {}\n',
        abi={"tag": "symmetrix.jit.cuda-plugin/1", "version": 1},
        build={"generator": "cuda-test-v1", "contract": "contract-a"},
        compute_capability="8.9",
        cache_root=tmp_path / "cuda-cache",
        nvcc=compiler,
    )
    contract = jit.prepare_execution_cuda_jit_artifact(
        'extern "C" __global__ void kernel() {}\n',
        abi={"tag": "symmetrix.jit.cuda-plugin/1", "version": 1},
        build={"generator": "cuda-test-v1", "contract": "contract-b"},
        compute_capability="12.0",
        cache_root=tmp_path / "cuda-cache",
        nvcc=compiler,
    )
    options = _prepare_cuda(jit, tmp_path, compiler, nvcc_flags=("-lineinfo",))
    source = jit.prepare_execution_cuda_jit_artifact(
        'extern "C" __global__ void changed_kernel() {}\n',
        abi={"tag": "symmetrix.jit.cuda-plugin/1", "version": 1},
        build={"generator": "cuda-test-v1", "contract": "contract-a"},
        compute_capability="12.0",
        cache_root=tmp_path / "cuda-cache",
        nvcc=compiler,
    )
    other_compiler = _fake_nvcc(tmp_path / "other-compiler")
    compiler_identity = _prepare_cuda(jit, tmp_path, other_compiler)

    results = (
        first,
        architecture,
        contract,
        options,
        source,
        compiler_identity,
    )
    assert all(result.available for result in results)
    assert len({result.cache_key for result in results}) == len(results)


def test_cuda_missing_or_wrapped_compiler_returns_clear_fallback(jit, tmp_path):
    missing = _prepare_cuda(jit, tmp_path, tmp_path / "missing-nvcc")
    assert missing.status == "fallback"
    assert "NVCC compiler executable was not found" in missing.reason

    wrapper = tmp_path / "nvcc_wrapper"
    wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    wrapper.chmod(0o755)
    wrapped = _prepare_cuda(jit, tmp_path, wrapper)
    assert wrapped.status == "fallback"
    assert "nvcc_wrapper is not a supported" in wrapped.reason


def test_cuda_compile_failure_and_reserved_flags_are_policy_results(
    jit, monkeypatch, tmp_path
):
    compiler = _fake_nvcc(tmp_path)
    monkeypatch.setenv("FAKE_NVCC_FAIL", "1")
    failed = _prepare_cuda(jit, tmp_path, compiler)
    assert failed.status == "fallback"
    assert "NVCC compilation exited with status 7" in failed.reason
    assert any("synthetic nvcc compile failure" in item for item in failed.diagnostics)
    assert list((tmp_path / "cuda-cache" / ".staging").iterdir()) == []

    monkeypatch.delenv("FAKE_NVCC_FAIL")
    override = _prepare_cuda(jit, tmp_path, compiler, nvcc_flags=("-arch=sm_80",))
    assert override.status == "fallback"
    assert "must not override the CUDA architecture" in override.reason


@pytest.mark.skipif(os.name != "posix", reason="POSIX cache permission boundary")
def test_cache_rejects_group_writable_root(jit, tmp_path, cxx):
    cache = tmp_path / "shared-cache"
    cache.mkdir(mode=0o700)
    cache.chmod(0o770)

    result = jit.prepare_jit_artifact(
        _source(),
        abi="test-abi",
        build="permission-test",
        cpu="test-cpu",
        cache_root=cache,
        cxx=cxx,
    )

    assert not result.available
    assert "private" in result.reason
    assert not (cache / "artifacts").exists()


def test_build_publish_validate_and_reuse(jit, tmp_path, cxx):
    built = _prepare(jit, tmp_path, cxx)
    assert built.status == "built"
    assert built.available and not built.cache_hit
    assert built.artifact_path.is_file()
    assert built.manifest_path.is_file()
    assert built.artifact_path.parent.name == built.cache_key
    assert {path.name for path in built.artifact_path.parent.iterdir()} == {
        "manifest.json",
        "plugin.cpp",
        "execution_plugin_gen11.so",
    }
    manifest = jit.validate_jit_manifest(
        built.artifact_path.parent, expected_key=built.cache_key
    )
    assert manifest == built.manifest
    assert manifest["source"] == "plugin.cpp"
    assert manifest["key_inputs"]["jit_generation_version"] == 11
    assert manifest["compiler_options"] == [
        "-std=c++20",
        "-O3",
        "-fPIC",
        "-shared",
    ]

    library = ctypes.CDLL(str(built.artifact_path))
    probe = library.symmetrix_jit_probe
    probe.restype = ctypes.c_int
    assert probe() == 42

    cached = _prepare(jit, tmp_path, cxx)
    assert cached.status == "cached"
    assert cached.cache_hit
    assert cached.cache_key == built.cache_key
    assert cached.artifact_path == built.artifact_path
    staging = tmp_path / "cache" / ".staging"
    assert list(staging.iterdir()) == []


def test_generation_version_publishes_and_preserves_distinct_entries(
    jit, tmp_path, cxx, monkeypatch
):
    generation_eleven = _prepare(jit, tmp_path, cxx)
    assert generation_eleven.status == "built"
    assert generation_eleven.artifact_path.name == "execution_plugin_gen11.so"

    monkeypatch.setattr(jit, "JIT_GENERATION_VERSION", 12)
    generation_twelve = _prepare(jit, tmp_path, cxx)
    assert generation_twelve.status == "built"
    assert generation_twelve.cache_key != generation_eleven.cache_key
    assert generation_twelve.artifact_path.name == "execution_plugin_gen12.so"
    assert generation_twelve.manifest["key_inputs"]["jit_generation_version"] == 12

    assert generation_eleven.artifact_path.is_file()
    assert generation_eleven.manifest_path.is_file()
    assert {generation_eleven.cache_key, generation_twelve.cache_key} <= {
        path.name for path in (tmp_path / "cache" / "artifacts").iterdir()
    }

    monkeypatch.setattr(jit, "JIT_GENERATION_VERSION", 11)
    cached_generation_eleven = _prepare(jit, tmp_path, cxx)
    assert cached_generation_eleven.status == "cached"
    assert cached_generation_eleven.artifact_path == generation_eleven.artifact_path


def test_changed_source_publishes_a_distinct_entry(jit, tmp_path, cxx):
    first = _prepare(jit, tmp_path, cxx, source=_source(1))
    second = _prepare(jit, tmp_path, cxx, source=_source(2))
    assert first.available and second.available
    assert first.cache_key != second.cache_key
    assert first.artifact_path != second.artifact_path


def test_invalid_cached_manifest_is_quarantined_and_rebuilt(jit, tmp_path, cxx):
    first = _prepare(jit, tmp_path, cxx)
    first.manifest_path.write_text("{not json", encoding="utf-8")

    rebuilt = _prepare(jit, tmp_path, cxx)
    assert rebuilt.status == "built"
    assert rebuilt.cache_key == first.cache_key
    assert any("invalid cached JIT entry" in item for item in rebuilt.diagnostics)
    assert any(
        "quarantined invalid JIT cache entry" in item for item in rebuilt.diagnostics
    )
    jit.validate_jit_manifest(rebuilt.artifact_path.parent)
    invalid_root = tmp_path / "cache" / ".invalid"
    assert list(invalid_root.iterdir()) == []


def test_load_broken_artifact_is_quarantined_before_rebuild(jit, tmp_path, cxx):
    first = _prepare(jit, tmp_path, cxx)
    quarantine = jit.quarantine_jit_artifact(first, "descriptor validation failed")
    assert not first.artifact_path.parent.exists()
    assert any(
        "descriptor validation failed" in item for item in quarantine.diagnostics
    )
    assert any(
        "quarantined invalid JIT cache entry" in item for item in quarantine.diagnostics
    )
    invalid_entries = list((tmp_path / "cache" / ".invalid").iterdir())
    assert len(invalid_entries) == 1
    assert quarantine.path == invalid_entries[0]

    rebuilt = _prepare(jit, tmp_path, cxx)
    assert rebuilt.status == "built"
    assert rebuilt.cache_key == first.cache_key
    jit.validate_jit_manifest(rebuilt.artifact_path.parent)
    diagnostics = jit.remove_jit_quarantine(quarantine)
    assert any("removed recovered JIT quarantine" in item for item in diagnostics)
    assert list((tmp_path / "cache" / ".invalid").iterdir()) == []


def test_repeated_load_quarantine_retains_only_latest_entry(jit, tmp_path, cxx):
    first = _prepare(jit, tmp_path, cxx)
    first_quarantine = jit.quarantine_jit_artifact(first, "first failure")
    rebuilt = _prepare(jit, tmp_path, cxx)
    second_quarantine = jit.quarantine_jit_artifact(rebuilt, "second failure")

    assert not first_quarantine.path.exists()
    assert second_quarantine.path.exists()
    invalid_entries = list((tmp_path / "cache" / ".invalid").iterdir())
    assert invalid_entries == [second_quarantine.path]


def test_stale_load_result_does_not_quarantine_new_publication(jit, tmp_path, cxx):
    first = _prepare(jit, tmp_path, cxx)
    first_quarantine = jit.quarantine_jit_artifact(first, "first failure")
    rebuilt = _prepare(jit, tmp_path, cxx)
    assert rebuilt.manifest["publication_id"] != first.manifest["publication_id"]

    stale_quarantine = jit.quarantine_jit_artifact(first, "stale first failure")

    assert stale_quarantine.path is None
    assert rebuilt.artifact_path.is_file()
    jit.validate_jit_manifest(rebuilt.artifact_path.parent)
    assert any("newer publication" in item for item in stale_quarantine.diagnostics)
    jit.remove_jit_quarantine(first_quarantine)


def test_manifest_rejects_path_traversal(jit, tmp_path, cxx):
    result = _prepare(jit, tmp_path, cxx)
    manifest = json.loads(result.manifest_path.read_text())
    manifest["artifact"] = "../outside.so"
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(jit.JitManifestError, match="not a basename"):
        jit.validate_jit_manifest(result.artifact_path.parent)


def test_manifest_binds_artifact_name_to_cache_key(jit, tmp_path, cxx):
    result = _prepare(jit, tmp_path, cxx)
    replacement = result.artifact_path.with_name("replacement.so")
    shutil.copy2(result.artifact_path, replacement)
    manifest = json.loads(result.manifest_path.read_text())
    manifest["artifact"] = replacement.name
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(jit.JitManifestError, match="cache-key inputs"):
        jit.validate_jit_manifest(result.artifact_path.parent)


def test_manifest_rejects_symlinked_cache_entry(jit, tmp_path, cxx):
    result = _prepare(jit, tmp_path, cxx)
    alias = tmp_path / "cache-entry-alias"
    alias.symlink_to(result.artifact_path.parent, target_is_directory=True)
    with pytest.raises(jit.JitManifestError, match="regular directory"):
        jit.validate_jit_manifest(alias)


def test_compile_failure_returns_diagnostics_and_cleans_staging(jit, tmp_path, cxx):
    failed = _prepare(jit, tmp_path, cxx, source="this is not valid C++\n")
    assert failed.status == "fallback"
    assert not failed.available
    assert failed.artifact_path is None
    assert "compilation exited" in failed.reason
    assert any("compiler stderr" in item for item in failed.diagnostics)
    assert failed.command
    staging = tmp_path / "cache" / ".staging"
    assert list(staging.iterdir()) == []


@pytest.mark.skipif(sys.platform != "linux", reason="Linux renameat2 coverage")
def test_no_replace_publication_preserves_existing_directory(jit, tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "value").write_text("source", encoding="utf-8")
    (destination / "value").write_text("winner", encoding="utf-8")

    assert not jit._publish_directory_no_replace(source, destination)
    assert (source / "value").read_text(encoding="utf-8") == "source"
    assert (destination / "value").read_text(encoding="utf-8") == "winner"


class _FakeCFunction:
    def __init__(self, result, error=0):
        self.result = result
        self.error = error
        self.calls = []
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.calls.append(args)
        ctypes.set_errno(self.error)
        return self.result


def test_linux_renameat2_prefers_exported_libc_symbol(jit, tmp_path):
    renameat2 = _FakeCFunction(0)
    syscall = _FakeCFunction(-1, errno.ENOSYS)
    library = SimpleNamespace(renameat2=renameat2, syscall=syscall)

    result = jit._linux_renameat2(library, tmp_path / "source", tmp_path / "dest")

    assert result == 0
    assert len(renameat2.calls) == 1
    assert syscall.calls == []


@pytest.mark.parametrize("use_syscall", [False, True])
def test_no_replace_publication_returns_false_for_eexist(
    jit, monkeypatch, tmp_path, use_syscall
):
    function = _FakeCFunction(-1, errno.EEXIST)
    library = (
        SimpleNamespace(renameat2=None, syscall=function)
        if use_syscall
        else SimpleNamespace(renameat2=function)
    )
    monkeypatch.setattr(jit.ctypes, "CDLL", lambda *args, **kwargs: library)
    monkeypatch.setattr(jit.platform, "machine", lambda: "x86_64")

    assert not jit._publish_directory_no_replace(
        tmp_path / "source", tmp_path / "destination"
    )
    assert len(function.calls) == 1
    if use_syscall:
        assert function.calls[0][0] == 316


def test_direct_renameat2_syscall_rejects_unsupported_architecture(
    jit, monkeypatch, tmp_path
):
    library = SimpleNamespace(renameat2=None, syscall=_FakeCFunction(0))
    monkeypatch.setattr(jit.ctypes, "CDLL", lambda *args, **kwargs: library)
    monkeypatch.setattr(jit.platform, "machine", lambda: "mystery64")

    with pytest.raises(jit.JitPublicationUnsupported, match="mystery64"):
        jit._publish_directory_no_replace(tmp_path / "source", tmp_path / "destination")


def test_direct_renameat2_syscall_enosys_is_fail_closed(jit, monkeypatch, tmp_path):
    library = SimpleNamespace(
        renameat2=None,
        syscall=_FakeCFunction(-1, errno.ENOSYS),
    )
    monkeypatch.setattr(jit.ctypes, "CDLL", lambda *args, **kwargs: library)
    monkeypatch.setattr(jit.platform, "machine", lambda: "x86_64")

    with pytest.raises(jit.JitPublicationUnsupported, match="does not support"):
        jit._publish_directory_no_replace(tmp_path / "source", tmp_path / "destination")


@pytest.mark.skipif(sys.platform != "linux", reason="Linux renameat2 coverage")
def test_direct_renameat2_syscall_publishes_without_libc_wrapper(
    jit, monkeypatch, tmp_path
):
    actual_library = ctypes.CDLL(None, use_errno=True)

    class SyscallOnlyLibrary:
        renameat2 = None
        syscall = actual_library.syscall

    monkeypatch.setattr(
        jit.ctypes, "CDLL", lambda *args, **kwargs: SyscallOnlyLibrary()
    )
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "value").write_text("published", encoding="utf-8")

    assert jit._publish_directory_no_replace(source, destination)
    assert not source.exists()
    assert (destination / "value").read_text(encoding="utf-8") == "published"


def test_clean_build_and_hit_do_not_acquire_key_lock(jit, monkeypatch, tmp_path):
    calls = []

    def compile_source(source, options):
        calls.append((source, tuple(options)))
        return b"lock-free cubin", ""

    def unexpected_lock(*args, **kwargs):
        raise AssertionError("clean JIT preparation must not acquire the key lock")

    monkeypatch.setattr(jit, "_cache_key_lock", unexpected_lock)
    built = _prepare_nvrtc(jit, tmp_path, compile_source)
    cached = _prepare_nvrtc(jit, tmp_path, compile_source)

    assert built.status == "built"
    assert cached.status == "cached"
    assert built.cache_key == cached.cache_key
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"lock_timeout": -1.0}, "lock_timeout must be nonnegative"),
        ({"stale_lock_age": 0.0}, "stale_lock_age must be positive"),
    ],
)
def test_clean_build_validates_quarantine_options(jit, tmp_path, kwargs, reason):
    compile_calls = 0

    def compile_source(source, options):
        nonlocal compile_calls
        compile_calls += 1
        return b"unexpected cubin", ""

    result = _prepare_nvrtc(jit, tmp_path, compile_source, **kwargs)

    assert result.status == "fallback"
    assert result.reason == reason
    assert compile_calls == 0


def test_unsupported_no_replace_cleans_staging(jit, monkeypatch, tmp_path):
    def unsupported(source, destination):
        raise jit.JitPublicationUnsupported("synthetic unsupported filesystem")

    monkeypatch.setattr(jit, "_publish_directory_no_replace", unsupported)
    result = _prepare_nvrtc(
        jit,
        tmp_path,
        lambda source, options: (b"unpublished cubin", ""),
    )

    assert result.status == "fallback"
    assert "synthetic unsupported filesystem" in result.reason
    assert list((tmp_path / "nvrtc-cache" / "artifacts").iterdir()) == []
    assert list((tmp_path / "nvrtc-cache" / ".staging").iterdir()) == []


@pytest.mark.skipif(sys.platform != "linux", reason="Linux renameat2 coverage")
def test_fresh_processes_converge_on_one_publication(jit, tmp_path):
    module_path = Path(jit.__file__).resolve()
    cache_root = tmp_path / "race-cache"
    gate = tmp_path / "start"
    worker = r"""
import importlib.util
import json
from pathlib import Path
import sys
import time

module_path = Path(sys.argv[1])
cache_root = Path(sys.argv[2])
gate = Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("jit_race_worker", module_path)
jit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = jit
spec.loader.exec_module(jit)
while not gate.exists():
    time.sleep(0.005)

def compile_source(source, options):
    time.sleep(0.2)
    return b"shared race cubin", ""

result = jit.prepare_nvrtc_jit_artifact(
    'extern "C" __global__ void kernel() {}\n',
    abi={"tag": "symmetrix.jit.cuda-module/1", "version": 1},
    build={"generator": "race-test-v1", "contract": "contract-a"},
    compute_capability="sm_120",
    cache_root=cache_root,
    nvrtc_information={
        "available": True,
        "library": "libnvrtc.so.13",
        "major": 13,
        "minor": 3,
        "supported_architectures": [120],
    },
    compile_source=compile_source,
)
print(json.dumps({
    "status": result.status,
    "cache_key": result.cache_key,
    "artifact_path": str(result.artifact_path),
    "publication_id": result.manifest["publication_id"],
    "artifact_sha256": result.manifest["artifact_sha256"],
}))
"""
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                worker,
                str(module_path),
                str(cache_root),
                str(gate),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(8)
    ]
    gate.touch()
    records = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        records.append(json.loads(stdout))

    assert [record["status"] for record in records].count("built") == 1
    assert [record["status"] for record in records].count("cached") == 7
    assert len({record["cache_key"] for record in records}) == 1
    assert len({record["artifact_path"] for record in records}) == 1
    assert len({record["publication_id"] for record in records}) == 1
    assert len({record["artifact_sha256"] for record in records}) == 1
    assert list((cache_root / ".staging").iterdir()) == []
    entries = list((cache_root / "artifacts").iterdir())
    assert len(entries) == 1
    jit.validate_jit_manifest(entries[0])


def test_missing_compiler_returns_fallback_with_reason(jit, tmp_path):
    result = jit.prepare_jit_artifact(
        _source(),
        abi="abi",
        build="build",
        cpu="cpu",
        cache_root=tmp_path,
        cxx=tmp_path / "missing-cxx",
    )
    assert result.status == "fallback"
    assert result.cache_key is None
    assert "was not found" in result.reason
    assert any("JitError" in item for item in result.diagnostics)


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock behavior")
def test_per_key_lock_times_out_with_owner_diagnostic(jit, tmp_path):
    lock_path = tmp_path / "locks" / ("a" * 64 + ".lock")
    first_diagnostics = []
    with (
        jit._cache_key_lock(
            lock_path, timeout=0.1, stale_age=60.0, diagnostics=first_diagnostics
        ),
        pytest.raises(jit.JitLockTimeout, match="timed out") as error,
        jit._cache_key_lock(
            lock_path,
            timeout=0.05,
            stale_age=60.0,
            diagnostics=[],
        ),
    ):
        pass
    assert f"pid={os.getpid()}" in str(error.value)


@pytest.mark.skipif(os.name == "nt", reason="POSIX stale-owner metadata")
def test_per_key_lock_recovers_stale_owner_metadata(jit, tmp_path):
    lock_path = tmp_path / "locks" / ("b" * 64 + ".lock")
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(
        json.dumps(
            {
                "pid": 999_999_999,
                "hostname": socket.gethostname(),
                "created_at": time.time() - 3600,
                "token": "abandoned",
            }
        ),
        encoding="utf-8",
    )
    diagnostics = []
    with jit._cache_key_lock(
        lock_path, timeout=0.1, stale_age=1.0, diagnostics=diagnostics
    ):
        assert any("recovered stale JIT lock owner" in item for item in diagnostics)
    assert lock_path.read_text() == ""
