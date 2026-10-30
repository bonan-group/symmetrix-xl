"""The M1 report must mirror dispatch precedence, not stored owner readiness."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def m1_dispatch_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("M1 binding policy regression requires a C++ compiler")
    root = Path(__file__).resolve().parents[2]
    work = tmp_path_factory.mktemp("m1-binding-dispatch")
    source = work / "probe.cpp"
    source.write_text(
        '#include "mace_m1_dispatch.hpp"\n'
        "#include <iostream>\n"
        "#include <string>\n"
        "int main(int argc, char** argv) {\n"
        "  if (argc == 6) {\n"
        "    const bool standard = symmetrix::execution::m1_standard_forward_available(\n"
        "        argv[2][0] == '1', argv[3][0] == '1', std::stoi(argv[4]),\n"
        "        argv[5][0] == '1');\n"
        "    std::cout << symmetrix::execution::m1_dispatched_implementation_id(\n"
        "        false, true, false, standard) << '\\n';\n"
        "    return 0;\n"
        "  }\n"
        "  if (argc != 5) return 2;\n"
        "  std::cout << symmetrix::execution::m1_dispatched_implementation_id(\n"
        "      argv[1][0] == '1', argv[2][0] == '1',\n"
        "      argv[3][0] == '1', argv[4][0] == '1') << '\\n';\n"
        "}\n"
    )
    binary = work / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "libsymmetrix/source"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.mark.parametrize(
    "artifact,recompute,terms,standard_ready,expected",
    [
        (False, False, False, False, "generic_m1"),
        (False, False, False, True, "generic_m1"),
        (False, False, True, False, "generic_m1"),
        (False, False, True, True, "generic_m1"),
        (False, True, False, False, "generic_m1"),
        (False, True, False, True, "builtin_m1"),
        (False, True, True, False, "device_m1_terms"),
        (False, True, True, True, "device_m1_terms"),
        (True, False, False, False, "generated_m1"),
        (True, False, False, True, "generated_m1"),
        (True, False, True, False, "generated_m1"),
        (True, False, True, True, "generated_m1"),
        (True, True, False, False, "generated_m1"),
        (True, True, False, True, "generated_m1"),
        (True, True, True, False, "generated_m1"),
        (True, True, True, True, "generated_m1"),
    ],
)
def test_m1_report_follows_physical_dispatch_priority(
    m1_dispatch_probe, artifact, recompute, terms, standard_ready, expected
):
    actual = subprocess.check_output(
        [
            str(m1_dispatch_probe),
            *[
                str(int(value))
                for value in (artifact, recompute, terms, standard_ready)
            ],
        ],
        text=True,
    ).strip()
    assert actual == expected


@pytest.mark.parametrize(
    "ready,host,tile,metal,expected",
    [
        (True, True, 0, False, "generic_m1"),
        (False, True, 16, False, "generic_m1"),
        (True, True, 1, False, "builtin_m1"),
        (True, True, 4, False, "builtin_m1"),
        (True, True, 8, False, "builtin_m1"),
        (True, True, 16, False, "builtin_m1"),
        (True, False, 0, False, "builtin_m1"),
        (False, False, 0, False, "generic_m1"),
        (True, True, 0, True, "builtin_m1"),
        (False, True, 0, True, "generic_m1"),
        (True, True, 16, True, "builtin_m1"),
    ],
)
def test_m1_report_accounts_for_standard_host_rollback(
    m1_dispatch_probe, ready, host, tile, metal, expected
):
    actual = subprocess.check_output(
        [
            str(m1_dispatch_probe),
            "standard",
            str(int(ready)),
            str(int(host)),
            str(tile),
            str(int(metal)),
        ],
        text=True,
    ).strip()
    assert actual == expected
