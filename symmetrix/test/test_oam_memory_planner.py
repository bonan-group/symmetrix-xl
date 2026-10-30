import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_oam_memory_planner_contract(tmp_path):
    source = tmp_path / "planner_test.cpp"
    source.write_text(
        r"""
#include "oam_memory_planner.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

using namespace symmetrix::oam;

int main() {
    OAMMemoryDimensions dimensions;
    dimensions.num_nodes = 3;
    dimensions.num_edges = 5;
    dimensions.scalar_bytes = 8;
    dimensions.harmonic_count = 4;
    dimensions.readout_width = 2;
    dimensions.readout_output_width = 1;
    dimensions.packed_input_values = 12;
    dimensions.packed_output_values = 8;
    dimensions.bilinear_forward_values = 21;
    dimensions.bilinear_reverse_values = 34;
    dimensions.reverse_partial_values = 15;
    dimensions.edge_chunk_values = 30;
    dimensions.layers.push_back({2, 3, 4, 5, 6, 7, 2, 6, 8});
    dimensions.layers.push_back({6, 4, 3, 8, 9, 5, 2, 9, 10});

    const auto speed = plan_oam_memory(dimensions, OAMRetentionPolicy::speed);
    const auto capacity = plan_oam_memory(dimensions, OAMRetentionPolicy::capacity);
    assert(speed.active_bytes >= capacity.active_bytes);
    assert(speed.active_bytes != 0);
    assert(speed.alignment == 64);
    for (const auto& slot : speed.slots) {
        assert(slot.offset % speed.alignment == 0);
        assert(slot.offset >= oam_arena_canary_bytes);
        assert(slot.offset + slot.bytes + oam_arena_canary_bytes
            <= speed.active_bytes);
    }
    assert(speed.slot("initial_features").bytes == 3 * 2 * 8);
    assert(speed.slot("reverse_partials").bytes == 15 * 8);
    assert(speed.slot("edge_chunk_scratch").bytes == 30 * 8);
    assert(speed.slot("final_reductions").bytes == (5 + 3 + 3 * 3 + 6) * 8);
    assert(speed.slot("packed_linear_scratch").bytes == (12 + 8) * 8);
    const auto& bilinear_forward = speed.slot("bilinear_forward_scratch");
    const auto& bilinear_reverse = speed.slot("bilinear_reverse_scratch");
    assert(bilinear_forward.bytes == 34 * 8);
    assert(bilinear_forward.offset == bilinear_reverse.offset);
    assert(bilinear_forward.live_end <= bilinear_reverse.live_begin);

    const auto& capacity_forward = capacity.slot("forward_scratch");
    const auto& capacity_reverse = capacity.slot("reverse_scratch");
    assert(capacity_forward.offset == capacity_reverse.offset);
    assert(capacity_forward.live_end <= capacity_reverse.live_begin);
    const auto& speed_forward = speed.slot("forward_scratch");
    const auto& speed_reverse = speed.slot("reverse_scratch");
    assert(speed_forward.offset != speed_reverse.offset);

    OAMTransientArenaState arena;
    arena.admit(capacity, "initial graph");
    assert(arena.replacements == 1);
    assert(arena.retained_capacity_bytes == capacity.active_bytes);
    arena.admit(capacity, "same graph");
    assert(arena.replacements == 1);

    auto smaller_dimensions = dimensions;
    smaller_dimensions.num_nodes = 1;
    smaller_dimensions.num_edges = 1;
    const auto smaller = plan_oam_memory(
        smaller_dimensions, OAMRetentionPolicy::capacity);
    arena.admit(smaller, "smaller graph");
    assert(arena.replacements == 1);
    assert(arena.active_bytes == smaller.active_bytes);
    assert(arena.retained_capacity_bytes == capacity.active_bytes);

    auto larger_dimensions = dimensions;
    larger_dimensions.num_nodes = 100;
    const auto larger = plan_oam_memory(
        larger_dimensions, OAMRetentionPolicy::capacity);
    arena.admit(larger, "node growth");
    assert(arena.replacements == 2);
    assert(arena.high_water_reason == "node growth");

    bool overflow = false;
    try {
        auto invalid = dimensions;
        invalid.num_nodes = std::numeric_limits<std::size_t>::max();
        (void)plan_oam_memory(invalid, OAMRetentionPolicy::speed);
    } catch (const std::overflow_error&) {
        overflow = true;
    }
    assert(overflow);

    bool bad_alias = false;
    try {
        validate_oam_alias({"left", 0, 64, 1, 4}, {"right", 0, 64, 3, 5});
    } catch (const std::logic_error&) {
        bad_alias = true;
    }
    assert(bad_alias);
}
"""
    )
    executable = tmp_path / "planner_test"
    compile_result = subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{ROOT / 'libsymmetrix/source'}",
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False
    )
    assert run_result.returncode == 0, run_result.stderr
