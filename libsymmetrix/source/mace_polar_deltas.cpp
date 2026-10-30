#include "mace_polar_deltas.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "KokkosBlas.hpp"
#include "tools_kokkos.hpp"

// The PolarMACE backbone deltas (mace_polar_deltas.hpp).  Everything here is
// the oracle's own arithmetic at the pinned checkout: the tied-channel tensor
// product of ``SparseUvuTensorProduct``, mace's ``GatedEquivariantBlock``, the
// three readout stacks, and the unrolled chain of ``PolarMACE.forward``.  The
// long-range operator group's portable kernels (long_range_kernels.hpp) are
// driven exactly as the declarative layer states them, with one shared
// geometry cache per evaluation.
namespace symmetrix::mace_polar {

namespace {

template<typename Precision>
std::vector<Precision> tensor_values(const nlohmann::json& value)
{
    return value.at("values").get<std::vector<Precision>>();
}

template<typename Precision>
void require_rows(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> view, const int columns,
    const char* what)
{
    if (static_cast<int>(view.extent(1)) != columns)
        throw std::invalid_argument(
            std::string("polar delta ") + what + " expects "
            + std::to_string(columns) + " columns, got "
            + std::to_string(view.extent(1)) + ".");
}

} // namespace

// ---------------------------------------------------------------------------
// The node-local tied-channel tensor product
// ---------------------------------------------------------------------------

template<typename Precision>
NodeLocalUvuTensorProductKokkosT<Precision>::NodeLocalUvuTensorProductKokkosT(
    const nlohmann::json& definition)
{
    const Irreps input_1(definition.at("irreps_in1").get<std::string>());
    const Irreps input_2(definition.at("irreps_in2").get<std::string>());
    const Irreps output(definition.at("irreps_out").get<std::string>());
    input_1_dimension_ = input_1.dimension();
    input_2_dimension_ = input_2.dimension();
    output_dimension_ = output.dimension();
    const std::vector<Precision> weight_values =
        tensor_values<Precision>(definition.at("weight"));
    const std::vector<Precision> mask_values =
        tensor_values<Precision>(definition.at("output_mask"));
    if (mask_values.size() != static_cast<std::size_t>(output_dimension_))
        throw std::invalid_argument(
            "polar uvu tensor product output mask dimension is inconsistent.");
    for (const auto& instruction : definition.at("instructions")) {
        if (instruction.at("connection_mode").get<std::string>() != "uvu")
            throw std::invalid_argument(
                "the polar delta tensor products admit only tied-channel "
                "('uvu') paths.");
        const IrrepBlock& block_1 =
            input_1.blocks.at(instruction.at("i_in1").get<int>());
        const IrrepBlock& block_2 =
            input_2.blocks.at(instruction.at("i_in2").get<int>());
        const IrrepBlock& block_out =
            output.blocks.at(instruction.at("i_out").get<int>());
        const std::vector<int> path_shape =
            instruction.at("path_shape").get<std::vector<int>>();
        if (path_shape.size() != 2
            || path_shape[0] != block_1.multiplicity
            || path_shape[1] != block_2.multiplicity)
            throw std::invalid_argument(
                "a polar uvu path's weight shape is inconsistent with its "
                "irreps.");
        if (block_out.multiplicity != block_1.multiplicity)
            throw std::invalid_argument(
                "polar uvu paths require the output multiplicity to match the "
                "first input's.");
        Path path;
        path.input_1_offset = block_1.offset;
        path.input_2_offset = block_2.offset;
        path.output_offset = block_out.offset;
        path.multiplicity_1 = block_1.multiplicity;
        path.multiplicity_2 = block_2.multiplicity;
        const int width_1 = 2 * block_1.l + 1;
        const int width_2 = 2 * block_2.l + 1;
        const int width_out = 2 * block_out.l + 1;
        if (width_out == 1 && width_1 == width_2) {
            path.contraction = true;
            path.component_width = width_1;
        } else if (width_2 == 1 && width_out == width_1) {
            path.contraction = false;
            path.component_width = width_1;
        } else {
            throw std::invalid_argument(
                "the polar delta tensor products admit only the (l x l -> 0) "
                "and (l x 0 -> l) tied-channel paths.");
        }
        path.path_weight = instruction.at("path_weight").get<Precision>();
        path.inverse_sqrt_width =
            Precision(1) / Kokkos::sqrt(Precision(path.component_width));
        path.weight_offset = weight_count_;
        weight_count_ += path.multiplicity_1 * path.multiplicity_2;
        paths_.push_back(path);
    }
    if (weight_values.size() != static_cast<std::size_t>(weight_count_))
        throw std::invalid_argument(
            "a polar uvu tensor product's weight count is inconsistent with "
            "its paths.");
    weights_ = toKokkosView("polar uvu weights", weight_values);
    output_mask_ = toKokkosView("polar uvu output mask", mask_values);
    path_table_ = Kokkos::View<Path*, Kokkos::SharedSpace>(
        "polar uvu paths", paths_.size());
    for (std::size_t index = 0; index < paths_.size(); ++index)
        path_table_(index) = paths_[index];
    dense_512_specialization_ = !paths_.empty()
        && std::all_of(paths_.begin(), paths_.end(), [](const Path& path) {
               return path.multiplicity_1 == 512
                   && path.multiplicity_2 == 512;
           });
    build_contribution_tables();
}

template<typename Precision>
void NodeLocalUvuTensorProductKokkosT<Precision>::build_contribution_tables()
{
    // Invert the path table into per-element contribution lists.  Entries are
    // appended path-major, channel-minor -- the exact order in which the
    // serial formulations accumulate into each element -- so the (sample,
    // element)-parallel kernels below reproduce the serial results
    // bit-identically.
    //
    // Entry kinds, by ``slot`` semantics and folded loop:
    //   0 forward contraction: element = output + slot; folds the other loop.
    //   1 forward modulation:  element = output + slot * width + component.
    //   2 reverse contraction, in-1 adjoint: element = in_1 + slot * width
    //     + component; folds the other loop.
    //   3 reverse contraction, in-2 adjoint: element = in_2 + slot * width
    //     + component; folds the channel loop.
    //   4 reverse modulation, in-1 adjoint: element = in_1 + slot * width
    //     + component.
    //   5 reverse modulation, in-2 adjoint: element = in_2 + slot; folds the
    //     channel loop (mixed_seed over components per channel).
    std::vector<std::vector<ElementEntry>> forward(output_dimension_);
    std::vector<std::vector<ElementEntry>> reverse_in1(input_1_dimension_);
    std::vector<std::vector<ElementEntry>> reverse_in2(input_2_dimension_);
    auto append = [](std::vector<ElementEntry>& list, int path, int slot,
                     int component, int kind) {
        ElementEntry entry;
        entry.path = path;
        entry.slot = slot;
        entry.component = component;
        entry.kind = kind;
        list.push_back(entry);
    };
    for (int index = 0; index < static_cast<int>(paths_.size()); ++index) {
        const Path& path = paths_[index];
        const int width = path.component_width;
        if (path.contraction) {
            for (int channel = 0; channel < path.multiplicity_1; ++channel) {
                append(forward[path.output_offset + channel], index, channel,
                    0, 0);
                for (int component = 0; component < width; ++component)
                    append(reverse_in1[path.input_1_offset + channel * width
                            + component],
                        index, channel, component, 2);
            }
            for (int other = 0; other < path.multiplicity_2; ++other)
                for (int component = 0; component < width; ++component)
                    append(reverse_in2[path.input_2_offset + other * width
                            + component],
                        index, other, component, 3);
        } else {
            for (int channel = 0; channel < path.multiplicity_1; ++channel) {
                for (int component = 0; component < width; ++component) {
                    append(forward[path.output_offset + channel * width
                            + component],
                        index, channel, component, 1);
                    append(reverse_in1[path.input_1_offset + channel * width
                            + component],
                        index, channel, component, 4);
                }
            }
            // One entry per (path, other): the entry itself folds the channel
            // loop, so appending it inside the channel loop above would
            // scale the in-2 adjoint by the channel count.
            for (int other = 0; other < path.multiplicity_2; ++other)
                append(reverse_in2[path.input_2_offset + other], index,
                    other, 0, 5);
        }
    }
    const auto upload_entries = [](const char* name,
                                   const std::vector<std::vector<ElementEntry>>&
                                       lists) {
        std::size_t count = 0;
        for (const auto& list : lists)
            count += list.size();
        Kokkos::View<ElementEntry*, Kokkos::SharedSpace> view(
            Kokkos::view_alloc(std::string(name)), count);
        std::size_t offset = 0;
        for (const auto& list : lists) {
            for (const ElementEntry& entry : list)
                view(offset++) = entry;
        }
        return view;
    };
    const auto upload_offsets = [](const char* name,
                                   const std::vector<std::vector<ElementEntry>>&
                                       lists) {
        Kokkos::View<int*, Kokkos::SharedSpace> view(
            Kokkos::view_alloc(std::string(name)), lists.size() + 1);
        std::size_t offset = 0;
        view(0) = 0;
        for (std::size_t index = 0; index < lists.size(); ++index) {
            offset += lists[index].size();
            view(index + 1) = static_cast<int>(offset);
        }
        return view;
    };
    forward_entries_ = upload_entries("polar uvu forward entries", forward);
    forward_offsets_ = upload_offsets("polar uvu forward offsets", forward);
    reverse_in1_entries_ = upload_entries(
        "polar uvu reverse in-1 entries", reverse_in1);
    reverse_in1_offsets_ = upload_offsets(
        "polar uvu reverse in-1 offsets", reverse_in1);
    reverse_in2_entries_ = upload_entries(
        "polar uvu reverse in-2 entries", reverse_in2);
    reverse_in2_offsets_ = upload_offsets(
        "polar uvu reverse in-2 offsets", reverse_in2);
}

template<typename Precision>
void NodeLocalUvuTensorProductKokkosT<Precision>::prepare(const int samples)
{
    if (!dense_512_specialization_)
        return;
    if (samples < 0)
        throw std::invalid_argument(
            "polar uvu workspace sample extent must be non-negative.");
    int max_packed_width = 1;
    int max_multiplicity_1 = 0;
    int max_multiplicity_2 = 0;
    for (const Path& path : paths_) {
        max_packed_width = std::max(max_packed_width,
            path.contraction ? path.component_width : 1);
        max_multiplicity_1 = std::max(
            max_multiplicity_1, path.multiplicity_1);
        max_multiplicity_2 = std::max(
            max_multiplicity_2, path.multiplicity_2);
    }
    if (samples > std::numeric_limits<int>::max() / max_packed_width)
        throw std::length_error(
            "polar uvu packed row extent exceeds INT_MAX.");
    const int rows = samples * max_packed_width;
    if (workspace_samples_ == samples && workspace_rows_ == rows
        && packed_input_2_workspace_.data() != nullptr)
        return;
    if (packed_input_2_workspace_.data() != nullptr)
        Kokkos::fence("Replace polar uvu dense workspace");
    packed_input_2_workspace_ = Kokkos::View<Precision**,
        Kokkos::LayoutRight>(
        Kokkos::view_alloc("polar uvu packed input-2 workspace",
            Kokkos::WithoutInitializing), rows, max_multiplicity_2);
    mixed_workspace_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        Kokkos::view_alloc("polar uvu mixed workspace",
            Kokkos::WithoutInitializing), rows, max_multiplicity_1);
    left_workspace_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        Kokkos::view_alloc("polar uvu left workspace",
            Kokkos::WithoutInitializing), rows, max_multiplicity_1);
    grad_input_2_workspace_ = Kokkos::View<Precision**,
        Kokkos::LayoutRight>(
        Kokkos::view_alloc("polar uvu grad input-2 workspace",
            Kokkos::WithoutInitializing), rows, max_multiplicity_2);
    workspace_samples_ = samples;
    workspace_rows_ = rows;
    ++workspace_replacement_count_;
}

template<typename Precision>
void NodeLocalUvuTensorProductKokkosT<Precision>::evaluate(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> input_1,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> input_2,
    Kokkos::View<Precision**, Kokkos::LayoutRight> output) const
{
    if (input_1.extent(0) != input_2.extent(0)
        || input_1.extent(0) != output.extent(0))
        throw std::invalid_argument(
            "polar uvu tensor product batch dimensions are inconsistent.");
    require_rows<Precision>(input_1, input_1_dimension_, "input_1");
    require_rows<Precision>(input_2, input_2_dimension_, "input_2");
    require_rows<Precision>(output, output_dimension_, "output");
    if (output.extent(0)
        > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error(
            "polar uvu sample extent exceeds INT_MAX.");
    const int samples = static_cast<int>(output.extent(0));
    if (samples == 0)
        return;
    if (dense_512_specialization_) {
        if (workspace_samples_ != samples
            || packed_input_2_workspace_.data() == nullptr)
            throw std::logic_error(
                "polar uvu dense workspace was not prepared for the batch.");
        Kokkos::deep_copy(
            Kokkos::DefaultExecutionSpace{}, output, Precision(0));
        using WeightMatrix = Kokkos::View<const Precision**,
            Kokkos::LayoutRight,
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        using WorkspaceMatrix = Kokkos::View<Precision**,
            Kokkos::LayoutRight,
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        for (const Path& host_path : paths_) {
            const Path path = host_path;
            const int packed_width =
                path.contraction ? path.component_width : 1;
            const int rows = samples * packed_width;
            const WorkspaceMatrix packed_input_2(
                packed_input_2_workspace_.data(), rows,
                path.multiplicity_2);
            const WorkspaceMatrix mixed(
                mixed_workspace_.data(), rows, path.multiplicity_1);
            const std::size_t packed_elements =
                static_cast<std::size_t>(rows) * path.multiplicity_2;
            Kokkos::parallel_for(
                "polar uvu dense pack input-2",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(0, packed_elements),
                KOKKOS_LAMBDA(const std::size_t flat) {
                    const int row = static_cast<int>(
                        flat / path.multiplicity_2);
                    const int other = static_cast<int>(
                        flat % path.multiplicity_2);
                    const int sample = row / packed_width;
                    const int component = row % packed_width;
                    packed_input_2(row, other) = input_2(sample,
                        path.input_2_offset + other * packed_width
                            + component);
                });
            const WeightMatrix weight_matrix(
                weights_.data() + path.weight_offset, path.multiplicity_1,
                path.multiplicity_2);
            KokkosBlas::gemm(
                "N", "T", Precision(1), packed_input_2, weight_matrix,
                Precision(0), mixed);
            const std::size_t output_channels =
                static_cast<std::size_t>(samples) * path.multiplicity_1;
            Kokkos::parallel_for(
                "polar uvu dense forward combine",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(0, output_channels),
                KOKKOS_LAMBDA(const std::size_t index) {
                    const int sample = static_cast<int>(
                        index / path.multiplicity_1);
                    const int channel = static_cast<int>(
                        index % path.multiplicity_1);
                    const Precision scale =
                        path.path_weight * path.inverse_sqrt_width;
                    if (path.contraction) {
                        Precision value = Precision(0);
                        for (int component = 0;
                            component < path.component_width; ++component)
                            value += input_1(sample,
                                         path.input_1_offset
                                             + channel * path.component_width
                                             + component)
                                * mixed(sample * path.component_width
                                        + component,
                                    channel);
                        output(sample, path.output_offset + channel) +=
                            scale * value;
                    } else {
                        const Precision value = scale * mixed(sample, channel);
                        for (int component = 0;
                            component < path.component_width; ++component)
                            output(sample,
                                path.output_offset
                                    + channel * path.component_width
                                    + component)
                                += value
                                * input_1(sample,
                                    path.input_1_offset
                                        + channel * path.component_width
                                        + component);
                    }
                });
        }
        auto mask = output_mask_;
        const int output_dimension = output_dimension_;
        Kokkos::parallel_for(
            "polar uvu dense output mask",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>,
                Kokkos::DefaultExecutionSpace>(
                {0, 0}, {samples, output_dimension}),
            KOKKOS_LAMBDA(const int sample, const int element) {
                output(sample, element) *= mask(element);
            });
        return;
    }
    auto paths = path_table_;
    auto weights = weights_;
    auto mask = output_mask_;
    auto entries = forward_entries_;
    auto offsets = forward_offsets_;
    // One work item per (sample, output element).  The original formulation
    // gave one thread per sample and walked the whole path table serially
    // inside it -- at 1500 nodes that is 1500 threads on a 170-SM device and
    // the kernel becomes latency-bound on uncached weight reads.  The inverted
    // contribution lists keep each element's accumulation sequence identical
    // to the serial walk (path-major, channel-minor), so this is the same
    // arithmetic reordered across threads, bit for bit.
    Kokkos::parallel_for(
        "polar uvu tensor product",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>,
            Kokkos::DefaultExecutionSpace>({0, 0}, {samples,
                output_dimension_}),
        KOKKOS_LAMBDA(const int sample, const int element) {
            Precision accumulator = Precision(0);
            const int end = offsets(element + 1);
            for (int index = offsets(element); index < end; ++index) {
                const ElementEntry entry = entries(index);
                const Path path = paths(entry.path);
                const Precision scale =
                    path.path_weight * path.inverse_sqrt_width;
                if (entry.kind == 0) {
                    Precision path_sum = Precision(0);
                    for (int other = 0; other < path.multiplicity_2;
                        ++other) {
                        Precision pair = Precision(0);
                        for (int component = 0;
                            component < path.component_width; ++component)
                            pair += input_1(sample,
                                    path.input_1_offset
                                        + entry.slot * path.component_width
                                        + component)
                                * input_2(sample,
                                    path.input_2_offset
                                        + other * path.component_width
                                        + component);
                        path_sum += pair
                            * weights(path.weight_offset
                                + entry.slot * path.multiplicity_2 + other);
                    }
                    accumulator += scale * path_sum;
                } else {
                    Precision mixed = Precision(0);
                    for (int other = 0; other < path.multiplicity_2; ++other)
                        mixed += weights(path.weight_offset
                                + entry.slot * path.multiplicity_2 + other)
                            * input_2(sample, path.input_2_offset + other);
                    accumulator += scale * mixed
                        * input_1(sample,
                            path.input_1_offset
                                + entry.slot * path.component_width
                                + entry.component);
                }
            }
            output(sample, element) = accumulator * mask(element);
        });
}

template<typename Precision>
void NodeLocalUvuTensorProductKokkosT<Precision>::reverse(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> input_1,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> input_2,
    Kokkos::View<Precision**, Kokkos::LayoutRight> input_1_adjoint,
    Kokkos::View<Precision**, Kokkos::LayoutRight> input_2_adjoint,
    const bool initialize_adjoints) const
{
    if (output_adjoint.extent(0) != input_1.extent(0)
        || output_adjoint.extent(0) != input_2.extent(0)
        || output_adjoint.extent(0) != input_1_adjoint.extent(0)
        || output_adjoint.extent(0) != input_2_adjoint.extent(0))
        throw std::invalid_argument(
            "polar uvu tensor product reverse batch dimensions are "
            "inconsistent.");
    require_rows<Precision>(output_adjoint, output_dimension_, "output_adjoint");
    require_rows<Precision>(input_1, input_1_dimension_, "input_1");
    require_rows<Precision>(input_2, input_2_dimension_, "input_2");
    require_rows<Precision>(input_1_adjoint, input_1_dimension_, "input_1_adjoint");
    require_rows<Precision>(input_2_adjoint, input_2_dimension_, "input_2_adjoint");
    if (output_adjoint.extent(0)
        > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error(
            "polar uvu sample extent exceeds INT_MAX.");
    const int samples = static_cast<int>(output_adjoint.extent(0));
    if (initialize_adjoints) {
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            input_1_adjoint, Precision(0));
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            input_2_adjoint, Precision(0));
    }
    if (samples == 0)
        return;
    if (dense_512_specialization_) {
        if (workspace_samples_ != samples
            || packed_input_2_workspace_.data() == nullptr)
            throw std::logic_error(
                "polar uvu dense workspace was not prepared for the batch.");
        using WeightMatrix = Kokkos::View<const Precision**,
            Kokkos::LayoutRight,
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        using WorkspaceMatrix = Kokkos::View<Precision**,
            Kokkos::LayoutRight,
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        auto mask = output_mask_;
        for (const Path& host_path : paths_) {
            const Path path = host_path;
            const int packed_width =
                path.contraction ? path.component_width : 1;
            const int rows = samples * packed_width;
            const WorkspaceMatrix packed_input_2(
                packed_input_2_workspace_.data(), rows,
                path.multiplicity_2);
            const WorkspaceMatrix mixed(
                mixed_workspace_.data(), rows, path.multiplicity_1);
            const std::size_t packed_input_elements =
                static_cast<std::size_t>(rows) * path.multiplicity_2;
            Kokkos::parallel_for(
                "polar uvu dense reverse pack input-2",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    0, packed_input_elements),
                KOKKOS_LAMBDA(const std::size_t flat) {
                    const int row = static_cast<int>(
                        flat / path.multiplicity_2);
                    const int other = static_cast<int>(
                        flat % path.multiplicity_2);
                    const int sample = row / packed_width;
                    const int component = row % packed_width;
                    packed_input_2(row, other) = input_2(sample,
                        path.input_2_offset + other * packed_width
                            + component);
                });
            const WeightMatrix weight_matrix(
                weights_.data() + path.weight_offset, path.multiplicity_1,
                path.multiplicity_2);
            KokkosBlas::gemm(
                "N", "T", Precision(1), packed_input_2, weight_matrix,
                Precision(0), mixed);
            const std::size_t input_1_elements =
                static_cast<std::size_t>(samples) * path.multiplicity_1;
            Kokkos::parallel_for(
                "polar uvu dense reverse input-1",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(0, input_1_elements),
                KOKKOS_LAMBDA(const std::size_t index) {
                    const int sample = static_cast<int>(
                        index / path.multiplicity_1);
                    const int channel = static_cast<int>(
                        index % path.multiplicity_1);
                    const Precision scale =
                        path.path_weight * path.inverse_sqrt_width;
                    if (path.contraction) {
                        const Precision seed = scale
                            * mask(path.output_offset + channel)
                            * output_adjoint(
                                sample, path.output_offset + channel);
                        for (int component = 0;
                            component < path.component_width; ++component)
                            input_1_adjoint(sample,
                                path.input_1_offset
                                    + channel * path.component_width
                                    + component)
                                += seed
                                * mixed(sample * path.component_width
                                        + component,
                                    channel);
                    } else {
                        const Precision value = mixed(sample, channel);
                        for (int component = 0;
                            component < path.component_width; ++component) {
                            const int element = path.output_offset
                                + channel * path.component_width + component;
                            input_1_adjoint(sample,
                                path.input_1_offset
                                    + channel * path.component_width
                                    + component)
                                += scale * mask(element)
                                * output_adjoint(sample, element) * value;
                        }
                    }
                });

            const WorkspaceMatrix left(
                left_workspace_.data(), rows, path.multiplicity_1);
            const std::size_t left_elements =
                static_cast<std::size_t>(rows) * path.multiplicity_1;
            Kokkos::parallel_for(
                "polar uvu dense reverse seed pack",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(0, left_elements),
                KOKKOS_LAMBDA(const std::size_t flat) {
                    const int row = static_cast<int>(
                        flat / path.multiplicity_1);
                    const int channel = static_cast<int>(
                        flat % path.multiplicity_1);
                    const int sample = row / packed_width;
                    const int component = row % packed_width;
                    const Precision scale =
                        path.path_weight * path.inverse_sqrt_width;
                    if (path.contraction) {
                        left(row, channel) = scale
                            * mask(path.output_offset + channel)
                            * output_adjoint(
                                sample, path.output_offset + channel)
                            * input_1(sample,
                                path.input_1_offset
                                    + channel * path.component_width
                                    + component);
                    } else {
                        Precision value = Precision(0);
                        for (int output_component = 0;
                            output_component < path.component_width;
                            ++output_component) {
                            const int element = path.output_offset
                                + channel * path.component_width
                                + output_component;
                            value += mask(element)
                                * output_adjoint(sample, element)
                                * input_1(sample,
                                    path.input_1_offset
                                        + channel * path.component_width
                                        + output_component);
                        }
                        left(row, channel) = scale * value;
                    }
                });
            const WorkspaceMatrix grad_input_2(
                grad_input_2_workspace_.data(), rows,
                path.multiplicity_2);
            KokkosBlas::gemm(
                "N", "N", Precision(1), left, weight_matrix, Precision(0),
                grad_input_2);
            const std::size_t grad_input_elements =
                static_cast<std::size_t>(rows) * path.multiplicity_2;
            Kokkos::parallel_for(
                "polar uvu dense reverse unpack input-2",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    0, grad_input_elements),
                KOKKOS_LAMBDA(const std::size_t flat) {
                    const int row = static_cast<int>(
                        flat / path.multiplicity_2);
                    const int other = static_cast<int>(
                        flat % path.multiplicity_2);
                    const int sample = row / packed_width;
                    const int component = row % packed_width;
                    input_2_adjoint(sample,
                        path.input_2_offset + other * packed_width
                            + component)
                        += grad_input_2(row, other);
                });
        }
        return;
    }
    auto paths = path_table_;
    auto weights = weights_;
    auto mask = output_mask_;
    // Same inversion as the forward: one work item per (sample, adjoint
    // element), contributions accumulated in the serial walk's order.  The
    // two adjoints are filled by separate launches so their elements stay
    // disjoint and no atomics are needed.
    {
        auto entries = reverse_in1_entries_;
        auto offsets = reverse_in1_offsets_;
        Kokkos::parallel_for(
            "polar uvu tensor product reverse in-1",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>,
                Kokkos::DefaultExecutionSpace>({0, 0}, {samples,
                    input_1_dimension_}),
            KOKKOS_LAMBDA(const int sample, const int element) {
                Precision accumulator = Precision(0);
                const int end = offsets(element + 1);
                for (int index = offsets(element); index < end; ++index) {
                    const ElementEntry entry = entries(index);
                    const Path path = paths(entry.path);
                    const Precision scale =
                        path.path_weight * path.inverse_sqrt_width;
                    if (entry.kind == 2) {
                        const Precision seed = scale
                            * mask(path.output_offset + entry.slot)
                            * output_adjoint(sample,
                                path.output_offset + entry.slot);
                        for (int other = 0; other < path.multiplicity_2;
                            ++other) {
                            const Precision path_seed = seed
                                * weights(path.weight_offset
                                    + entry.slot * path.multiplicity_2
                                    + other);
                            accumulator += path_seed
                                * input_2(sample,
                                    path.input_2_offset
                                        + other * path.component_width
                                        + entry.component);
                        }
                    } else {
                        const int output_element = path.output_offset
                            + entry.slot * path.component_width
                            + entry.component;
                        const Precision incoming = scale
                            * mask(output_element)
                            * output_adjoint(sample, output_element);
                        Precision mixed = Precision(0);
                        for (int other = 0; other < path.multiplicity_2;
                            ++other)
                            mixed += weights(path.weight_offset
                                    + entry.slot * path.multiplicity_2 + other)
                                * input_2(sample,
                                    path.input_2_offset + other);
                        accumulator += incoming * mixed;
                    }
                }
                input_1_adjoint(sample, element) += accumulator;
            });
    }
    {
        auto entries = reverse_in2_entries_;
        auto offsets = reverse_in2_offsets_;
        Kokkos::parallel_for(
            "polar uvu tensor product reverse in-2",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>,
                Kokkos::DefaultExecutionSpace>({0, 0}, {samples,
                    input_2_dimension_}),
            KOKKOS_LAMBDA(const int sample, const int element) {
                Precision accumulator = Precision(0);
                const int end = offsets(element + 1);
                for (int index = offsets(element); index < end; ++index) {
                    const ElementEntry entry = entries(index);
                    const Path path = paths(entry.path);
                    const Precision scale =
                        path.path_weight * path.inverse_sqrt_width;
                    if (entry.kind == 3) {
                        for (int channel = 0; channel < path.multiplicity_1;
                            ++channel) {
                            const Precision seed = scale
                                * mask(path.output_offset + channel)
                                * output_adjoint(sample,
                                    path.output_offset + channel);
                            accumulator += seed
                                * weights(path.weight_offset
                                    + channel * path.multiplicity_2
                                    + entry.slot)
                                * input_1(sample,
                                    path.input_1_offset
                                        + channel * path.component_width
                                        + entry.component);
                        }
                    } else {
                        for (int channel = 0; channel < path.multiplicity_1;
                            ++channel) {
                            Precision mixed_seed = Precision(0);
                            for (int component = 0;
                                component < path.component_width;
                                ++component) {
                                const int output_element =
                                    path.output_offset
                                        + channel * path.component_width
                                        + component;
                                const Precision incoming = scale
                                    * mask(output_element)
                                    * output_adjoint(sample, output_element);
                                mixed_seed += incoming
                                    * input_1(sample,
                                        path.input_1_offset
                                            + channel * path.component_width
                                            + component);
                            }
                            accumulator += mixed_seed
                                * weights(path.weight_offset
                                    + channel * path.multiplicity_2
                                    + entry.slot);
                        }
                    }
                }
                input_2_adjoint(sample, element) += accumulator;
            });
    }
}

// ---------------------------------------------------------------------------
// The gated nonlinearity
// ---------------------------------------------------------------------------

template<typename Precision>
PolarGateKokkosT<Precision>::PolarGateKokkosT(const nlohmann::json& definition)
{
    const Irreps scalars(definition.at("irreps_scalars").get<std::string>());
    const Irreps gates(definition.at("irreps_gates").get<std::string>());
    const Irreps gated(definition.at("irreps_gated").get<std::string>());
    // The activation constants are plain numeric lists (the oracle's own
    // scalar second-moment normalizations, not model tensors).
    const std::vector<Precision> scalar_constants =
        definition.at("scalar_activation_constants").get<std::vector<Precision>>();
    const std::vector<Precision> gate_constants =
        definition.at("gate_activation_constants").get<std::vector<Precision>>();
    scalar_constant_ = scalar_constants.empty() ? Precision(1)
                                                : scalar_constants.front();
    gate_constant_ = gate_constants.empty() ? Precision(1)
                                            : gate_constants.front();
    const auto resolve = [] (const std::string& name, const char* what) {
        if (name == "silu")
            return Activation::silu;
        if (name == "sigmoid")
            return Activation::sigmoid;
        if (name == "identity" || name.empty())
            return Activation::identity;
        throw std::invalid_argument(
            std::string("polar gate ") + what + " has an unsupported "
            "activation '" + name + "' (the vocabulary is silu, sigmoid, "
            "identity).");
    };
    // The extractor records the activation names as plain strings (the
    // oracle's own names), so the kernel resolves exactly that pair.
    scalar_activation_ = resolve(
        definition.value("scalar_activation", std::string("silu")),
        "scalar activation");
    gate_activation_ = resolve(
        definition.value("gate_activation", std::string("sigmoid")),
        "activation");

    // The oracle's own layout rule: the input is (scalars + gates + gated)
    // stable-sorted by (l, -parity, multiplicity) and simplified.  The kernel
    // reproduces that order rather than assuming it, so a block whose
    // declaration would sort differently is refused rather than mis-read.
    struct Group {
        int multiplicity, l, parity, source;
    };
    std::vector<Group> groups;
    const auto append = [&groups] (const Irreps& irreps, const int source) {
        for (const IrrepBlock& block : irreps.blocks)
            groups.push_back({block.multiplicity, block.l, block.parity, source});
    };
    append(scalars, 0);
    append(gates, 1);
    append(gated, 2);
    std::vector<int> order(groups.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&groups] (const int left,
        const int right) {
        const Group& a = groups[static_cast<std::size_t>(left)];
        const Group& b = groups[static_cast<std::size_t>(right)];
        if (a.l != b.l)
            return a.l < b.l;
        if (a.parity != b.parity)
            return a.parity > b.parity;
        return a.multiplicity < b.multiplicity;
    });
    for (std::size_t index = 0; index < order.size(); ++index)
        if (order[index] != static_cast<int>(index))
            throw std::invalid_argument(
                "the polar gate's declared (scalars, gates, gated) order is not "
                "the sorted order the oracle lays the input out in.");
    int offset = 0;
    bool scalar_seen = false, gate_seen = false;
    for (const Group& group : groups) {
        const int length = group.multiplicity * (2 * group.l + 1);
        if (group.source == 0) {
            if (!scalar_seen) {
                scalar_offset_ = offset;
                scalar_seen = true;
            }
            scalar_length_ += length;
        } else if (group.source == 1) {
            if (!gate_seen) {
                gate_offset_ = offset;
                gate_seen = true;
            }
            gate_length_ += length;
        } else {
            GatedGroup gated_group;
            gated_group.offset = offset;
            gated_group.length = length;
            gated_group.component_width = 2 * group.l + 1;
            gated_group.multiplicity = group.multiplicity;
            gated_.push_back(gated_group);
        }
        offset += length;
        input_dimension_ += length;
    }
    // The oracle indexes each gated group's own gate channels out of the gate
    // slice in gated order.
    int gate_cursor = 0;
    for (GatedGroup& group : gated_) {
        group.gate_offset = gate_cursor;
        gate_cursor += group.multiplicity;
    }
    if (gate_cursor != gate_length_)
        throw std::invalid_argument(
            "the polar gate's gate channels must match its gated channels.");
    output_dimension_ = scalar_length_;
    for (const GatedGroup& group : gated_)
        output_dimension_ += group.length;
    const Irreps declared_in(definition.at("irreps_in").get<std::string>());
    const Irreps declared_out(definition.at("irreps_out").get<std::string>());
    if (declared_in.dimension() != input_dimension_)
        throw std::invalid_argument(
            "the polar gate's declared input irreps do not match its block "
            "layout.");
    if (declared_out.dimension() != output_dimension_)
        throw std::invalid_argument(
            "the polar gate's declared output irreps do not match its block "
            "layout.");
    group_table_ = Kokkos::View<GatedGroup*, Kokkos::SharedSpace>(
        "polar gate groups", gated_.size());
    for (std::size_t index = 0; index < gated_.size(); ++index)
        group_table_(index) = gated_[index];

    std::vector<ForwardElement> forward(
        static_cast<std::size_t>(output_dimension_));
    std::vector<ReverseElement> reverse(
        static_cast<std::size_t>(input_dimension_));
    for (int index = 0; index < scalar_length_; ++index) {
        const int input_element = scalar_offset_ + index;
        forward[static_cast<std::size_t>(index)] = {input_element, -1};
        reverse[static_cast<std::size_t>(input_element)] =
            {0, index, 0, 0, -1};
    }
    int written = scalar_length_;
    for (const GatedGroup& group : gated_) {
        for (int channel = 0; channel < group.multiplicity; ++channel) {
            const int gate_element =
                gate_offset_ + group.gate_offset + channel;
            const int output_offset =
                written + channel * group.component_width;
            const int source_offset =
                group.offset + channel * group.component_width;
            reverse[static_cast<std::size_t>(gate_element)] =
                {1, output_offset, source_offset, group.component_width,
                    gate_element};
            for (int component = 0; component < group.component_width;
                ++component) {
                const int output_element = output_offset + component;
                const int input_element = source_offset + component;
                forward[static_cast<std::size_t>(output_element)] =
                    {input_element, gate_element};
                reverse[static_cast<std::size_t>(input_element)] =
                    {2, output_element, 0, 0, gate_element};
            }
        }
        written += group.length;
    }
    if (written != output_dimension_
        || std::any_of(forward.begin(), forward.end(),
            [] (const ForwardElement& element) {
                return element.input_element < 0;
            })
        || std::any_of(reverse.begin(), reverse.end(),
            [] (const ReverseElement& element) { return element.kind < 0; }))
        throw std::invalid_argument(
            "the polar gate's element ownership table is incomplete.");
    forward_elements_ =
        Kokkos::View<ForwardElement*, Kokkos::SharedSpace>(
            "polar gate forward elements", forward.size());
    reverse_elements_ =
        Kokkos::View<ReverseElement*, Kokkos::SharedSpace>(
            "polar gate reverse elements", reverse.size());
    for (std::size_t index = 0; index < forward.size(); ++index)
        forward_elements_(index) = forward[index];
    for (std::size_t index = 0; index < reverse.size(); ++index)
        reverse_elements_(index) = reverse[index];
}

template<typename Precision>
void PolarGateKokkosT<Precision>::evaluate(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> input,
    Kokkos::View<Precision**, Kokkos::LayoutRight> output) const
{
    if (input.extent(0) != output.extent(0))
        throw std::invalid_argument(
            "polar gate batch dimensions are inconsistent.");
    require_rows<Precision>(input, input_dimension_, "gate input");
    require_rows<Precision>(output, output_dimension_, "gate output");
    const Precision scalar_constant = scalar_constant_;
    const Precision gate_constant = gate_constant_;
    const Activation scalar_activation = scalar_activation_;
    const Activation gate_activation = gate_activation_;
    const int output_dimension = output_dimension_;
    if (output_dimension == 0)
        return;
    if (input.extent(0) > std::numeric_limits<std::size_t>::max()
            / static_cast<std::size_t>(output_dimension))
        throw std::length_error("the polar gate output extent overflows size_t.");
    const std::size_t elements = input.extent(0)
        * static_cast<std::size_t>(output_dimension);
    auto forward_elements = forward_elements_;
    Kokkos::parallel_for(
        "polar gate",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(0, elements),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t sample = flat / output_dimension;
            const int output_element =
                static_cast<int>(flat % output_dimension);
            const ForwardElement element = forward_elements(output_element);
            if (element.gate_element < 0) {
                const Precision value = input(sample, element.input_element);
                Precision activated = value;
                if (scalar_activation == Activation::silu)
                    activated = value / (Precision(1) + Kokkos::exp(-value));
                else if (scalar_activation == Activation::sigmoid)
                    activated = Precision(1)
                        / (Precision(1) + Kokkos::exp(-value));
                output(sample, output_element) = scalar_constant * activated;
                return;
            }
            const Precision value = input(sample, element.gate_element);
            Precision activated = value;
            if (gate_activation == Activation::silu)
                activated = value / (Precision(1) + Kokkos::exp(-value));
            else if (gate_activation == Activation::sigmoid)
                activated = Precision(1)
                    / (Precision(1) + Kokkos::exp(-value));
            output(sample, output_element) = gate_constant * activated
                * input(sample, element.input_element);
        });
}

template<typename Precision>
void PolarGateKokkosT<Precision>::reverse(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> input,
    Kokkos::View<Precision**, Kokkos::LayoutRight> input_adjoint,
    const bool initialize_input_adjoint) const
{
    if (output_adjoint.extent(0) != input.extent(0)
        || output_adjoint.extent(0) != input_adjoint.extent(0))
        throw std::invalid_argument(
            "polar gate reverse batch dimensions are inconsistent.");
    require_rows<Precision>(output_adjoint, output_dimension_, "gate output adjoint");
    require_rows<Precision>(input, input_dimension_, "gate input");
    require_rows<Precision>(input_adjoint, input_dimension_, "gate input adjoint");
    if (initialize_input_adjoint)
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            input_adjoint, Precision(0));
    const Precision scalar_constant = scalar_constant_;
    const Precision gate_constant = gate_constant_;
    const Activation scalar_activation = scalar_activation_;
    const Activation gate_activation = gate_activation_;
    const int input_dimension = input_dimension_;
    if (input_dimension == 0)
        return;
    if (input.extent(0) > std::numeric_limits<std::size_t>::max()
            / static_cast<std::size_t>(input_dimension))
        throw std::length_error("the polar gate input extent overflows size_t.");
    const std::size_t elements = input.extent(0)
        * static_cast<std::size_t>(input_dimension);
    auto reverse_elements = reverse_elements_;
    Kokkos::parallel_for(
        "polar gate reverse",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(0, elements),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t sample = flat / input_dimension;
            const int input_element = static_cast<int>(flat % input_dimension);
            const ReverseElement element = reverse_elements(input_element);
            if (element.kind == 0) {
                const Precision value = input(sample, input_element);
                Precision derivative = Precision(1);
                if (scalar_activation == Activation::silu) {
                    const Precision sigmoid =
                        Precision(1) / (Precision(1) + Kokkos::exp(-value));
                    derivative =
                        sigmoid * (Precision(1) + value * (Precision(1) - sigmoid));
                } else if (scalar_activation == Activation::sigmoid) {
                    const Precision sigmoid =
                        Precision(1) / (Precision(1) + Kokkos::exp(-value));
                    derivative = sigmoid * (Precision(1) - sigmoid);
                }
                input_adjoint(sample, input_element) += scalar_constant
                    * derivative * output_adjoint(sample, element.output_element);
                return;
            }
            const Precision value = input(sample, element.gate_element);
            Precision activated = value, derivative = Precision(1);
            if (gate_activation == Activation::silu) {
                const Precision sigmoid =
                    Precision(1) / (Precision(1) + Kokkos::exp(-value));
                activated = value * sigmoid;
                derivative = sigmoid
                    * (Precision(1) + value * (Precision(1) - sigmoid));
            } else if (gate_activation == Activation::sigmoid) {
                const Precision sigmoid =
                    Precision(1) / (Precision(1) + Kokkos::exp(-value));
                activated = sigmoid;
                derivative = sigmoid * (Precision(1) - sigmoid);
            }
            if (element.kind == 2) {
                input_adjoint(sample, input_element) +=
                    output_adjoint(sample, element.output_element)
                    * gate_constant * activated;
                return;
            }
            Precision gate_seed = Precision(0);
            for (int component = 0; component < element.component_width;
                ++component)
                gate_seed += output_adjoint(
                    sample, element.output_element + component)
                    * input(sample, element.source_offset + component);
            input_adjoint(sample, input_element) +=
                gate_constant * derivative * gate_seed;
        });
}

// ---------------------------------------------------------------------------
// The readouts
// ---------------------------------------------------------------------------

template<typename Precision>
PolarGatedReadoutKokkosT<Precision>::PolarGatedReadoutKokkosT(
    const nlohmann::json& definition)
    :linear_1_(polar_specialized_linear(
         definition.at("linear_1"),"update_readout_first")),
     linear_mid_(polar_specialized_linear(
         definition.at("linear_mid"),"update_readout_mid")),
     linear_2_(polar_specialized_linear(
         definition.at("linear_2"),"update_readout_final")),
     gate_(definition.at("gate"))
{
    if (linear_mid_.input_dimension() != gate_.output_dimension()
        || linear_mid_.output_dimension() != gate_.input_dimension()
        || linear_2_.input_dimension() != gate_.output_dimension())
        throw std::invalid_argument(
            "the polar gated readout's linear and gate dimensions are "
            "inconsistent.");
}

template<typename Precision>
void PolarGatedReadoutKokkosT<Precision>::prepare(const int nodes)
{
    if (static_cast<int>(hidden_.extent(0)) == nodes && hidden_.extent(1) != 0)
        return;
    const int gate_in = gate_.input_dimension();
    const int gate_out = gate_.output_dimension();
    // hidden_ and gated_ hold the two *wide* values the reverse needs (the
    // gate's inputs, one per gate application); the two narrow scratch
    // buffers hold the gate outputs, and the two adjoint buffers alternate
    // between the narrow and wide spaces.
    hidden_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar readout hidden", nodes, gate_in);
    gated_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar readout gated", nodes, gate_in);
    gate_input_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar readout gate input", nodes, gate_out);
    gate_output_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar readout gate output", nodes, gate_out);
    adjoint_a_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar readout adjoint a", nodes, gate_out);
    adjoint_b_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar readout adjoint b", nodes, gate_in);
}

template<typename Precision>
void PolarGatedReadoutKokkosT<Precision>::evaluate(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> input,
    Kokkos::View<Precision**, Kokkos::LayoutRight> output)
{
    require_rows<Precision>(input, input_dimension(), "readout input");
    require_rows<Precision>(output, output_dimension(), "readout output");
    linear_1_.evaluate(input, hidden_);
    gate_.evaluate(hidden_, gate_input_);
    linear_mid_.evaluate(gate_input_, gated_);
    gate_.evaluate(gated_, gate_output_);
    linear_2_.evaluate(gate_output_, output);
}

template<typename Precision>
void PolarGatedReadoutKokkosT<Precision>::reverse(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> input,
    Kokkos::View<Precision**, Kokkos::LayoutRight> input_adjoint,
    const bool initialize_input_adjoint)
{
    (void)input;
    linear_2_.reverse(output_adjoint, adjoint_a_, true);
    gate_.reverse(adjoint_a_, gated_, adjoint_b_, true);
    linear_mid_.reverse(adjoint_b_, adjoint_a_, true);
    gate_.reverse(adjoint_a_, hidden_, adjoint_b_, true);
    linear_1_.reverse(adjoint_b_, input_adjoint, initialize_input_adjoint);
}

template<typename Precision>
PolarNonLinearBiasReadoutKokkosT<Precision>::PolarNonLinearBiasReadoutKokkosT(
    const nlohmann::json& definition)
    :linear_1_(polar_specialized_linear(
         definition.at("linear_1"),"fukui_first")),
     linear_mid_(polar_specialized_linear(
         definition.at("linear_mid"),"fukui_mid")),
     linear_2_(polar_specialized_linear(
         definition.at("linear_2"),"fukui_final"))
{
    const Irreps hidden(
        definition.at("linear_1").at("irreps_out").get<std::string>());
    for (const IrrepBlock& block : hidden.blocks)
        if (block.l != 0)
            throw std::invalid_argument(
                "the polar non-linear-bias readout's hidden irreps must be "
                "scalars (the oracle activates them elementwise).");
    // The activation constants are emitted as a plain numeric list (they are
    // the oracle's own scalar second-moment normalizations, not model tensors).
    const std::vector<Precision> constants =
        definition.at("activation_constants").get<std::vector<Precision>>();
    activation_constant_ = constants.empty() ? Precision(1) : constants.front();
    const std::string name = definition.value("activation", std::string("silu"));
    if (name == "silu")
        activation_ = Activation::silu;
    else if (name == "sigmoid")
        activation_ = Activation::sigmoid;
    else if (name == "identity" || name.empty())
        activation_ = Activation::identity;
    else
        throw std::invalid_argument(
            "polar readout has an unsupported activation '" + name + "'.");
}

template<typename Precision>
void PolarNonLinearBiasReadoutKokkosT<Precision>::prepare(const int nodes)
{
    if (static_cast<int>(first_pre_.extent(0)) == nodes
        && first_pre_.extent(1) != 0)
        return;
    const int hidden = linear_1_.output_dimension();
    first_pre_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar bias readout first", nodes, hidden);
    first_activated_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar bias readout first activated", nodes, hidden);
    second_pre_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar bias readout second", nodes, hidden);
    second_activated_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar bias readout second activated", nodes, hidden);
    adjoint_a_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar bias readout adjoint a", nodes, hidden);
    adjoint_b_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar bias readout adjoint b", nodes, hidden);
}

template<typename Precision>
void PolarNonLinearBiasReadoutKokkosT<Precision>::activate(
    Kokkos::View<Precision**, Kokkos::LayoutRight> pre,
    Kokkos::View<Precision**, Kokkos::LayoutRight> activated) const
{
    const Precision constant = activation_constant_;
    const Activation activation = activation_;
    Kokkos::parallel_for(
        "polar bias readout activation", pre.size(),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t row = flat / pre.extent(1);
            const std::size_t column = flat % pre.extent(1);
            const Precision value = pre(row, column);
            Precision result = value;
            if (activation == Activation::silu)
                result = value / (Precision(1) + Kokkos::exp(-value));
            else if (activation == Activation::sigmoid)
                result = Precision(1) / (Precision(1) + Kokkos::exp(-value));
            activated(row, column) = constant * result;
        });
}

template<typename Precision>
void PolarNonLinearBiasReadoutKokkosT<Precision>::reverse_activate(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> pre,
    Kokkos::View<Precision**, Kokkos::LayoutRight> seed) const
{
    const Precision constant = activation_constant_;
    const Activation activation = activation_;
    auto mutable_seed = seed;
    Kokkos::parallel_for(
        "polar bias readout reverse activation", seed.size(),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t row = flat / seed.extent(1);
            const std::size_t column = flat % seed.extent(1);
            const Precision value = pre(row, column);
            Precision derivative = Precision(1);
            if (activation == Activation::silu) {
                const Precision sigmoid =
                    Precision(1) / (Precision(1) + Kokkos::exp(-value));
                derivative =
                    sigmoid * (Precision(1) + value * (Precision(1) - sigmoid));
            } else if (activation == Activation::sigmoid) {
                const Precision sigmoid =
                    Precision(1) / (Precision(1) + Kokkos::exp(-value));
                derivative = sigmoid * (Precision(1) - sigmoid);
            }
            mutable_seed(row, column) *= constant * derivative;
        });
}

template<typename Precision>
void PolarNonLinearBiasReadoutKokkosT<Precision>::evaluate(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> input,
    Kokkos::View<Precision**, Kokkos::LayoutRight> output)
{
    require_rows<Precision>(input, input_dimension(), "bias readout input");
    require_rows<Precision>(output, output_dimension(), "bias readout output");
    linear_1_.evaluate(input, first_pre_);
    activate(first_pre_, first_activated_);
    linear_mid_.evaluate(first_activated_, second_pre_);
    activate(second_pre_, second_activated_);
    linear_2_.evaluate(second_activated_, output);
}

template<typename Precision>
void PolarNonLinearBiasReadoutKokkosT<Precision>::reverse(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> input,
    Kokkos::View<Precision**, Kokkos::LayoutRight> input_adjoint,
    const bool initialize_input_adjoint)
{
    (void)input;
    linear_2_.reverse(output_adjoint, adjoint_a_, true);
    reverse_activate(second_pre_, adjoint_a_);
    linear_mid_.reverse(adjoint_a_, adjoint_b_, true);
    reverse_activate(first_pre_, adjoint_b_);
    linear_1_.reverse(adjoint_b_, input_adjoint, initialize_input_adjoint);
}

// ---------------------------------------------------------------------------
// The potential embedding
// ---------------------------------------------------------------------------

template<typename Precision>
PolarPotentialEmbeddingKokkosT<Precision>::PolarPotentialEmbeddingKokkosT(
    const nlohmann::json& definition)
    :potential_linear_(polar_specialized_linear(
         definition.at("potential_linear"),"potential_embedding")),
     node_feats_linear_(polar_specialized_linear(
         definition.at("node_feats_linear"),"node_embedding")),
     charge_embedding_(polar_specialized_linear(
         definition.at("charge_embedding"),"charge_embedding"))
{
    if (potential_linear_.output_dimension() != node_feats_linear_.output_dimension()
        || potential_linear_.output_dimension()
            != charge_embedding_.output_dimension())
        throw std::invalid_argument(
            "the polar potential embedding's three linears must share one "
            "output width.");

    const auto& potential_definition = definition.at("potential_linear");
    const auto& node_definition = definition.at("node_feats_linear");
    const auto& charge_definition = definition.at("charge_embedding");
    constexpr const char* hidden_irreps = "512x0e+512x1o";
    const auto has_paths = [] (const nlohmann::json& linear,
                               const std::vector<std::pair<int, int>>& paths) {
        const auto& instructions = linear.at("instructions");
        if (instructions.size() != paths.size())
            return false;
        for (std::size_t index = 0; index < paths.size(); ++index)
            if (instructions.at(index).at("i_in").get<int>()
                    != paths[index].first
                || instructions.at(index).at("i_out").get<int>()
                    != paths[index].second)
                return false;
        return true;
    };
    const auto has_plain_finalization = [] (const nlohmann::json& linear) {
        const auto bias = linear.at("bias").at("values")
            .get<std::vector<Precision>>();
        const auto mask = linear.at("output_mask").at("values")
            .get<std::vector<Precision>>();
        return bias.empty() && std::all_of(
            mask.begin(), mask.end(), [] (const Precision value) {
                return value == Precision(1);
            });
    };
    use_composed_schedule_ =
        potential_definition.at("irreps_in").get<std::string>()
            == "2x0e+2x1o+2x0e+2x1o"
        && node_definition.at("irreps_in").get<std::string>() == hidden_irreps
        && charge_definition.at("irreps_in").get<std::string>()
            == "1x0e+1x1o+1x0e+1x1o"
        && potential_definition.at("irreps_out").get<std::string>()
            == hidden_irreps
        && node_definition.at("irreps_out").get<std::string>() == hidden_irreps
        && charge_definition.at("irreps_out").get<std::string>()
            == hidden_irreps
        && has_paths(potential_definition,
            {{0, 0}, {1, 1}, {2, 0}, {3, 1}})
        && has_paths(node_definition, {{0, 0}, {1, 1}})
        && has_paths(charge_definition,
            {{0, 0}, {1, 1}, {2, 0}, {3, 1}})
        && has_plain_finalization(potential_definition)
        && has_plain_finalization(node_definition)
        && has_plain_finalization(charge_definition);
    if (!use_composed_schedule_)
        return;

    constexpr int packed_input_width = 518;
    constexpr int packed_output_width = 512;
    std::vector<PolarPotentialEmbeddingPackedInputSource> scalar_sources;
    std::vector<PolarPotentialEmbeddingPackedInputSource> vector_sources;
    std::vector<int> scalar_block_offsets[3];
    std::vector<int> vector_block_offsets[3];
    const nlohmann::json* definitions[3] = {
        &potential_definition, &node_definition, &charge_definition};
    for (int input_kind = 0; input_kind < 3; ++input_kind) {
        const Irreps input(
            definitions[input_kind]->at("irreps_in").get<std::string>());
        scalar_block_offsets[input_kind].assign(input.blocks.size(), -1);
        vector_block_offsets[input_kind].assign(input.blocks.size(), -1);
        for (std::size_t block_index = 0; block_index < input.blocks.size();
            ++block_index) {
            const IrrepBlock& block = input.blocks[block_index];
            auto& sources = block.l == 0 ? scalar_sources : vector_sources;
            auto& offsets = block.l == 0
                ? scalar_block_offsets[input_kind]
                : vector_block_offsets[input_kind];
            offsets[block_index] = static_cast<int>(sources.size());
            const int component_width = 2 * block.l + 1;
            for (int source = 0; source < block.multiplicity; ++source)
                sources.push_back(
                    {input_kind, block.offset, component_width, source});
        }
    }
    if (scalar_sources.size() != packed_input_width
        || vector_sources.size() != packed_input_width)
        throw std::invalid_argument(
            "the admitted M potential embedding packed width is inconsistent.");

    std::vector<Precision> scalar_weights(
        static_cast<std::size_t>(packed_input_width) * packed_output_width,
        Precision(0));
    std::vector<Precision> vector_weights(scalar_weights.size(), Precision(0));
    const auto append_weights = [&] (const nlohmann::json& linear,
                                    const int input_kind) {
        const Irreps input(linear.at("irreps_in").get<std::string>());
        const Irreps output(linear.at("irreps_out").get<std::string>());
        const auto weights = linear.at("weight").at("values")
            .get<std::vector<Precision>>();
        int weight_offset = 0;
        for (const auto& instruction : linear.at("instructions")) {
            const int input_block_index = instruction.at("i_in").get<int>();
            const IrrepBlock& input_block = input.blocks.at(input_block_index);
            const IrrepBlock& output_block =
                output.blocks.at(instruction.at("i_out").get<int>());
            const Precision path_weight =
                instruction.at("path_weight").get<Precision>();
            auto& combined = input_block.l == 0
                ? scalar_weights : vector_weights;
            const int packed_offset = input_block.l == 0
                ? scalar_block_offsets[input_kind][input_block_index]
                : vector_block_offsets[input_kind][input_block_index];
            if (packed_offset < 0
                || output_block.multiplicity != packed_output_width)
                throw std::invalid_argument(
                    "the admitted M potential embedding block is inconsistent.");
            for (int source = 0; source < input_block.multiplicity; ++source)
                for (int target = 0; target < packed_output_width; ++target)
                    combined[static_cast<std::size_t>(packed_offset + source)
                            * packed_output_width + target] = path_weight
                        * weights[weight_offset
                            + source * packed_output_width + target];
            const auto shape = instruction.at("path_shape")
                .get<std::vector<int>>();
            weight_offset += std::accumulate(
                shape.begin(), shape.end(), 1, std::multiplies<int>());
        }
    };
    append_weights(potential_definition, 0);
    append_weights(node_definition, 1);
    append_weights(charge_definition, 2);

    scalar_input_sources_ =
        Kokkos::View<PolarPotentialEmbeddingPackedInputSource*,
            Kokkos::SharedSpace>(
            "polar embedding scalar input sources", packed_input_width);
    vector_input_sources_ =
        Kokkos::View<PolarPotentialEmbeddingPackedInputSource*,
            Kokkos::SharedSpace>(
            "polar embedding vector input sources", packed_input_width);
    for (int index = 0; index < packed_input_width; ++index) {
        scalar_input_sources_(index) = scalar_sources[index];
        vector_input_sources_(index) = vector_sources[index];
    }
    const auto upload_weights = [=] (const char* name,
                                     const std::vector<Precision>& values) {
        Kokkos::View<Precision**, Kokkos::LayoutRight> result(
            Kokkos::view_alloc(std::string(name)), packed_input_width,
            packed_output_width);
        auto host = Kokkos::create_mirror_view(result);
        std::copy(values.begin(), values.end(), host.data());
        Kokkos::deep_copy(result, host);
        return result;
    };
    scalar_weights_ = upload_weights(
        "polar embedding scalar weights", scalar_weights);
    vector_weights_ = upload_weights(
        "polar embedding vector weights", vector_weights);
}

template<typename Precision>
void PolarPotentialEmbeddingKokkosT<Precision>::prepare(const int nodes)
{
    if (use_composed_schedule_) {
        if (nodes < 0 || nodes > std::numeric_limits<int>::max() / 4)
            throw std::length_error(
                "the polar embedding packed row extent exceeds INT_MAX.");
        if (composed_workspace_nodes_ == nodes && packed_input_.data())
            return;
        packed_input_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
            "polar embedding packed input", 4 * nodes, 518);
        packed_output_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
            "polar embedding packed output", 4 * nodes, 512);
        composed_workspace_nodes_ = nodes;
        return;
    }
    if (static_cast<int>(potential_part_.extent(0)) == nodes
        && potential_part_.extent(1) != 0)
        return;
    const int output = output_dimension();
    potential_part_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar embedding potential", nodes, output);
    node_part_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar embedding node features", nodes, output);
    charge_part_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar embedding charges", nodes, output);
    potential_part_adjoint_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar embedding potential adjoint", nodes, output);
    node_part_adjoint_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar embedding node adjoint", nodes, output);
    charge_part_adjoint_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar embedding charge adjoint", nodes, output);
}

template<typename Precision>
void PolarPotentialEmbeddingKokkosT<Precision>::evaluate(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> potential_features,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> node_features,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> charges,
    Kokkos::View<Precision**, Kokkos::LayoutRight> output)
{
    require_rows<Precision>(output, output_dimension(), "embedding output");
    if (use_composed_schedule_) {
        require_rows<Precision>(potential_features, potential_input_dimension(),
            "embedding potential features");
        require_rows<Precision>(node_features, node_input_dimension(),
            "embedding node features");
        require_rows<Precision>(charges, charge_input_dimension(),
            "embedding charges");
        if (potential_features.extent(0) != output.extent(0)
            || node_features.extent(0) != output.extent(0)
            || charges.extent(0) != output.extent(0))
            throw std::invalid_argument(
                "polar delta embedding batch dimensions are inconsistent.");
        if (output.extent(0) == 0)
            return;
        if (output.extent(0)
            > static_cast<std::size_t>(std::numeric_limits<int>::max() / 4))
            throw std::length_error(
                "the polar embedding packed row extent exceeds INT_MAX.");
        const int nodes = static_cast<int>(output.extent(0));
        if (composed_workspace_nodes_ != nodes || !packed_input_.data())
            throw std::logic_error(
                "the composed polar embedding workspace is not prepared.");
        auto packed_input = packed_input_;
        auto packed_output = packed_output_;
        auto scalar_sources = scalar_input_sources_;
        auto vector_sources = vector_input_sources_;
        const std::size_t packed_elements = packed_input.size();
        Kokkos::parallel_for(
            "polar embedding composed pack",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(0, packed_elements),
            KOKKOS_LAMBDA(const std::size_t flat) {
                const int row = static_cast<int>(flat / 518);
                const int packed_source = static_cast<int>(flat % 518);
                const bool scalar = row < nodes;
                const int local_row = scalar ? row : row - nodes;
                const int sample = scalar ? local_row : local_row / 3;
                const int component = scalar ? 0 : local_row % 3;
                const auto source = scalar
                    ? scalar_sources(packed_source)
                    : vector_sources(packed_source);
                const int input_element = source.input_offset
                    + source.source * source.component_width + component;
                packed_input(row, packed_source) = source.input_kind == 0
                    ? potential_features(sample, input_element)
                    : source.input_kind == 1
                        ? node_features(sample, input_element)
                        : charges(sample, input_element);
            });
        auto scalar_input = Kokkos::subview(
            packed_input, std::make_pair(0, nodes), Kokkos::ALL);
        auto scalar_output = Kokkos::subview(
            packed_output, std::make_pair(0, nodes), Kokkos::ALL);
        auto vector_input = Kokkos::subview(
            packed_input, std::make_pair(nodes, 4 * nodes), Kokkos::ALL);
        auto vector_output = Kokkos::subview(
            packed_output, std::make_pair(nodes, 4 * nodes), Kokkos::ALL);
        KokkosBlas::gemm("N", "N", Precision(1), scalar_input,
            scalar_weights_, Precision(0), scalar_output);
        KokkosBlas::gemm("N", "N", Precision(1), vector_input,
            vector_weights_, Precision(0), vector_output);
        Kokkos::parallel_for(
            "polar embedding composed unpack",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(0, output.size()),
            KOKKOS_LAMBDA(const std::size_t flat) {
                const int sample = static_cast<int>(flat / 2048);
                const int output_element = static_cast<int>(flat % 2048);
                if (output_element < 512)
                    output(sample, output_element) =
                        packed_output(sample, output_element);
                else {
                    const int local = output_element - 512;
                    const int target = local / 3;
                    const int component = local % 3;
                    output(sample, output_element) = packed_output(
                        nodes + sample * 3 + component, target);
                }
            });
        return;
    }
    potential_linear_.evaluate(potential_features, potential_part_);
    node_feats_linear_.evaluate(node_features, node_part_);
    charge_embedding_.evaluate(charges, charge_part_);
    auto potential_part = potential_part_;
    auto node_part = node_part_;
    auto charge_part = charge_part_;
    const std::size_t columns = output.extent(1);
    Kokkos::parallel_for(
        "polar embedding sum",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(0, output.size()),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t row = flat / columns;
            const std::size_t column = flat % columns;
            output(row, column) = potential_part(row, column)
                + node_part(row, column) + charge_part(row, column);
        });
}

template<typename Precision>
void PolarPotentialEmbeddingKokkosT<Precision>::reverse(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**, Kokkos::LayoutRight> potential_adjoint,
    Kokkos::View<Precision**, Kokkos::LayoutRight> node_features_adjoint,
    Kokkos::View<Precision**, Kokkos::LayoutRight> charges_adjoint,
    const bool initialize_adjoints)
{
    if (use_composed_schedule_) {
        require_rows<Precision>(output_adjoint, output_dimension(),
            "embedding output adjoint");
        require_rows<Precision>(potential_adjoint, potential_input_dimension(),
            "embedding potential adjoint");
        require_rows<Precision>(node_features_adjoint, node_input_dimension(),
            "embedding node adjoint");
        require_rows<Precision>(charges_adjoint, charge_input_dimension(),
            "embedding charge adjoint");
        if (potential_adjoint.extent(0) != output_adjoint.extent(0)
            || node_features_adjoint.extent(0) != output_adjoint.extent(0)
            || charges_adjoint.extent(0) != output_adjoint.extent(0))
            throw std::invalid_argument(
                "polar delta embedding reverse batch dimensions are "
                "inconsistent.");
        if (output_adjoint.extent(0) == 0)
            return;
        if (output_adjoint.extent(0)
            > static_cast<std::size_t>(std::numeric_limits<int>::max() / 4))
            throw std::length_error(
                "the polar embedding packed row extent exceeds INT_MAX.");
        const int nodes = static_cast<int>(output_adjoint.extent(0));
        if (composed_workspace_nodes_ != nodes || !packed_input_.data())
            throw std::logic_error(
                "the composed polar embedding workspace is not prepared.");
        auto packed_input = packed_input_;
        auto packed_output = packed_output_;
        auto scalar_sources = scalar_input_sources_;
        auto vector_sources = vector_input_sources_;
        Kokkos::parallel_for(
            "polar embedding composed reverse pack",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(0, packed_output.size()),
            KOKKOS_LAMBDA(const std::size_t flat) {
                const int row = static_cast<int>(flat / 512);
                const int target = static_cast<int>(flat % 512);
                if (row < nodes)
                    packed_output(row, target) =
                        output_adjoint(row, target);
                else {
                    const int local_row = row - nodes;
                    const int sample = local_row / 3;
                    const int component = local_row % 3;
                    packed_output(row, target) = output_adjoint(
                        sample, 512 + target * 3 + component);
                }
            });
        auto scalar_input = Kokkos::subview(
            packed_input, std::make_pair(0, nodes), Kokkos::ALL);
        auto scalar_output = Kokkos::subview(
            packed_output, std::make_pair(0, nodes), Kokkos::ALL);
        auto vector_input = Kokkos::subview(
            packed_input, std::make_pair(nodes, 4 * nodes), Kokkos::ALL);
        auto vector_output = Kokkos::subview(
            packed_output, std::make_pair(nodes, 4 * nodes), Kokkos::ALL);
        KokkosBlas::gemm("N", "T", Precision(1), scalar_output,
            scalar_weights_, Precision(0), scalar_input);
        KokkosBlas::gemm("N", "T", Precision(1), vector_output,
            vector_weights_, Precision(0), vector_input);
        Kokkos::parallel_for(
            "polar embedding composed reverse unpack",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(0, packed_input.size()),
            KOKKOS_LAMBDA(const std::size_t flat) {
                const int row = static_cast<int>(flat / 518);
                const int packed_source = static_cast<int>(flat % 518);
                const bool scalar = row < nodes;
                const int local_row = scalar ? row : row - nodes;
                const int sample = scalar ? local_row : local_row / 3;
                const int component = scalar ? 0 : local_row % 3;
                const auto source = scalar
                    ? scalar_sources(packed_source)
                    : vector_sources(packed_source);
                const int input_element = source.input_offset
                    + source.source * source.component_width + component;
                const Precision value = packed_input(row, packed_source);
                if (source.input_kind == 0) {
                    if (initialize_adjoints)
                        potential_adjoint(sample, input_element) = value;
                    else
                        potential_adjoint(sample, input_element) += value;
                } else if (source.input_kind == 1) {
                    if (initialize_adjoints)
                        node_features_adjoint(sample, input_element) = value;
                    else
                        node_features_adjoint(sample, input_element) += value;
                } else {
                    if (initialize_adjoints)
                        charges_adjoint(sample, input_element) = value;
                    else
                        charges_adjoint(sample, input_element) += value;
                }
            });
        return;
    }
    potential_linear_.reverse(output_adjoint, potential_adjoint, initialize_adjoints);
    node_feats_linear_.reverse(
        output_adjoint, node_features_adjoint, initialize_adjoints);
    charge_embedding_.reverse(output_adjoint, charges_adjoint, initialize_adjoints);
}

} // namespace symmetrix::mace_polar

#include "mace_polar_deltas_chain.tpp"
