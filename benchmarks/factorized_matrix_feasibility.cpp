#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>

namespace {

using View3 = Kokkos::View<float***, Kokkos::LayoutRight>;
using HostView3 = Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::HostSpace>;
using TeamMember = Kokkos::TeamPolicy<>::member_type;

struct Options {
    int tasks = 864;
    int neighbors = 91;
    int embedding = 64;
    int columns = 128;
    int batch_tasks = 32;
    int warmups = 2;
    int repeats = 5;
};

int parse_positive(const char* value, const char* option)
{
    const long parsed = std::strtol(value, nullptr, 10);
    if (parsed <= 0 || parsed > std::numeric_limits<int>::max())
        throw std::invalid_argument(std::string(option) + " must be a positive integer");
    return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int index=1; index<argc; index += 2) {
        if (index+1 >= argc)
            throw std::invalid_argument(std::string("missing value for ") + argv[index]);
        const std::string option = argv[index];
        const int value = parse_positive(argv[index+1], argv[index]);
        if (option == "--tasks") options.tasks = value;
        else if (option == "--neighbors") options.neighbors = value;
        else if (option == "--embedding") options.embedding = value;
        else if (option == "--columns") options.columns = value;
        else if (option == "--batch-tasks") options.batch_tasks = value;
        else if (option == "--warmups") options.warmups = value;
        else if (option == "--repeats") options.repeats = value;
        else throw std::invalid_argument("unknown option " + option);
    }
    options.batch_tasks = std::min(options.batch_tasks, options.tasks);
    return options;
}

enum class Route {
    scalar,
    batched_blocked,
    batched_unblocked,
    kokkos_blas,
};

const char* route_name(Route route)
{
    switch (route) {
    case Route::scalar: return "scalar";
    case Route::batched_blocked: return "kokkos_batched_blocked";
    case Route::batched_unblocked: return "kokkos_batched_unblocked";
    case Route::kokkos_blas: return "kokkos_blas_tpl";
    }
    return "unknown";
}

struct Workspace {
    View3 radial;
    View3 coupling;
    View3 aggregate;
    View3 aggregate_adjoint;
    View3 coupling_adjoint;
    View3 radial_adjoint;

    Workspace(const Options& options)
        : radial("factorized radial", options.batch_tasks, options.neighbors, options.embedding),
          coupling("factorized coupling", options.batch_tasks, options.neighbors, options.columns),
          aggregate("factorized aggregate", options.batch_tasks, options.embedding, options.columns),
          aggregate_adjoint(
              "factorized aggregate adjoint", options.batch_tasks, options.embedding, options.columns),
          coupling_adjoint(
              "factorized coupling adjoint", options.batch_tasks, options.neighbors, options.columns),
          radial_adjoint(
              "factorized radial adjoint", options.batch_tasks, options.neighbors, options.embedding)
    {}

    std::size_t bytes() const
    {
        return sizeof(float) * (
            radial.size()+coupling.size()+aggregate.size()+aggregate_adjoint.size()
            +coupling_adjoint.size()+radial_adjoint.size());
    }
};

struct Inputs {
    View3 radial;
    View3 coupling;
    View3 aggregate_adjoint;

    Inputs(const Options& options)
        : radial("radial inputs", options.tasks, options.neighbors, options.embedding),
          coupling("coupling inputs", options.tasks, options.neighbors, options.columns),
          aggregate_adjoint(
              "aggregate adjoint inputs", options.tasks, options.embedding, options.columns)
    {}
};

struct HostOutputs {
    HostView3 aggregate;
    HostView3 coupling_adjoint;
    HostView3 radial_adjoint;
};

struct ErrorReport {
    double aggregate = 0.0;
    double coupling_adjoint = 0.0;
    double radial_adjoint = 0.0;
};

void initialize_inputs(Inputs& inputs)
{
    const auto radial_view = inputs.radial;
    const auto coupling_view = inputs.coupling;
    const auto adjoint_view = inputs.aggregate_adjoint;
    Kokkos::parallel_for(
        "initialize factorized radial inputs",
        inputs.radial.size(),
        KOKKOS_LAMBDA (const std::size_t flat) {
            radial_view.data()[flat] =
                0.01f * static_cast<float>(static_cast<int>(flat % 31) - 15);
        });
    Kokkos::parallel_for(
        "initialize factorized coupling inputs",
        inputs.coupling.size(),
        KOKKOS_LAMBDA (const std::size_t flat) {
            coupling_view.data()[flat] =
                0.02f * static_cast<float>(static_cast<int>(flat % 29) - 14);
        });
    Kokkos::parallel_for(
        "initialize factorized aggregate adjoints",
        inputs.aggregate_adjoint.size(),
        KOKKOS_LAMBDA (const std::size_t flat) {
            adjoint_view.data()[flat] =
                0.015f * static_cast<float>(static_cast<int>(flat % 23) - 11);
        });
    Kokkos::fence();
}

void pack_chunk(
    const Options& options,
    const Inputs& inputs,
    Workspace& workspace,
    int task_begin,
    int task_count)
{
    const auto radial_input = inputs.radial;
    const auto coupling_input = inputs.coupling;
    const auto adjoint_input = inputs.aggregate_adjoint;
    const auto radial = workspace.radial;
    const auto coupling = workspace.coupling;
    const auto aggregate_adjoint = workspace.aggregate_adjoint;
    const std::size_t radial_per_task =
        static_cast<std::size_t>(options.neighbors)*options.embedding;
    const std::size_t coupling_per_task =
        static_cast<std::size_t>(options.neighbors)*options.columns;
    const std::size_t aggregate_per_task =
        static_cast<std::size_t>(options.embedding)*options.columns;
    const std::size_t total = static_cast<std::size_t>(task_count)
        *(radial_per_task+coupling_per_task+aggregate_per_task);
    Kokkos::parallel_for(
        "pack factorized matrix inputs",
        total,
        KOKKOS_LAMBDA (const std::size_t flat) {
            const std::size_t task_stride =
                radial_per_task+coupling_per_task+aggregate_per_task;
            const int local_task = flat/task_stride;
            const std::size_t within = flat%task_stride;
            const int global_task = task_begin+local_task;
            if (within < radial_per_task) {
                radial.data()[local_task*radial_per_task+within] =
                    radial_input.data()[global_task*radial_per_task+within];
            } else if (within < radial_per_task+coupling_per_task) {
                const std::size_t offset = within-radial_per_task;
                coupling.data()[local_task*coupling_per_task+offset] =
                    coupling_input.data()[global_task*coupling_per_task+offset];
            } else {
                const std::size_t offset = within-radial_per_task-coupling_per_task;
                aggregate_adjoint.data()[local_task*aggregate_per_task+offset] =
                    adjoint_input.data()[global_task*aggregate_per_task+offset];
            }
        });
    Kokkos::deep_copy(workspace.aggregate, 0.0f);
    Kokkos::deep_copy(workspace.coupling_adjoint, 0.0f);
    Kokkos::deep_copy(workspace.radial_adjoint, 0.0f);
}

void invoke_scalar(const Options& options, Workspace& workspace, int task_count)
{
    const auto radial = workspace.radial;
    const auto coupling = workspace.coupling;
    const auto aggregate_adjoint = workspace.aggregate_adjoint;
    const auto aggregate = workspace.aggregate;
    const auto coupling_adjoint = workspace.coupling_adjoint;
    const auto radial_adjoint = workspace.radial_adjoint;
    Kokkos::parallel_for(
        "factorized scalar forward transpose",
        Kokkos::TeamPolicy<>(task_count, Kokkos::AUTO),
        KOKKOS_LAMBDA (const TeamMember& member) {
            const int task = member.league_rank();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, options.embedding*options.columns),
                [=] (const int flat) {
                    const int embedding = flat/options.columns;
                    const int column = flat%options.columns;
                    float value = 0.0f;
                    for (int neighbor=0; neighbor<options.neighbors; ++neighbor)
                        value += radial(task,neighbor,embedding)
                            *coupling(task,neighbor,column);
                    aggregate(task,embedding,column) = value;
                });
            member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, options.neighbors*options.columns),
                [=] (const int flat) {
                    const int neighbor = flat/options.columns;
                    const int column = flat%options.columns;
                    float value = 0.0f;
                    for (int embedding=0; embedding<options.embedding; ++embedding)
                        value += radial(task,neighbor,embedding)
                            *aggregate_adjoint(task,embedding,column);
                    coupling_adjoint(task,neighbor,column) = value;
                });
            member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, options.neighbors*options.embedding),
                [=] (const int flat) {
                    const int neighbor = flat/options.embedding;
                    const int embedding = flat%options.embedding;
                    float value = 0.0f;
                    for (int column=0; column<options.columns; ++column)
                        value += coupling(task,neighbor,column)
                            *aggregate_adjoint(task,embedding,column);
                    radial_adjoint(task,neighbor,embedding) = value;
                });
        });
}

template <typename Algorithm>
void invoke_batched(const Options& options, Workspace& workspace, int task_count)
{
    const auto radial = workspace.radial;
    const auto coupling = workspace.coupling;
    const auto aggregate_adjoint = workspace.aggregate_adjoint;
    const auto aggregate = workspace.aggregate;
    const auto coupling_adjoint = workspace.coupling_adjoint;
    const auto radial_adjoint = workspace.radial_adjoint;
    Kokkos::parallel_for(
        "factorized batched forward transpose",
        Kokkos::TeamPolicy<>(task_count, Kokkos::AUTO),
        KOKKOS_LAMBDA (const TeamMember& member) {
            const int task = member.league_rank();
            const auto radial_task = Kokkos::subview(radial, task, Kokkos::ALL, Kokkos::ALL);
            const auto coupling_task =
                Kokkos::subview(coupling, task, Kokkos::ALL, Kokkos::ALL);
            const auto aggregate_task =
                Kokkos::subview(aggregate, task, Kokkos::ALL, Kokkos::ALL);
            const auto aggregate_adjoint_task =
                Kokkos::subview(aggregate_adjoint, task, Kokkos::ALL, Kokkos::ALL);
            const auto coupling_adjoint_task =
                Kokkos::subview(coupling_adjoint, task, Kokkos::ALL, Kokkos::ALL);
            const auto radial_adjoint_task =
                Kokkos::subview(radial_adjoint, task, Kokkos::ALL, Kokkos::ALL);
            KokkosBatched::TeamGemm<
                TeamMember,
                KokkosBatched::Trans::Transpose,
                KokkosBatched::Trans::NoTranspose,
                Algorithm>::invoke(
                    member, 1.0f, radial_task, coupling_task, 0.0f, aggregate_task);
            member.team_barrier();
            KokkosBatched::TeamGemm<
                TeamMember,
                KokkosBatched::Trans::NoTranspose,
                KokkosBatched::Trans::NoTranspose,
                Algorithm>::invoke(
                    member, 1.0f, radial_task, aggregate_adjoint_task,
                    0.0f, coupling_adjoint_task);
            member.team_barrier();
            KokkosBatched::TeamGemm<
                TeamMember,
                KokkosBatched::Trans::NoTranspose,
                KokkosBatched::Trans::Transpose,
                Algorithm>::invoke(
                    member, 1.0f, coupling_task, aggregate_adjoint_task,
                    0.0f, radial_adjoint_task);
        });
}

void invoke_kokkos_blas(Workspace& workspace, int task_count)
{
    for (int task=0; task<task_count; ++task) {
        const auto radial = Kokkos::subview(
            workspace.radial, task, Kokkos::ALL, Kokkos::ALL);
        const auto coupling = Kokkos::subview(
            workspace.coupling, task, Kokkos::ALL, Kokkos::ALL);
        const auto aggregate = Kokkos::subview(
            workspace.aggregate, task, Kokkos::ALL, Kokkos::ALL);
        const auto aggregate_adjoint = Kokkos::subview(
            workspace.aggregate_adjoint, task, Kokkos::ALL, Kokkos::ALL);
        const auto coupling_adjoint = Kokkos::subview(
            workspace.coupling_adjoint, task, Kokkos::ALL, Kokkos::ALL);
        const auto radial_adjoint = Kokkos::subview(
            workspace.radial_adjoint, task, Kokkos::ALL, Kokkos::ALL);
        KokkosBlas::gemm("T", "N", 1.0f, radial, coupling, 0.0f, aggregate);
        KokkosBlas::gemm(
            "N", "N", 1.0f, radial, aggregate_adjoint, 0.0f, coupling_adjoint);
        KokkosBlas::gemm(
            "N", "T", 1.0f, coupling, aggregate_adjoint, 0.0f, radial_adjoint);
    }
}

void invoke_route(Route route, const Options& options, Workspace& workspace, int task_count)
{
    switch (route) {
    case Route::scalar:
        invoke_scalar(options, workspace, task_count);
        return;
    case Route::batched_blocked:
        invoke_batched<KokkosBatched::Algo::Gemm::Blocked>(
            options, workspace, task_count);
        return;
    case Route::batched_unblocked:
        invoke_batched<KokkosBatched::Algo::Gemm::Unblocked>(
            options, workspace, task_count);
        return;
    case Route::kokkos_blas:
        invoke_kokkos_blas(workspace, task_count);
        return;
    }
}

double evaluate_once(
    Route route,
    const Options& options,
    const Inputs& inputs,
    Workspace& workspace)
{
    Kokkos::Timer timer;
    for (int task_begin=0; task_begin<options.tasks; task_begin += options.batch_tasks) {
        const int task_count = std::min(options.batch_tasks, options.tasks-task_begin);
        pack_chunk(options, inputs, workspace, task_begin, task_count);
        invoke_route(route, options, workspace, task_count);
        Kokkos::fence();
    }
    return 1000.0*timer.seconds();
}

double median(std::vector<double> samples)
{
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size()/2;
    if (samples.size()%2)
        return samples[middle];
    return 0.5*(samples[middle-1]+samples[middle]);
}

HostOutputs copy_outputs_to_host(const Workspace& workspace)
{
    return {
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), workspace.aggregate),
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), workspace.coupling_adjoint),
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), workspace.radial_adjoint),
    };
}

double max_abs_difference(const HostView3& lhs, const HostView3& rhs)
{
    if (lhs.size() != rhs.size())
        throw std::invalid_argument("Factorized validation output extents do not match.");
    double result = 0.0;
    for (std::size_t index=0; index<lhs.size(); ++index)
        result = std::max(
            result,
            std::abs(static_cast<double>(lhs.data()[index])-rhs.data()[index]));
    return result;
}

ErrorReport compare_route_to_scalar(
    Route route,
    const Options& options,
    const Inputs& inputs,
    Workspace& workspace,
    const HostOutputs& reference)
{
    const int task_count = std::min(options.batch_tasks, options.tasks);
    pack_chunk(options, inputs, workspace, 0, task_count);
    invoke_route(route, options, workspace, task_count);
    Kokkos::fence();
    const auto actual = copy_outputs_to_host(workspace);
    return {
        max_abs_difference(actual.aggregate, reference.aggregate),
        max_abs_difference(actual.coupling_adjoint, reference.coupling_adjoint),
        max_abs_difference(actual.radial_adjoint, reference.radial_adjoint),
    };
}

struct Result {
    Route route;
    ErrorReport validation;
    std::vector<double> samples_ms;
};

Result benchmark_route(
    Route route,
    ErrorReport validation,
    const Options& options,
    const Inputs& inputs,
    Workspace& workspace)
{
    for (int warmup=0; warmup<options.warmups; ++warmup)
        static_cast<void>(evaluate_once(route, options, inputs, workspace));
    Result result{route, validation, {}};
    result.samples_ms.reserve(options.repeats);
    for (int repeat=0; repeat<options.repeats; ++repeat)
        result.samples_ms.push_back(evaluate_once(route, options, inputs, workspace));
    return result;
}

}  // namespace

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    int status = 0;
    try {
        const auto options = parse_options(argc, argv);
        Inputs inputs(options);
        initialize_inputs(inputs);
        Workspace workspace(options);
        const std::vector<Route> routes = {
            Route::scalar,
            Route::batched_blocked,
            Route::batched_unblocked,
            Route::kokkos_blas,
        };
        const int validation_tasks = std::min(options.batch_tasks, options.tasks);
        pack_chunk(options, inputs, workspace, 0, validation_tasks);
        invoke_route(Route::scalar, options, workspace, validation_tasks);
        Kokkos::fence();
        const auto reference = copy_outputs_to_host(workspace);
        auto results = std::vector<Result>();
        results.reserve(routes.size());
        for (const auto route : routes) {
            const auto validation = compare_route_to_scalar(
                route, options, inputs, workspace, reference);
            constexpr double tolerance = 2.0e-3;
            if (validation.aggregate > tolerance
                || validation.coupling_adjoint > tolerance
                || validation.radial_adjoint > tolerance)
                throw std::runtime_error(
                    std::string(route_name(route))
                    + " does not match the scalar forward/transpose reference.");
            results.push_back(benchmark_route(
                route, validation, options, inputs, workspace));
        }

        const double scalar_median = median(results.front().samples_ms);
        std::cout << std::setprecision(10);
        std::cout << "{\n"
                  << "  \"execution_space\": \"" << Kokkos::DefaultExecutionSpace::name()
                  << "\",\n"
                  << "  \"shape\": {\"tasks\": " << options.tasks
                  << ", \"neighbors\": " << options.neighbors
                  << ", \"embedding\": " << options.embedding
                  << ", \"columns\": " << options.columns << "},\n"
                  << "  \"batch_tasks\": " << options.batch_tasks << ",\n"
                  << "  \"workspace_bytes\": " << workspace.bytes() << ",\n"
                  << "  \"includes\": [\"packing\", \"zeroing\", \"forward\", "
                     "\"transpose_coupling\", \"transpose_radial\", \"synchronization\"],\n"
                  << "  \"routes\": [\n";
        for (std::size_t index=0; index<results.size(); ++index) {
            const auto& result = results[index];
            const double route_median = median(result.samples_ms);
            std::cout << "    {\"name\": \"" << route_name(result.route)
                      << "\", \"median_ms\": " << route_median
                      << ", \"speedup_vs_scalar\": " << scalar_median/route_median
                      << ", \"max_abs_error\": {\"forward\": "
                      << result.validation.aggregate
                      << ", \"transpose_coupling\": "
                      << result.validation.coupling_adjoint
                      << ", \"transpose_radial\": "
                      << result.validation.radial_adjoint << "}"
                      << ", \"samples_ms\": [";
            for (std::size_t sample=0; sample<result.samples_ms.size(); ++sample) {
                if (sample) std::cout << ", ";
                std::cout << result.samples_ms[sample];
            }
            std::cout << "]}" << (index+1 == results.size() ? "\n" : ",\n");
        }
        std::cout << "  ]\n}\n";
    } catch (const std::exception& error) {
        std::cerr << "factorized_matrix_feasibility: " << error.what() << '\n';
        status = 1;
    }
    Kokkos::finalize();
    return status;
}
