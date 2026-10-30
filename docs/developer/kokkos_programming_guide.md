# Kokkos programming patterns

This is a how-to guide for writing Kokkos code in Symmetrix-XL. It explains the
patterns already used by the repository and how to choose among them when
adding a numerical stage. It assumes that you are comfortable with Python,
NumPy, and concurrent code, but may be new to C++ performance portability.

## What Kokkos is and what problem it solves

[Kokkos](https://kokkos.org/) is an open-source C++ performance-portability
programming model. It provides coupled abstractions for **where and how work
runs**—execution spaces, policies, and parallel operations—and **where and how
data is stored**—memory spaces, multidimensional views, and layouts. A
scientific kernel can therefore express its parallel work and data-access
requirements once, while a build selects an implementation for a multicore CPU
or an accelerator such as an NVIDIA or AMD GPU.

The problem is not merely that CPUs and GPUs use different launch syntax. They
also expose different parallel hierarchies, preferred memory layouts, address
spaces, synchronization behavior, and optimized libraries. Maintaining an
OpenMP implementation beside separate CUDA and HIP implementations would
duplicate scientific logic and let those versions drift. Kokkos supplies a
common model while retaining explicit control over iteration policies, memory
placement, transfers, and synchronization—the details needed for performance.

Symmetrix-XL uses Kokkos for its native model evaluation, graph construction,
force and stress reductions, and LAMMPS integration. The same source structure
can be compiled for the repository's CPU, CUDA, or HIP configurations, while
backend-specific implementations remain available where measurement shows that
a portable route is insufficient. Kokkos is not an automatic loop translator:
developers must still define write ownership, choose a policy and layout, and
establish correct data-movement and synchronization boundaries. Those choices
are the patterns explained in this guide.

### Official resources

Use this guide for Symmetrix-XL conventions and the upstream material for the
complete API and programming model:

- [Kokkos project website](https://kokkos.org/) — ecosystem overview and
  project entry point.
- [Kokkos Core documentation](https://kokkos.org/kokkos-core-wiki/) — current
  programming guide and API reference.
- [Programming-model overview](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ProgrammingModel.html)
  — execution spaces, patterns, policies, memory spaces, layouts, and traits.
- [Views chapter](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/View.html)
  — multidimensional arrays, layouts, memory spaces, mirrors, and unmanaged
  memory.
- [Parallel dispatch chapter](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html)
  — `parallel_for`, `parallel_reduce`, `parallel_scan`, functors, and lambdas.
- [Hierarchical parallelism chapter](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/HierarchicalParallelism.html)
  — teams, nested policies, vector lanes, and scratch memory.
- [Execution-policy API reference](https://kokkos.org/kokkos-core-wiki/API/core/Execution-Policies.html)
  — `RangePolicy`, `MDRangePolicy`, `TeamPolicy`, and nested policy details.
- [Official Kokkos tutorials](https://github.com/kokkos/kokkos-tutorials) —
  lectures, worked examples, and exercises.
- [Kokkos Core source repository](https://github.com/kokkos/kokkos) — releases,
  build documentation, issues, and implementation source.
- [Kokkos Kernels documentation](https://kokkos.org/kokkos-kernels/docs/index.html)
  — portable dense, batched, sparse, and graph kernels used alongside Kokkos
  Core.

For a new contributor, the most useful order is the programming-model overview,
the Views chapter, parallel dispatch, and hierarchical parallelism. Return to
the API reference when implementing a particular policy rather than trying to
read it end to end.

### Learn by changing a small kernel

The upstream tutorials are most useful when they are treated as short, compiled
experiments rather than reference material. Start with the exercises on parallel
dispatch and views, then make one controlled change at a time: change the
execution space, make a view explicit `LayoutRight`, or replace a flat loop with
an `MDRangePolicy`. Build and run the same exercise on the CPU and, when
available, a GPU. The point is to see that Kokkos preserves the algorithmic
contract while the mapping and performance characteristics change.

Do not copy their CMake commands into this repository: Symmetrix-XL owns its
Kokkos configuration and build through `tools/symmetrix_build.py`. Copy the
programming pattern, then implement it against the repository's existing
execution-space, allocation, and error-handling conventions.

## The mental model

The shortest useful translation from NumPy to Kokkos is:

| Python or NumPy idea | Kokkos equivalent in this repository |
|---|---|
| `numpy.ndarray` | `Kokkos::View` |
| `a.shape[k]` | `a.extent(k)` |
| `a.size` | `a.size()` |
| C-contiguous array | `Kokkos::View<T**, Kokkos::LayoutRight>` |
| `a[start:stop]` | `Kokkos::subview(a, Kokkos::make_pair(start, stop))` |
| vectorized elementwise expression | `Kokkos::parallel_for` |
| `sum(...)` | `Kokkos::parallel_reduce` |
| `cumsum(...)` used to form offsets | `Kokkos::parallel_scan` |
| temporary array local to a work group | team scratch memory |
| indexed addition such as `np.add.at` | `Kokkos::atomic_add` or a structured reduction |
| CPU/GPU synchronization | `execution_space.fence()` or `Kokkos::fence()` |
| transfer between CPU and accelerator | mirror plus `Kokkos::deep_copy` |

The important difference is that a `View` has a memory space and a layout in
addition to a dtype and shape. A kernel has an execution space. Code is valid
only when that execution space can access every view it touches.

Make this an explicit design check at an interface between components. The
following assertion documents the contract and turns an accidental host/device
view mismatch into a clear compile-time diagnostic:

```cpp
template<class ExecutionSpace, class View>
void consume_on(ExecutionSpace, const View&) {
    static_assert(Kokkos::SpaceAccessibility<
        ExecutionSpace, typename View::memory_space>::accessible,
        "execution space cannot access this view's memory space");
}
```

This does not replace testing: accessibility says that an access is legal, not
that the view's layout or access pattern is efficient.

Kokkos is selected when the extension is built. A single installed extension
has one `Kokkos::DefaultExecutionSpace`; Python does not switch an installed
extension between CPU, CUDA, and HIP at runtime. Build and qualify separate
extensions for those backends.

### Choose from write ownership, not loop shape

Before choosing an API, ask which worker owns each output element. That answer
usually selects the pattern:

| Data flow | Repository pattern |
|---|---|
| Each iteration writes a different element or row | `RangePolicy` or `MDRangePolicy` |
| One outer item has cooperating inner work | `TeamPolicy` plus a nested range |
| Many values produce one scalar or one value per team | `parallel_reduce` |
| Counts must become offsets | count, `parallel_scan`, then fill |
| Irregular inputs update shared destinations | atomics, or restructure as a segmented reduction |
| A whole dense array is a matrix operation | KokkosBlas; KokkosBatched or team loops inside workers |
| Host or Python data enters or leaves the device pipeline | unmanaged host view, mirror, and `deep_copy` |

Do not begin by translating nested Python `for` loops literally. First identify
the logical output item, whether writes overlap, and which temporary data can be
reused. Kokkos policies express that ownership and cooperation.

## Where Kokkos lives in this repository

The main layers are:

| Area | Role | Good starting points |
|---|---|---|
| Runtime utilities | initialization, finalization, copies, host conversion, scratch admission | `libsymmetrix/source/tools_kokkos.{hpp,cpp}` |
| Small numerical components | focused examples of views and kernels | `cubic_spline_kokkos.cpp`, `zbl_kokkos.cpp`, `multilayer_perceptron_kokkos.cpp` |
| Graph construction | count/scan/fill algorithms and growable topology | `neighbor_graph_kokkos.cpp`, `dispersion_graph_kokkos.cpp` |
| Model evaluators | retained state, execution streams, reductions, and backend specialization | `mace_kokkos*.{hpp,cpp}`, `mace_nonlinear_kokkos*`, `oam_kokkos*` |
| Python boundary | NumPy validation, upload, execution, and result download | `symmetrix/source/cpp/utilities_kokkos.hpp`, `symmetrix/source/cpp/*kokkos.cpp` |
| LAMMPS integration | dual views, communication, MPI ownership, and device-aware execution | `pair_symmetrix/pair_symmetrix_mace_kokkos.{h,cpp}` |

The MACE evaluator is large enough to be partitioned by responsibility. Read
{doc}`/mace_kokkos_source_layout` before moving methods or shared state between
its translation units.

## A first repository-style kernel

This small function contains the conventions expected in a new portable stage:
host validation, an explicit execution-space instance, growable output, widened
flattened indexing, local view handles, a named launch, and an optional
completion contract.

```cpp
void scale_rows(
    const Kokkos::DefaultExecutionSpace& execution_space,
    Kokkos::View<const double**, Kokkos::LayoutRight> input,
    Kokkos::View<double**, Kokkos::LayoutRight>& output,
    const double scale,
    const bool completion_fence)
{
    const std::size_t rows = input.extent(0);
    const std::size_t columns = input.extent(1);
    if (rows != 0
            && columns > std::numeric_limits<std::size_t>::max()/rows)
        throw std::length_error("scale_rows extent overflow");
    const std::size_t count = rows*columns;

    if (output.extent(0) != rows || output.extent(1) != columns)
        Kokkos::realloc(
            Kokkos::WithoutInitializing, output, rows, columns);

    const auto source = input;
    const auto destination = output;
    using Policy = Kokkos::RangePolicy<
        Kokkos::DefaultExecutionSpace,
        Kokkos::IndexType<std::size_t>>;
    Kokkos::parallel_for(
        "scale rows",
        Policy(execution_space, 0, count),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t row = flat/columns;
            const std::size_t column = flat%columns;
            destination(row,column) = scale*source(row,column);
        });

    if (completion_fence)
        execution_space.fence("scale rows completion");
}
```

This is a template, not a demand that every kernel allocate output or expose a
fence flag. Evaluator-owned methods often admit workspace earlier and remain
asynchronous through several stages. Preserve the surrounding component's
lifecycle and execution-space contract.

## Tutorial: derive a Kokkos function from `CubicSplineKokkos`

This walkthrough starts with an existing, intentionally small repository
function: `CubicSplineKokkos::generate_coefficients` in
`libsymmetrix/source/cubic_spline_kokkos.cpp`. It turns knot values and first
derivatives into four cubic coefficients per interval. The function is a good
first reading target because the mathematics is local, while its construction,
data transfer, kernel launch, and Python test show the full path a small native
feature follows.

### 1. State the contract before writing the kernel

For `n` knots there are `n - 1` intervals. The constructor validates `n >= 2`,
matching value/derivative lengths, and a finite positive spacing `h`. It then
allocates `4 * (n - 1)` coefficients. For interval `i`, the kernel must write
only `c(4*i)` through `c(4*i + 3)` from `values(i)`, `values(i + 1)`, and the
matching derivatives.

That last sentence is the parallelization proof: interval `i` owns four output
slots that no other interval writes. It needs neither an atomic nor a reduction,
so one `RangePolicy` iteration per interval is the simplest correct mapping.
Do this ownership exercise before thinking about team size or GPU tuning.

### 2. Keep ownership in view handles, not host pointers

The vector-taking overload wraps its `std::vector` inputs in unmanaged host
views, allocates execution-space views, and deep-copies before it launches the
view-taking overload. The vectors are borrowed only for the upload; the spline
retains its owned `c` view. That is the same lifetime rule used at the NumPy
boundary: a kernel may capture a Kokkos view handle, but must not retain a
pointer into caller-owned memory.

### 3. Express the ownership proof as a kernel

The repository implementation is deliberately compact. Written in the more
explicit style expected of a new pipeline stage, its central loop is:

```cpp
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using Policy = Kokkos::RangePolicy<ExecutionSpace>;

void generate_coefficients(
    const ExecutionSpace& execution_space,
    const double h,
    Kokkos::View<const double*> values,
    Kokkos::View<const double*> derivs,
    Kokkos::View<double*> coefficients)
{
    const int intervals = static_cast<int>(values.extent(0)) - 1;
    const auto c = coefficients;
    Kokkos::parallel_for(
        "cubic spline coefficients", Policy(execution_space, 0, intervals),
        KOKKOS_LAMBDA(const int i) {
            const int base = 4*i;
            c(base) = values(i);
            c(base + 1) = derivs(i);
            c(base + 2) = (-3.0*values(i) - 2.0*h*derivs(i)
                + 3.0*values(i + 1) - h*derivs(i + 1))/(h*h);
            c(base + 3) = (2.0*values(i) + h*derivs(i)
                - 2.0*values(i + 1) + h*derivs(i + 1))/(h*h*h);
        });
}
```

This is a teaching extraction, not a second implementation to add beside the
existing overloads. The production code establishes the preconditions in the
constructor and is intentionally limited to a small, validated spline. A new
general-purpose function must validate its own extents and use `std::size_t` or
an explicit 64-bit policy index if its launch count could exceed `int`.

### 4. Prove the feature at the public boundary

`symmetrix/test/test_cubic_spline_kokkos.py` compares values and derivatives
against SciPy's `CubicSpline`, and verifies invalid constructor input and
out-of-range evaluation. Follow that shape for a new function:

1. Test the scientific result against an independent oracle or a small exact
   hand calculation.
2. Test empty/minimal and invalid host inputs before the kernel is launched.
3. Test the binding surface that a user invokes, not only a private helper.
4. Rebuild and repeat on every available intended execution backend; a passing
   serial test does not establish CUDA or HIP device-callability.

Once this example is comfortable, `neighbor_graph_kokkos.cpp` is the next
step: its count/scan/fill pipeline adds irregular output and atomics without
changing the same core process of stating ownership, choosing a policy, and
testing the public result.

## Views: arrays with an address-space contract

A view declaration encodes value type, rank, layout, and optionally memory
space:

```cpp
Kokkos::View<double*> values("values", count);                    // rank 1
Kokkos::View<double**, Kokkos::LayoutRight> matrix("matrix", n, k);
Kokkos::View<double*[3]> positions("positions", atoms);           // fixed inner extent
Kokkos::View<const int*> node_types;                              // read-only elements
```

`LayoutRight` makes the rightmost index contiguous, matching a C-contiguous
NumPy array. Use it when a buffer is exchanged with Python or row-major BLAS,
or when the algorithm explicitly assumes row-major storage. Do not assume that
an unqualified multidimensional `View` has a particular layout on every
backend. Much of the evaluator also uses flat rank-one views and computes its
own offsets to make layout explicit.

Choose a layout from the indices that adjacent workers vary, not from a
preference for rows or columns. On a GPU, arranging that adjacent threads read
adjacent addresses is usually the first locality goal; on a CPU, the relevant
unit may instead be the vectorized inner loop and cache reuse. `LayoutRight`
is therefore an interoperation contract at the Python boundary, not a universal
performance default. Benchmark layout changes with the real access pattern.

A `View` is a small, reference-counted handle. Copying it is shallow:

```cpp
auto output = this->output;  // both handles refer to the same allocation
```

That makes view capture in a kernel cheap. It also means that assigning one
view to another does not copy elements. Use `Kokkos::deep_copy` for data and
use a mirror when the source and destination live in different memory spaces.

The repository commonly passes mutable views as output arguments and const
views as inputs. Preserve that distinction in new APIs; it makes data flow
visible and prevents accidental writes.

### Allocation and capacity

Named allocations appear in profiler output, so use stable, descriptive
labels. Default allocation value-initializes elements. For a buffer that a
kernel will completely overwrite, avoid the initialization cost:

```cpp
auto values = Kokkos::View<double*>(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "stage values"), count);
```

`Kokkos::realloc` may replace the allocation and discard or reinitialize its
contents. The retained evaluators generally use grow-only capacity: reuse a
large enough buffer, otherwise replace it with
`Kokkos::realloc(Kokkos::WithoutInitializing, ...)`. See
`symmetrix_oam_detail::ensure_view_capacity` in `oam_kokkos.hpp`. Launch bounds
describe the active extent; the allocation may be larger.

This is the Kokkos equivalent of maintaining a reusable NumPy workspace rather
than allocating a new array in every function call. It is particularly
important on GPUs and in molecular-dynamics loops.

Only use `WithoutInitializing` when all active elements are written before any
read. Accumulators, counters, and atomic destinations must first be initialized,
usually with `Kokkos::deep_copy(view, 0)`.

### Subviews are borrowed slices

Subviews do not copy:

```cpp
auto active = Kokkos::subview(
    storage, Kokkos::make_pair(std::size_t{0}, active_count));
```

They retain the underlying allocation, but their layout may become
`LayoutStride`. Do not pass an arbitrary multidimensional subview to code that
requires contiguous `LayoutRight` storage without checking its contract.

### Unmanaged views borrow foreign memory

The Python bindings wrap NumPy storage in an unmanaged host view, then copy it
to a view accessible by the default execution space:

```cpp
using HostInput = Kokkos::View<
    const double**, Kokkos::LayoutRight, Kokkos::HostSpace,
    Kokkos::MemoryUnmanaged>;

HostInput host(array.data(), rows, columns);
auto device = Kokkos::create_mirror_view_and_copy(
    Kokkos::DefaultExecutionSpace(), host);
```

`MemoryUnmanaged` means Kokkos does not own or extend the lifetime of the
pointer. The NumPy array must remain alive until the copy completes. Do not
retain an unmanaged NumPy view in a long-lived evaluator. The helpers in
`symmetrix/source/cpp/utilities_kokkos.hpp` implement the normal upload path.

For results, create a host mirror and copy into a newly owned NumPy array. This
repository does not expose accelerator allocations directly as NumPy memory.

## Launching work

Every launch should have a stable label. Labels are both documentation and the
names shown by Kokkos profiling tools.

### Flat independent work: `RangePolicy`

Use a range when each iteration owns an independent item:

```cpp
const Kokkos::DefaultExecutionSpace execution_space;
Kokkos::parallel_for(
    "normalize rows",
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
        execution_space, 0, rows),
    KOKKOS_LAMBDA(const int row) {
        // This iteration owns row.
    });
```

This corresponds to a vectorized operation over the leading dimension. Pass
the execution-space instance explicitly when the function participates in an
ordered evaluator pipeline.

For flattened ranges that can exceed 32-bit indexing, widen before
multiplication and declare the policy index type:

```cpp
using policy = Kokkos::RangePolicy<
    Kokkos::DefaultExecutionSpace, Kokkos::IndexType<std::size_t>>;
Kokkos::parallel_for("flat kernel", policy(execution_space, 0, total), ...);
```

The repository uses `int` for validated model and graph dimensions where the
contract is explicitly 32-bit, and `std::size_t` or 64-bit types for flattened
extents and potentially large execution ranges.

### Rectangular iteration: `MDRangePolicy`

Use an MD range when the iteration space itself is naturally rectangular and
each tuple writes a disjoint element:

```cpp
Kokkos::parallel_for(
    "copy feature tensor",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {0, 0, 0}, {nodes, lm_count, channels}),
    KOKKOS_LAMBDA(const int node, const int lm, const int channel) {
        output(node, lm, channel) = input(node, lm, channel);
    });
```

This is clearer than flattening and manually applying division and modulo. Use
flattening when memory order, launch geometry, or a very large extent makes it
the better contract.

### Hierarchical work: `TeamPolicy`

Teams map well to Python code shaped like an outer loop whose iterations each
contain parallel inner work. In this repository, one league entry is commonly
one atom, receiver, batch row, or graph segment:

```cpp
using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
using Member = TeamPolicy::member_type;

Kokkos::parallel_for(
    "per receiver",
    TeamPolicy(execution_space, receivers, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Member& team) {
        const int receiver = team.league_rank();
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, neighbor_count(receiver)),
            [=](const int local_edge) {
                // Parallel work within this receiver.
            });
        team.team_barrier();  // Required before a following phase reads team data.
        Kokkos::single(Kokkos::PerTeam(team), [=]() {
            // Exactly one worker performs this update.
        });
    });
```

Use `TeamThreadRange` for work shared by team threads. The more advanced
evaluator kernels also use `ThreadVectorRange`, `TeamVectorRange`, and
`TeamVectorMDRange` for a third, SIMD-like level. Introduce those only after
the ownership and reduction structure is correct with simpler policies.

Nested Kokkos algorithms are collective operations. All team members must
encounter them consistently. A `team.team_barrier()` is required when one
nested phase produces data that another phase reads and the intervening
collective does not itself provide the required synchronization.

Do not put a team barrier or a nested collective behind a condition that only
some members of the team take. Branch on values shared by every member, or
restructure the work. Teams are independent: a team barrier never coordinates
two different league entries.

### Team scratch memory

Scratch memory is fast, team-local temporary storage, analogous to a small
temporary NumPy array allocated once per work group rather than once per
element. Admit its size before launching:

```cpp
using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
using Member = TeamPolicy::member_type;
using Scratch = Member::scratch_memory_space;

const auto bytes = admitted_team_scratch_bytes<>(
    "my kernel", {static_cast<std::size_t>(width), sizeof(double)});
auto policy = TeamPolicy(execution_space, batch, Kokkos::AUTO)
    .set_scratch_size(0, Kokkos::PerTeam(bytes));

Kokkos::parallel_for("my kernel", policy,
    KOKKOS_LAMBDA(const Member& team) {
        Kokkos::View<double*, Scratch, Kokkos::MemoryUnmanaged>
            temporary(team.team_scratch(0), width);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, width),
            [=](const int i) { temporary(i) = 0.0; });
        team.team_barrier();
        // All team members may now consume `temporary`.
    });
```

`admitted_team_scratch_bytes` checks multiplication overflow and the backend's
per-team limit. `multilayer_perceptron_kokkos.cpp` contains compact examples of
this pattern.

## Reductions, scans, and shared writes

Parallel code must make write ownership explicit. There are four common cases:

1. One iteration owns one destination element: assign directly.
2. A team combines values: use a nested `parallel_reduce`, then write once with
   `Kokkos::single(Kokkos::PerTeam(...))`.
3. Many unrelated iterations scatter into the same elements: use atomics or
   redesign into a segmented reduction.
4. Variable-length output needs offsets: count, scan, then fill.

### Reductions

```cpp
double total = 0.0;
Kokkos::parallel_reduce(
    "sum energy",
    Kokkos::RangePolicy<decltype(execution_space)>(
        execution_space, 0, count),
    KOKKOS_LAMBDA(const int i, double& update) {
        update += energy(i);
    },
    total);
```

The result is available to host code when the reduction returns. The codebase
also reduces into scalar subviews so the result remains in device storage and
can feed the next enqueued kernel without a host round trip; see stress
reduction in `mace_kokkos_evaluate.cpp`.

Floating-point reductions are not generally bitwise reproducible across
execution spaces or team sizes. Tests should use scientifically justified
tolerances and still catch real numerical regressions.

### Prefix scans and compressed graph structures

Neighbor and edge data use compressed offsets. Their construction follows the
same pattern as building CSR data in Python:

1. Count entries per bin, node, or receiver.
2. Exclusive-scan counts into offsets.
3. Copy offsets to cursors when concurrent filling is required.
4. Fill at positions obtained with `atomic_fetch_add`.

The `final` flag in a Kokkos scan matters:

```cpp
Kokkos::parallel_scan(
    "build offsets", count + 1,
    KOKKOS_LAMBDA(const int i, int& update, const bool final) {
        if (final)
            offsets(i) = update;
        if (i < count)
            update += sizes(i);
    });
```

Do not write the output during the provisional scan pass.
`neighbor_graph_kokkos.cpp` is the repository's clearest complete example.

### Atomics

Force assembly is a true scatter: many directed edges can update the same
atom. `MACEKokkos::reduce_node_forces` therefore uses `Kokkos::atomic_add`.
Atomics are correct but can serialize heavily. Prefer direct ownership or a
team/segmented reduction when the topology offers one; use atomics when the
scatter is genuinely irregular and measure their cost.

Never perform a plain `+=` from parallel iterations unless the destination is
provably lane-private or otherwise disjoint. On CPU SIMD paths, also follow
{doc}`openmp_simd`.

## Device-callable C++

Code inside `KOKKOS_LAMBDA`, and helpers it calls, must compile for every
enabled execution space.

Use:

- `KOKKOS_LAMBDA` for normal value capture;
- `KOKKOS_CLASS_LAMBDA` only when a kernel intentionally calls or captures
  class state through `this`;
- `KOKKOS_FUNCTION` or `KOKKOS_INLINE_FUNCTION` on helper functions called
  from device code;
- Kokkos math functions such as `Kokkos::exp`, `Kokkos::sqrt`, and
  `Kokkos::isfinite` in portable kernels.

Device code cannot use Python objects, allocate a `std::vector`, throw an
exception, write to an iostream, or call an arbitrary host library. Validate
shapes, ranges, and model invariants on the host before launching. If a kernel
must detect invalid data, write or atomically increment a device flag, copy it
back, and throw from host code.

Avoid accidentally capturing host containers, references to stack data, or a
host pointer. Capture view handles and plain scalar values. Treat every
lambda's captures as part of its memory-space contract.

`ZBLKokkos::compute_value_gradient` demonstrates a rarer dual host/device
member function: it retains host mirrors and uses `KOKKOS_IF_ON_HOST` and
`KOKKOS_IF_ON_DEVICE` to read from the correct allocation. Most new code should
instead keep host validation and device execution as separate functions.

## Ordering, asynchrony, and fences

GPU launches are asynchronous with respect to the host. Work submitted to the
same execution-space instance is ordered, so a pipeline can enqueue kernels and
instance-aware deep copies without fencing between every stage:

```cpp
Kokkos::parallel_for("produce", policy_on(execution_space), producer);
Kokkos::deep_copy(execution_space, destination, source);
Kokkos::parallel_for("consume", policy_on(execution_space), consumer);
```

Use a fence when the host must consume a result, before timing an interval,
before releasing or replacing storage still used by outstanding work, or when
work on different execution-space instances must meet. Prefer
`execution_space.fence("reason")` to a global fence when only one stream needs
synchronization.

`ordered_kokkos_deep_copy` in `tools_kokkos.hpp` preserves accelerator launch
order without forcing an unnecessary host synchronization. Evaluator methods
that accept a `completion_fence` flag make the host-completion contract
explicit. Do not add unconditional fences merely to make a race disappear;
identify the producing and consuming execution instances and establish the
smallest correct dependency.

The Python boundary usually needs completion before returning a NumPy result.
A host mirror copy supplies that dependency. Pure device-to-device stages
should normally remain asynchronous.

## Runtime lifecycle from Python

Kokkos has process-wide lifecycle state. Python family resolution initializes
it lazily through `_init_kokkos()`. That wrapper performs the OpenMP runtime
preflight before calling `Kokkos::initialize`. Thread count is read from
`KOKKOS_NUM_THREADS`, falling back to `OMP_NUM_THREADS`, during initialization;
changing it afterward is too late.

Long-lived native evaluators contain a `KokkosLiveObjectGuard`. Finalization is
rejected while such objects exist, because destroying views after Kokkos has
finalized is unsafe. From Python:

```python
if not native._kokkos_is_initialized():
    native._init_kokkos()

evaluator = ...
# use evaluator
del evaluator

if native._kokkos_is_initialized():
    native._finalize_kokkos()
```

Application code normally lets the package manage initialization. Explicit
finalization is mainly for command-line tools, tests, and benchmarks that need
a clean shutdown. After finalization, this runtime deliberately refuses to
construct another evaluator in the same process.

Use `_kokkos_default_execution_space()` and `_openmp_runtime_info()` for
diagnostics rather than inferring the active backend from filenames or
environment variables.

## Backend-aware code without backend sprawl

Portable scientific logic should use `Kokkos::DefaultExecutionSpace`. When a
real algorithmic specialization is required, prefer compile-time traits:

```cpp
if constexpr (device_execution_space<Kokkos::DefaultExecutionSpace>) {
    // Device-specific route.
} else {
    // Host route.
}
```

`device_backend.hpp` defines the neutral trait; family-specific CUDA/HIP
specializations live with the family that owns their ABI and state. Preprocessor
guards such as `KOKKOS_ENABLE_CUDA`, `KOKKOS_ENABLE_HIP`, and
`KOKKOS_ENABLE_OPENMP` remain appropriate for backend headers, external runtime
APIs, and code that cannot be parsed for another backend.

Do not silently fall back from a requested device path to host execution. The
execution planner and evaluator capability checks make backend admission
explicit, and runtime failure on the selected device is an error.

For dense whole-array operations, prefer KokkosBlas. Inside team workers use
KokkosBatched, native team/vector loops, or the repository's qualified helper.
Calling external CBLAS from OpenMP workers is allowed only through the
centralized runtime-compatibility policy; see `host_worker_blas.hpp` and
`host_dense_kernels.hpp`.

## Design choice: generic Kokkos or a generated-code ABI?

Symmetrix-XL has two deliberately different ways to execute scientific work.
They are complementary, not interchangeable implementations to select ad hoc:

| Route | What owns the algorithm | Best use | Artifact requirement |
|---|---|---|---|
| Generic Kokkos (`portable`, T1) | Checked-in C++ and Kokkos views/policies | Correctness backbone, new algorithms, broad model contracts, and diagnosis | None |
| Generated ABI (`generated`, T3) | A versioned, model-contract-specific host or device artifact | Measured hot stages whose shapes, layouts, precision, and ownership are admitted by the contract | Matching prepared artifact |

The generic Kokkos implementation is the normal starting point. It keeps the
scientific operation inspectable in the native source, compiles for the
repository's supported execution spaces, and provides the parity target for a
specialized implementation. It should be designed for correct ownership and
portable memory access, not treated as disposable scaffolding.

Generated code exists where a model contract can make a costly operation much
more specific: for example, a generated receiver-owned program can fuse known
tensor-product structure, traversal, and tiling and avoid materializing
per-edge intermediates. The generated artifact is not just source emitted at
runtime. It is an ABI contract between the generator and native loader that
binds model identity, precision, capabilities, generation, and—on device
paths—the target architecture. The loader rejects a missing, stale, or
incompatible required artifact; it must not silently run an unrelated host or
generic route for a `direct` request.

### Choosing the route

Choose generic Kokkos when the operation has a new or broad input contract,
when its primary need is portability or debuggability, or when no measurement
shows that specialization is worthwhile. First give it a public parity test and
backend coverage. A generated ABI becomes a candidate only after all of the
following are true:

- The operation is demonstrably hot in an end-to-end profile.
- Its admitted shapes, layouts, precision, ownership, and backend scope can be
  expressed as a stable contract.
- The generic route remains available as a correctness oracle for the admitted
  contract.
- Artifact identity, readiness diagnostics, and mismatch rejection can be
  tested at the Python boundary and in the native loader.

Do not add a code generator merely to avoid writing Kokkos. Conversely, do not
force a known, qualified generated path through generic views merely to make
the source look uniform. The execution planner binds the highest qualified
tier for each operator node; its terms are `reference` (T0), `portable` (T1),
`frozen` (T2), and `generated` (T3). See {doc}`execution_planner` for that
binding process and {doc}`jit_and_artifacts` for artifact identity and cache
rules.

### ABI changes are versioned interface changes

An ABI packet is a binary interface, not an internal C++ struct that can be
changed casually. When a generated kernel needs a new field, schedule, or
capability, update the producer and consumer as one versioned change; validate
all sizes, version fields, and capability bits before dispatch; and reject old
artifacts rather than guessing their layout. Update the JIT generation version
with the generator as directed by `jit_generation_version.hpp`, then rebuild
and prepare fresh artifacts in a private task-specific cache.

Keep family-specific generated state and packet types behind the owning
family's adapter. Shared Kokkos owners should keep only neutral contracts and
opaque handles, so a model that never installs a generated plan continues to
use the generic schedule without importing a family ABI. This separation is
why a generic Kokkos improvement and an ABI specialization can evolve without
silently changing one another's scientific or runtime contract.

## Python binding pattern

A normal bound operation has four phases:

1. pybind11 validates dtype and C-contiguity, often with
   `py::array::c_style | py::array::forcecast`.
2. Host code validates ranks, shapes, integer ranges, and scientific
   invariants before launch.
3. An unmanaged host view is copied into evaluator-owned Kokkos storage, then
   native methods enqueue the computation.
4. Results are copied through a host mirror into a Python-owned array.

Keep Python and pybind11 out of `libsymmetrix`. That layer accepts views,
spans, scalars, and standard C++ model data. The binding layer owns Python
exceptions, array conversion, and result packaging.

Be cautious about releasing the GIL. It is useful for long native work, but the
released region must not touch Python objects or call Python APIs, and every
borrowed NumPy buffer must remain alive. Follow an existing neighboring binding
rather than adding release guards mechanically.

## A practical implementation recipe

For a new portable numerical stage:

1. Write down input and output shapes, ownership of every write, and whether
   the active extents can exceed 32-bit indexing.
2. Add the native declaration near the component that owns its data. Keep
   public inputs const and outputs mutable.
3. Validate host-visible contracts before allocating or launching.
4. Reuse retained workspace. Allocate without initialization only for complete
   overwrites; explicitly zero accumulators.
5. Start with `RangePolicy` or `MDRangePolicy`. Move to teams only when there
   is meaningful per-item cooperation or reusable local data.
6. Pass the caller's execution-space instance through every stage that belongs
   to the same asynchronous pipeline.
7. Use reductions, scans, or atomics according to write ownership; never rely
   on CPU serial behavior.
8. Give allocations, launches, and fences stable semantic labels.
9. Add a pybind conversion only at the outer boundary, using the existing
   upload/download helpers.
10. Test the smallest native or Python surface first, then rebuild and test
    each applicable backend separately.

### Review checklist for a new kernel

Use this before requesting review:

- [ ] The owning source file is the correct semantic owner; no broad shared
  header was added merely for convenience.
- [ ] Every input/output shape and integer narrowing is validated on the host.
- [ ] Large products widen their operands before multiplication.
- [ ] Every parallel write is disjoint, reduced, scanned, or atomic by design.
- [ ] The execution space can access every view used by the kernel; a host/device
  boundary has an explicit mirror or `deep_copy`.
- [ ] Multidimensional layout follows the dominant parallel access pattern, and
  `LayoutRight` is used deliberately for C-contiguous interoperation.
- [ ] The lambda captures only views and device-safe scalar values.
- [ ] Device-called helpers carry the required `KOKKOS_FUNCTION` annotation
  and use portable math.
- [ ] `WithoutInitializing` storage is completely written before it is read;
  accumulators and atomic destinations are initialized.
- [ ] The caller's execution-space instance is preserved across the pipeline.
- [ ] A fence exists only where a host/lifetime/timing boundary requires one.
- [ ] Allocation, kernel, profiling-region, and fence labels describe the
  scientific or execution stage.
- [ ] The Python binding copies borrowed input into owned Kokkos storage and
  returns Python-owned output.
- [ ] Tests cover empty/minimal shapes, representative nontrivial shapes,
  invalid inputs, numerical parity, and every intended precision/backend that
  is available.
- [ ] Performance-sensitive changes report allocation behavior, warm timings,
  and the repository's primary metric rather than kernel time alone.

## Common mistakes in this codebase

- Treating `View` assignment as an elementwise copy.
- Reading device memory directly on the host instead of using a host mirror.
- Assuming default layout is C-contiguous on every backend.
- Keeping an unmanaged NumPy or `std::vector` view after its owner can die.
- Capturing a host container or pointer in `KOKKOS_LAMBDA`.
- Assuming that memory accessibility implies a good memory-access pattern.
- Using `int` for a flattened product before proving the product fits.
- Reading a `WithoutInitializing` allocation before every active element is
  written.
- Reallocating a view while an asynchronous kernel may still use its previous
  allocation.
- Writing shared output with plain `+=`.
- Omitting the `final` guard in `parallel_scan` output writes.
- Adding a global fence between every kernel and destroying useful overlap.
- Changing `KOKKOS_NUM_THREADS` after initialization.
- Finalizing while native evaluator objects are alive.
- Calling host BLAS from workers without the runtime compatibility checks.
- Testing only a CPU build for a kernel intended to compile on CUDA or HIP.

## Build, test, and profile

Use the maintained build frontend and a separate fresh build root for CPU,
CUDA, and HIP. The exact commands and cache invalidation rules are in
{doc}`build_and_test`. A source change that affects native code requires a
rebuild and a fresh Python process; do not compare results from a process that
loaded the old extension.

For a focused CPU development cycle:

```bash
source_fingerprint=$(
  .venv/bin/python -c \
  'from pathlib import Path; from tools._symmetrix_build.source_provenance import symmetrix_source_fingerprint; print(symmetrix_source_fingerprint(Path.cwd()))'
)
uv pip install --no-build-isolation --reinstall-package symmetrix-xl \
  -e './symmetrix[test]' \
  --config-setting="cmake.define.SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256=$source_fingerprint"
.venv/bin/pytest path/to/focused_test.py
```

After correctness, inspect runtime provenance with `symmetrix doctor --json`
and `symmetrix version --probe --json`. For performance work, name regions with
`Kokkos::Profiling::pushRegion`/`popRegion` or a small RAII wrapper, warm JIT
artifacts before timing, and fence at timing boundaries. See
{doc}`runtime_and_profiling` for backend tools and benchmark requirements.

On CPU, set affinity and Kokkos/OpenMP/BLAS thread counts before importing
NumPy or the extension. On GPU, first verify the exact interpreter can see the
device and record the GPU, driver, architecture, and competing processes.

## Suggested reading order

1. `tools_kokkos.hpp` and `tools_kokkos.cpp` for lifecycle and data movement.
2. `cubic_spline_kokkos.cpp` for a minimal kernel, while noting that scalar
   evaluation copies coefficients to the host and is not a hot-path pattern.
3. `zbl_kokkos.cpp` for team reductions and device-callable member functions.
4. `neighbor_graph_kokkos.cpp` for count/scan/fill graph construction.
5. `multilayer_perceptron_kokkos.cpp` for teams, scratch memory, reductions,
   explicit execution instances, and optional completion fences.
6. `mace_kokkos_evaluate.cpp` for force and stress assembly.
7. {doc}`/mace_kokkos_source_layout`, then the larger evaluator owners relevant
   to the feature being changed.
8. `pair_symmetrix_mace_kokkos.cpp` only when working on LAMMPS, MPI, dual-view
   synchronization, or device-aware communication.

Kokkos documentation explains the API surface; these files show the ownership,
asynchrony, and performance contracts expected by this repository.
