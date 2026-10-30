#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "family_interface.hpp"

// The execution layer: a registry that maps an operator kind to tiered
// implementations, and a planner that binds the highest qualified tier per
// operator node.  The registry is family-neutral data plus predicates; a
// family declares its own implementations, and its qualification predicates
// read the family's own readiness state, so a binding report states what the
// evaluator can actually run rather than what the registry hopes for.
//
// The architecture/execution split is the point: the program (what a family
// computes) holds no implementation choice, and this layer (how a node runs)
// holds no model semantics.
namespace symmetrix::execution {

// Implementation tiers.
//
//   * `reference` (T0) is single-thread execution of the generic implementation
//     on the default host execution space; it defines contract numerics.
//   * `portable` (T1) is the generic Kokkos implementation, host-first, and is
//     the correctness backbone.
//   * `frozen` (T2) is a legacy built-in module topology.  It is still admitted
//     for the single-layer contracts that ship it and is never created anew;
//     generated operator modules replace it once they cover the contract.
//   * `generated` (T3) is compiled per contract and architecture through the
//     JIT/RTC artifact pipeline and is the production direct implementation.
enum class ImplementationTier {
    reference = 0,
    portable = 1,
    frozen = 2,
    generated = 3,
};

inline const char* implementation_tier_name(const ImplementationTier tier)
{
    switch (tier) {
    case ImplementationTier::reference: return "reference";
    case ImplementationTier::portable: return "portable";
    case ImplementationTier::frozen: return "frozen";
    case ImplementationTier::generated: return "generated";
    }
    return "reference";
}

// The precision and accumulator policy an implementation runs under.  Keeping
// them separate states the contract explicitly: a float32 implementation may
// still accumulate in float64, and a binding report that omitted the
// accumulator could not tell those apart.
struct PrecisionPolicy {
    std::string compute = "float64";
    std::string accumulator = "float64";
};

// Which buffer layouts an implementation consumes and produces.
struct LayoutCompatibility {
    std::string operands = "node_major";
    std::string result = "node_major";
};

// Everything a qualification predicate may read: the node being bound and the
// family-backend state that decides whether an implementation is available.
struct ImplementationContext {
    std::string operator_id;
    std::string capability_key;
    OperatorKind kind = OperatorKind::custom;
    std::string custom_kind;
    std::string target;
    std::string precision;
    bool artifact_ready = false;
    std::string artifact_id;
};

// One tiered implementation an operator kind offers.
struct ImplementationDescriptor {
    // Stable implementation id, for example a module or artifact spelling.
    std::string id;
    OperatorKind kind = OperatorKind::custom;
    std::string custom_kind;
    // The capability key the operator node must declare to be served by this
    // implementation; an empty key matches every node of the kind.
    std::string capability_key;
    ImplementationTier tier = ImplementationTier::portable;
    std::string target = "host";
    PrecisionPolicy precision;
    LayoutCompatibility layout;
    FamilyCapabilities capabilities;
    // Identity of the implementation generation, so a report can pin which
    // build of an implementation was bound.
    std::string identity_fingerprint;
    // Whether this implementation is available for the context.
    std::function<bool(const ImplementationContext&)> qualified;
};

// The plan's decision for one operator node.
struct ImplementationBinding {
    std::string operator_id;
    OperatorKind kind = OperatorKind::custom;
    std::string custom_kind;
    std::string capability_key;
    // The implementation the registry selected for this node, named for the
    // tier that qualified it.  `selected_implementation_id` below names the one
    // the execution path dispatches, which the family resolves from its own
    // dispatch state.
    std::string implementation_id;
    ImplementationTier tier = ImplementationTier::reference;
    std::string target;
    PrecisionPolicy precision;
    std::string identity_fingerprint;
    // The implementation id the execution path dispatches for this node: the
    // binding's authority on what ran.  The registry plans from readiness
    // flags, which say what a node *can* run; where a family's dispatch adds a
    // gate those flags do not carry -- a module that is loaded but not in use,
    // a profile that is not selected -- the family supplies the answer through
    // the registry's resolver and the registry only records it.  A consumer
    // reads this field rather than deriving the implementation from the tier,
    // and it equals `implementation_id` for a family whose declaration names
    // the implementation its dispatch runs.  Empty when nothing was bound.
    std::string selected_implementation_id;
    // Whether any implementation was bound for this node.
    bool qualified = false;
    // Whether a qualified implementation *below* the generated tier -- the
    // portable (T1) or frozen (T2) tier -- serves this node.  Host admission
    // keys on this rather than on the selected tier, because the admission
    // invariant is that every admitted host family has a qualified portable
    // implementation per admitted contract instance, even when a generated one
    // is selected.  A generated implementation never counts: it needs an
    // artifact, which is exactly what the host invariant forbids depending on.
    bool portable_covered = false;
    // Whether the node is served only by a generated implementation, so the
    // artifact that qualified it is required for admission.
    // `artifact_required == qualified && !portable_covered`.
    bool artifact_required = false;
    std::string reason;
};

// The planner's output.  Versioned, so the report consumer can tell a plan that
// predates a field from one that reports it empty.
struct ImplementationBindingReport {
    std::uint32_t version = 1;
    std::string target;
    std::string precision;
    // The artifact readiness the plan was made against; an accelerator
    // backend's generated-only nodes require it.
    bool artifact_ready = false;
    // Whether every operator node was bound.  This is the plan's completeness,
    // not the backend verdict: `family_backend_admits` additionally requires
    // coverage below the generated tier on a host backend and the artifact on
    // an accelerator backend.
    bool admitted = false;
    std::size_t node_count = 0;
    std::size_t portable_node_count = 0;
    std::vector<ImplementationBinding> bindings;
};

namespace detail {

inline bool descriptor_serves(
    const ImplementationDescriptor& descriptor,
    const ImplementationContext& context)
{
    if (descriptor.kind != context.kind)
        return false;
    if (descriptor.kind == OperatorKind::custom
        && descriptor.custom_kind != context.custom_kind)
        return false;
    if (descriptor.target != context.target)
        return false;
    if (!descriptor.capability_key.empty()
        && descriptor.capability_key != context.capability_key)
        return false;
    return descriptor.qualified ? descriptor.qualified(context) : false;
}

// The best qualified descriptor: the highest tier, and among equal tiers the
// declaration order the family fixed.  A frozen tier never outranks a
// generated one, so the legacy topology is only selected when nothing newer is
// qualified.
inline const ImplementationDescriptor* best_qualified(
    const std::vector<ImplementationDescriptor>& descriptors,
    const ImplementationContext& context)
{
    const ImplementationDescriptor* best = nullptr;
    for (const auto& descriptor : descriptors) {
        if (!descriptor_serves(descriptor, context))
            continue;
        if (best == nullptr || descriptor.tier > best->tier)
            best = &descriptor;
    }
    return best;
}

// Whether a qualified implementation below the generated tier serves the node,
// which is the coverage that keeps a family runnable without any artifact.
inline bool has_portable_coverage(
    const std::vector<ImplementationDescriptor>& descriptors,
    const ImplementationContext& context)
{
    return std::any_of(descriptors.begin(), descriptors.end(),
        [&context] (const ImplementationDescriptor& descriptor) {
            return descriptor.tier < ImplementationTier::generated
                && descriptor_serves(descriptor, context);
        });
}

} // namespace detail

// The tiered implementation registry one family-backend declares.
class ImplementationRegistry {
public:
    // The family's statement of which implementation id its execution path
    // dispatches for a node the registry bound.  The registry is family-neutral
    // data plus predicates, so it can plan a node but cannot know the policy
    // the family's dispatcher gates on; a family whose dispatch is wider than
    // its declared readiness flags supplies this, and the registry records the
    // answer in the binding rather than leaving the report a tier ahead of the
    // run.  An empty result leaves the selected implementation in place.
    using SelectedImplementationResolver = std::function<std::string(
        const ImplementationContext&, const ImplementationBinding&)>;

    void set_selected_implementation_resolver(
        SelectedImplementationResolver resolver)
    {
        selected_implementation_resolver_ = std::move(resolver);
    }

    void declare(ImplementationDescriptor descriptor)
    {
        const std::string identity_reason = validate_operator_kind_identity(
            descriptor.kind, descriptor.custom_kind);
        if (!identity_reason.empty())
            throw std::invalid_argument(identity_reason);
        descriptors_.push_back(std::move(descriptor));
    }

    bool empty() const { return descriptors_.empty(); }
    std::size_t size() const { return descriptors_.size(); }

    const std::vector<ImplementationDescriptor>& implementations() const
    {
        return descriptors_;
    }

    // The highest qualified implementation for one node, if any is qualified.
    std::optional<ImplementationBinding> bind(
        const ImplementationContext& context) const
    {
        const ImplementationDescriptor* best =
            detail::best_qualified(descriptors_, context);
        const bool portable =
            detail::has_portable_coverage(descriptors_, context);
        if (best == nullptr) {
            ImplementationBinding unbound;
            unbound.operator_id = context.operator_id;
            unbound.kind = context.kind;
            unbound.custom_kind = context.custom_kind;
            unbound.capability_key = context.capability_key;
            unbound.target = context.target;
            unbound.precision = PrecisionPolicy{context.precision, context.precision};
            unbound.portable_covered = portable;
            unbound.reason = "no implementation is qualified for the operator on "
                "target '" + context.target + "'";
            return unbound;
        }
        ImplementationBinding binding;
        binding.operator_id = context.operator_id;
        binding.kind = context.kind;
        binding.custom_kind = context.custom_kind;
        binding.capability_key = context.capability_key;
        binding.implementation_id = best->id;
        binding.tier = best->tier;
        binding.target = best->target;
        binding.precision = best->precision;
        binding.identity_fingerprint = best->identity_fingerprint;
        binding.qualified = true;
        binding.portable_covered = portable;
        binding.artifact_required = !portable;
        binding.selected_implementation_id = binding.implementation_id;
        if (selected_implementation_resolver_) {
            const std::string dispatched =
                selected_implementation_resolver_(context, binding);
            if (!dispatched.empty())
                binding.selected_implementation_id = dispatched;
        }
        binding.reason = std::string("qualified ") + implementation_tier_name(best->tier)
            + " implementation '" + best->id + "'";
        return binding;
    }

private:
    std::vector<ImplementationDescriptor> descriptors_;
    SelectedImplementationResolver selected_implementation_resolver_;
};

// Bind every operator node of a compiled program.  One context template
// carries the family-backend state (target, precision, artifact readiness) and
// each node contributes its own operator id, capability key, and kind: a single
// artifact can be shared by several nodes, so the predicate sees the ready flag
// rather than one per-node artifact path.
inline ImplementationBindingReport plan_program(
    const Program& program,
    const ImplementationRegistry& registry,
    const std::string& target,
    const std::string& precision,
    const bool artifact_ready = false,
    const std::string& artifact_id = {})
{
    ImplementationBindingReport report;
    report.target = target.empty() ? "host" : target;
    report.precision = precision;
    report.artifact_ready = artifact_ready;
    report.node_count = program.operators.size();
    report.bindings.reserve(program.operators.size());
    for (const auto& node : program.operators) {
        const std::string identity_reason =
            validate_operator_kind_identity(node);
        if (!identity_reason.empty()) {
            ImplementationBinding invalid;
            invalid.operator_id = node.id;
            invalid.kind = node.kind;
            invalid.custom_kind = node.custom_kind;
            invalid.capability_key = node.capability_key;
            invalid.target = report.target;
            invalid.precision = PrecisionPolicy{precision, precision};
            invalid.reason = identity_reason;
            report.bindings.push_back(std::move(invalid));
            continue;
        }
        ImplementationContext context;
        context.operator_id = node.id;
        context.capability_key = node.capability_key;
        context.kind = node.kind;
        context.custom_kind = node.custom_kind;
        context.target = report.target;
        context.precision = precision;
        context.artifact_ready = artifact_ready;
        context.artifact_id = artifact_id;
        auto binding = registry.bind(context);
        if (!binding)
            continue;
        report.bindings.push_back(std::move(*binding));
    }
    report.admitted = std::all_of(report.bindings.begin(), report.bindings.end(),
        [] (const ImplementationBinding& binding) { return binding.qualified; });
    report.admitted = report.admitted
        && report.bindings.size() == report.node_count;
    report.portable_node_count = static_cast<std::size_t>(std::count_if(
        report.bindings.begin(), report.bindings.end(),
        [] (const ImplementationBinding& binding) {
            return binding.portable_covered;
        }));
    return report;
}

// Admission for a family-backend combination.
//
// A host backend requires a qualified implementation below the generated tier
// for every node: portable coverage is the invariant that keeps a host family
// runnable without generated code, so a host program served only by a
// generated implementation is not admitted even when its artifact is ready.
//
// An accelerator backend may admit a program whose nodes are served by
// generated implementations only, and then the artifact that qualified them is
// required: a node with no coverage below the generated tier carries
// `artifact_required`, and the report records the readiness the plan was made
// against.  Every node must still be qualified, so a request without a usable
// artifact fails admission instead of silently degrading.
inline bool family_backend_admits(const ImplementationBindingReport& report)
{
    if (!report.admitted)
        return false;
    if (report.target == "host")
        return std::all_of(report.bindings.begin(), report.bindings.end(),
            [] (const ImplementationBinding& binding) {
                return binding.portable_covered;
            });
    return std::all_of(report.bindings.begin(), report.bindings.end(),
        [&report] (const ImplementationBinding& binding) {
            return binding.qualified
                && (!binding.artifact_required || report.artifact_ready);
        });
}

// A family evaluator that also declares its implementation registry and the
// binding report of its compiled program.
template <typename Evaluator>
concept TieredFamilyEvaluator = FamilyEvaluator<Evaluator>
    && requires (const Evaluator& const_evaluator) {
        {
            const_evaluator.family_implementation_registry()
        } -> std::convertible_to<ImplementationRegistry>;
        {
            const_evaluator.family_binding_report()
        } -> std::convertible_to<ImplementationBindingReport>;
    };

} // namespace symmetrix::execution
