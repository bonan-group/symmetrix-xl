#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "execution_plan.hpp"

// The whole-plan candidate space a model family declares for the execution-plan
// report.  A family registers one descriptor per *whole* plan it can run --
// never a per-node implementation tier, which the implementation registry binds
// instead -- so the report's candidate list is family data rather than planner
// code, and the plan ids the frontends and the debug-plan pin match on have one
// native owner.
//
// A family that declares no plan space reports the one plan it selected rather
// than an empty candidate list; the report consumer applies that rule, so a
// declared space never invents a candidate the family did not declare.
namespace symmetrix::execution {

// One declared candidate's evaluation: the qualification predicate's verdict,
// the plan's predicted graph bytes, and the human-readable reason.
struct PlanCandidateEvaluation {
    bool qualified = false;
    std::optional<std::size_t> estimated_bytes;
    std::string reason;
    bool fixed_workspace = false;
};

// A whole-plan candidate: the stable plan id (the cross-language string
// contract), plus the qualification predicate and byte estimator that the
// report evaluates at graph time.
struct PlanCandidateDescriptor {
    std::string id;
    std::function<PlanCandidateEvaluation()> evaluate;
};

// The candidate space one family declares, in declaration order.  Declaration
// order is the report's candidate order and stays fixed: consumers compare the
// plan-id contract against it.
class PlanCandidateSpace {
public:
    void declare(
        std::string id, std::function<PlanCandidateEvaluation()> evaluate)
    {
        descriptors_.push_back({std::move(id), std::move(evaluate)});
    }

    bool empty() const { return descriptors_.empty(); }
    std::size_t size() const { return descriptors_.size(); }

    const std::vector<PlanCandidateDescriptor>& descriptors() const
    {
        return descriptors_;
    }

    // The declared plan ids in declaration order: the native side of the
    // cross-language plan-id contract.
    std::vector<std::string> ids() const
    {
        std::vector<std::string> declared;
        declared.reserve(descriptors_.size());
        for (const auto& descriptor : descriptors_)
            declared.push_back(descriptor.id);
        return declared;
    }

    // Whether the family declared this plan id.
    bool declares(const std::string& id) const
    {
        for (const auto& descriptor : descriptors_) {
            if (descriptor.id == id)
                return true;
        }
        return false;
    }

    // The report entries for the declared space, in declaration order.
    std::vector<ExecutionPlanCandidate> evaluate() const
    {
        std::vector<ExecutionPlanCandidate> candidates;
        candidates.reserve(descriptors_.size());
        for (const auto& descriptor : descriptors_) {
            const auto evaluation = descriptor.evaluate();
            ExecutionPlanCandidate candidate;
            candidate.id = descriptor.id;
            candidate.qualified = evaluation.qualified;
            candidate.estimated_bytes = evaluation.estimated_bytes;
            candidate.reason = evaluation.reason;
            candidate.fixed_workspace = evaluation.fixed_workspace;
            candidates.push_back(std::move(candidate));
        }
        return candidates;
    }

private:
    std::vector<PlanCandidateDescriptor> descriptors_;
};

} // namespace symmetrix::execution
