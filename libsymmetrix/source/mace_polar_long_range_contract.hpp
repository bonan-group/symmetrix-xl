#pragma once

#include <iomanip>
#include <sstream>
#include <string>

#include "long_range_operator_kinds.hpp"

namespace symmetrix::execution::polar {

// Electrostatic geometry and kernel shapes are reusable long-range data. The
// fixed recursion schedule, alpha/beta layout, and self-term policy belong to
// the MACE Polar program and therefore live in this family-owned aggregate.
struct PolarLongRangeContract : long_range::LongRangeContract {
    int recursion_step_count = 1;
    int spin_channel_count = 2;
    bool include_electrostatic_self_interaction = false;
    bool field_si = false;
};

inline std::string validate_polar_long_range_contract(
    const PolarLongRangeContract& contract)
{
    const std::string primitive_reason=
        long_range::validate_long_range_contract(contract);
    if(!primitive_reason.empty())return primitive_reason;
    if(contract.spin_channel_count!=2)
        return "spin_channel_count must be 2 (the alpha/beta equilibration)";
    const std::size_t density_width=static_cast<std::size_t>(
        (contract.density_max_l+1)*(contract.density_max_l+1));
    if(contract.charges_dimension
        !=static_cast<std::size_t>(contract.spin_channel_count)*density_width)
        return "charges_dimension must be spin_channel_count * "
            "(density_max_l + 1)^2";
    if(contract.potential_dimension
        !=static_cast<std::size_t>(contract.spin_channel_count)
            *contract.field_dimension)
        return "potential_dimension must be spin_channel_count * "
            "field_dimension";
    if(contract.recursion_step_count<1)
        return "recursion_step_count must be at least 1";
    return {};
}

// Preserve the serialized/program identity emitted before the ownership
// split. Moving family fields out of the neutral aggregate is an ownership
// correction, not a model-identity or cache-key change.
inline std::string polar_long_range_contract_fingerprint(
    const PolarLongRangeContract& contract)
{
    std::ostringstream parameters;
    parameters<<std::setprecision(17);
    sha256::Digest digest;
    digest.update("symmetrix-long-range-contract-v1\n");
    const auto fold=[&digest,&parameters](
        const std::string& name,const double value) {
        parameters.str(std::string());
        parameters.clear();
        parameters<<name<<"="<<value<<"\n";
        digest.update(parameters.str());
    };
    fold("kspace_cutoff",contract.kspace_cutoff);
    fold("density_max_l",static_cast<double>(contract.density_max_l));
    fold("density_smearing_width",contract.density_smearing_width);
    fold("feature_max_l",static_cast<double>(contract.feature_max_l));
    for(std::size_t index=0;
        index<contract.feature_smearing_widths.size();++index)
        fold("feature_smearing_width_"+std::to_string(index),
            contract.feature_smearing_widths[index]);
    fold("charges_dimension",static_cast<double>(contract.charges_dimension));
    fold("field_dimension",static_cast<double>(contract.field_dimension));
    fold("potential_dimension",static_cast<double>(contract.potential_dimension));
    fold("field_norm_factor",contract.field_norm_factor);
    fold("recursion_step_count",
        static_cast<double>(contract.recursion_step_count));
    fold("spin_channel_count",static_cast<double>(contract.spin_channel_count));
    digest.update("table:"+contract.table_dtype+"\n");
    digest.update("accumulator:"+contract.accumulator_dtype+"\n");
    fold("include_electrostatic_self_interaction",
        contract.include_electrostatic_self_interaction?1.0:0.0);
    fold("field_si",contract.field_si?1.0:0.0);
    return digest.finish();
}

} // namespace symmetrix::execution::polar
