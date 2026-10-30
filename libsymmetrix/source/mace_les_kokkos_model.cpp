#include "mace_les_kokkos.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include "nlohmann/json.hpp"

namespace {

const SelectedMACEModel& validate_mace_les_topology(
    const SelectedMACEModel& selected)
{
    if (selected.kind != MACEModelKind::mace_les)
        return selected;
    const auto& data = selected.data;
    if (data.value("has_field_coupling", false))
        throw std::invalid_argument(
            "MACELES does not support field-aware MACE models.");
    if (data.value("num_interactions", 2) != 2
        || data.value("single_layer_readout", false))
        throw std::invalid_argument(
            "MACELES requires a standard two-interaction MACE backbone.");
    return selected;
}

} // namespace

template <typename Precision>
MACELESKokkos<Precision>::MACELESKokkos(
    std::string filename,
    std::string requested_head)
  : MACELESKokkos(
        read_selected_mace_model(filename, requested_head))
{}

template <typename Precision>
MACELESKokkos<Precision>::MACELESKokkos(
    const SelectedMACEModel& selected)
  : MACEKokkos<Precision>(
        validate_mace_les_topology(selected),
        MACEModelKind::mace_les)
{
    const auto& data = selected.data;
    this->les_.parameters = MACELESParameters<Precision>::from_json(
        data.at("les"),
        this->num_interactions,
        this->num_channels,
        data.value("execution_contracts", nlohmann::json::object()));
    this->initialize_les_readouts();
}

template MACELESKokkos<float>::MACELESKokkos(std::string, std::string);
template MACELESKokkos<double>::MACELESKokkos(std::string, std::string);
template MACELESKokkos<float>::MACELESKokkos(const SelectedMACEModel&);
template MACELESKokkos<double>::MACELESKokkos(const SelectedMACEModel&);

template <typename Precision>
void MACELESKokkos<Precision>::initialize_les_readouts()
{
    if (!les_.parameters)
        throw std::logic_error("LES readouts require MACELES parameters.");
    les_.readout_1_weights = toKokkosView(
        "LES first readout weights", les_.parameters->readout_1_weights);
    les_.readout_2 = MultilayerPerceptronKokkos(
        std::vector<int>{
            this->num_channels, les_.parameters->readout_2_hidden_size, 1},
        std::vector<std::vector<double>>{
            les_.parameters->readout_2_weights_1,
            les_.parameters->readout_2_weights_2},
        les_.parameters->readout_2_scale_factor);
    les_.monopole =
        std::make_unique<symmetrix::les::LESMonopoleKokkos<Precision>>(
            les_.parameters->sigma, les_.parameters->dl,
            les_.parameters->remove_self_interaction);
    les_.energy_host = Kokkos::View<double,Kokkos::HostSpace>(
        "LES host total energy");
}


template void MACELESKokkos<float>::initialize_les_readouts();
template void MACELESKokkos<double>::initialize_les_readouts();
