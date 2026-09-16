template <typename Precision>
void MACEKokkos<Precision>::compute_electric_field_response(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    Kokkos::View<const double*> electric_field,
    const bool recompute_primal,
    const bool include_force_derivative)
{
    if (!has_field_coupling)
        throw std::invalid_argument(
            "MACEKokkos::compute_electric_field_hessian requires field coupling.");
    if (electric_field.size() != 3)
        throw std::invalid_argument(
            "MACEKokkos::compute_electric_field_hessian requires a graph-level electric field.");
    const std::size_t response_edges = neigh_indices.extent(0);
    const std::size_t coordinate_count = 3*response_edges;
    const bool compact_geometry = use_compact_edge_geometry();
    const auto unit_direction = execution_prepared_unit_direction;
    if (r.extent(0) != response_edges
        || (compact_geometry
            ? unit_direction.extent(0) < coordinate_count
            : xyz.extent(0) < coordinate_count))
        throw std::invalid_argument(
            "MACEField response geometry does not match the graph extents.");
    if (recompute_primal) {
        compute_node_energies_forces_field(
            num_nodes, node_types, num_neigh, neigh_indices, neigh_types,
            xyz, r, electric_field);
    }

    if (use_mh0_adjoint_reuse())
        reconstruct_mh0_field_response_state(
            num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);

    const auto response_mode = streamed_edges;
    if (include_force_derivative
        && (response_mode == MACEStreamedEdgesMode::generic
            || mace_uses_prepared_execution(response_mode)))
        compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
    if (response_mode != MACEStreamedEdgesMode::materialized)
        compute_R1(
            num_nodes, node_types, num_neigh, neigh_types, r,
            include_force_derivative);
    else if (!include_force_derivative)
        R1_deriv = decltype(R1_deriv)();

    Kokkos::View<const int*> response_edge_receivers;
    if (mace_uses_prepared_execution(response_mode)) {
        if (factorized_schedule_dirty
            || execution_schedule_num_nodes != num_nodes
            || execution_edge_receivers.extent(0) != response_edges
            || execution_direct_source_offsets.extent(0)
                != static_cast<std::size_t>(num_nodes)+1
            || execution_direct_source_edges.extent(0) != response_edges
            || !execution_edge_receivers.span_is_contiguous()
            || !execution_direct_source_offsets.span_is_contiguous()
            || !execution_direct_source_edges.span_is_contiguous())
            throw std::logic_error(
                "Factorized MACEField response topology does not match the current graph.");
        response_edge_receivers = execution_edge_receivers;
        macefield_response_receiver_ownership = "factorized_edge";
        macefield_response_source_ownership = "edge_atomic";
        macefield_response_factorized_topology_count += 1;
    } else if (response_mode != MACEStreamedEdgesMode::materialized) {
        response_edge_receivers = streamed_edge_receivers;
        macefield_response_receiver_ownership =
            response_edge_receivers.extent(0) == response_edges
                ? "streamed_edge" : "node";
        macefield_response_source_ownership = "edge_atomic";
    } else {
        macefield_response_receiver_ownership = "node";
        macefield_response_source_ownership = "node_atomic";
    }
    macefield_response_call_count += 1;

    if (electric_field_hessian.size() != 9)
        Kokkos::realloc(electric_field_hessian, 9);
    if (include_force_derivative) {
        if (electric_field_force_derivative.size() != 3*coordinate_count)
            Kokkos::realloc(
                electric_field_force_derivative, 3*coordinate_count);
    } else {
        electric_field_force_derivative =
            decltype(electric_field_force_derivative)();
    }
    Kokkos::deep_copy(electric_field_hessian, 0.0);
    if (include_force_derivative)
        Kokkos::deep_copy(electric_field_force_derivative, 0.0);

    Kokkos::View<int*> first_neigh("MACEField response first_neigh", num_nodes);
    Kokkos::parallel_scan(
        "MACEField response first_neigh",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });
    Kokkos::fence();

    const int channels = num_channels;
    const int lm_count = num_lm;
    const int LM_count = num_LM;
    const int lmax = l_max;
    const int Lmax = L_max;
    const int phi_rows = num_lelm1lm2;
    const int phi_outputs = num_lme;
    const auto response_hessian = electric_field_hessian;
    const auto response_forces = electric_field_force_derivative;
    const int polynomial_nodes = M1_poly_coeff.extent(1);

    Kokkos::View<Precision***,Kokkos::LayoutRight> H1_dot;
    Kokkos::View<Precision***,Kokkos::LayoutRight> Phi1r_dot;
    Kokkos::View<Precision***,Kokkos::LayoutRight> Phi1_dot;
    Kokkos::View<Precision***,Kokkos::LayoutRight> A1_dot;
    Kokkos::View<Precision**,Kokkos::LayoutRight> M1_dot;
    Kokkos::View<Precision***,Kokkos::LayoutRight> M1_value_dot;
    Kokkos::View<Precision***,Kokkos::LayoutRight> M1_gradient;
    Kokkos::View<Precision***,Kokkos::LayoutRight> M1_gradient_dot;
    Kokkos::View<double**,Kokkos::LayoutRight> H2_dot;
    Kokkos::View<double*> readout_output;
    Kokkos::View<double**,Kokkos::LayoutRight> H2_adj_local;
    Kokkos::View<double**,Kokkos::LayoutRight> H2_adj_dot;
    Kokkos::View<Precision***,Kokkos::LayoutRight> H1_adj_local;
    Kokkos::View<Precision***,Kokkos::LayoutRight> H1_adj_dot;
    Kokkos::View<Precision**,Kokkos::LayoutRight> M1_adj_local;
    Kokkos::View<Precision**,Kokkos::LayoutRight> M1_adj_dot;
    Kokkos::View<Precision***,Kokkos::LayoutRight> A1_adj_local;
    Kokkos::View<Precision***,Kokkos::LayoutRight> A1_adj_dot;
    Kokkos::View<Precision***,Kokkos::LayoutRight> dPhi1_local;
    Kokkos::View<Precision***,Kokkos::LayoutRight> dPhi1_dot;
    Kokkos::View<Precision***,Kokkos::LayoutRight> dPhi1r_local;
    Kokkos::View<Precision***,Kokkos::LayoutRight> dPhi1r_dot;
    Kokkos::View<Precision***,Kokkos::LayoutRight> H1_product_adj_dot;
    Kokkos::View<Precision***,Kokkos::LayoutRight> M0_adj_dot(
        "MACEField M0 adjoint tangent",
        include_force_derivative ? num_nodes : 0, LM_count, channels);
    Kokkos::View<Precision***,Kokkos::LayoutRight> A0_adj_dot(
        "MACEField A0 adjoint tangent",
        include_force_derivative ? num_nodes : 0, lm_count, channels);

    for (int seed=0; seed<3; ++seed) {
        H1_dot = decltype(H1_dot)(
            "MACEField H1_dot", num_nodes, LM_count, channels);
        Phi1r_dot = decltype(Phi1r_dot)(
            "MACEField Phi1r_dot", num_nodes, phi_rows, channels);
        Phi1_dot = decltype(Phi1_dot)(
            "MACEField Phi1_dot", num_nodes, phi_outputs, channels);
        A1_dot = decltype(A1_dot)(
            "MACEField A1_dot", num_nodes, lm_count, channels);
        const auto H1_before_field = H1_pre_field;
        const auto H1_up_weights = H1_linear_up_weights;
        const int field_entry_count = num_field_angular_entries;
        const auto field_input_lm = field_entry_input_lm;
        const auto field_component = field_entry_component;
        const auto field_output_lm = field_entry_output_lm;
        const auto field_path = field_entry_path;
        const auto field_coefficient = field_entry_coefficient;
        const auto field_up_matrix = field_path_up_matrix;
        Kokkos::parallel_for(
            "MACEField analytic field and linear-up tangent",
            Kokkos::MDRangePolicy<
                Kokkos::Rank<3,Kokkos::Iterate::Right>,
                Kokkos::IndexType<std::size_t>>(
                {0,0,0}, {num_nodes,LM_count,channels}),
            KOKKOS_LAMBDA (
                const std::size_t i,
                const std::size_t lm,
                const std::size_t output) {
                Precision value = 0.0;
                for (int entry=0; entry<field_entry_count; ++entry) {
                    if (field_output_lm(entry) != lm
                        || field_component(entry) != seed)
                        continue;
                    const int path = field_path(entry);
                    for (int input=0; input<channels; ++input) {
                        value += field_coefficient(entry)
                            * field_up_matrix(path,input,output)
                            * H1_before_field(i,field_input_lm(entry),input);
                    }
                }
                H1_dot(i,lm,output) = value;
            });

        Kokkos::deep_copy(Phi1_dot, 0.0);
        const auto phi_lm1 = Phi1_lm1;
        const auto phi_lm2 = Phi1_lm2;
        const auto phi_path = Phi1_lel1l2;
        const auto radial1 = R1;
        const auto harmonics = Y;
        Kokkos::parallel_for(
            "MACEField analytic Phi1r tangent",
            Kokkos::TeamPolicy<>(num_nodes*phi_rows, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                const int i = team.league_rank()/phi_rows;
                const int row = team.league_rank()%phi_rows;
                const int edge_begin = first_neigh(i);
                const int lm1 = phi_lm1(row);
                const int lm2 = phi_lm2(row);
                const int path = phi_path(row);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(team, channels),
                    [=] (const int channel) {
                        Precision value = 0.0;
                        for (int j=0; j<num_neigh(i); ++j) {
                            const std::size_t edge =
                                static_cast<std::size_t>(edge_begin+j);
                            const std::size_t harmonic_offset =
                                edge*static_cast<std::size_t>(lm_count);
                            value += radial1(edge,path*channels+channel)
                                *harmonics(harmonic_offset+lm1)
                                *H1_dot(neigh_indices(edge),lm2,channel);
                        }
                        Phi1r_dot(i,row,channel) = value;
                    });
            });

        const auto phi_lme = Phi1_lme;
        const auto phi_row = Phi1_lelm1lm2;
        const auto phi_cg = Phi1_clebsch_gordan;
        Kokkos::parallel_for(
            "MACEField analytic Phi1 tangent",
            Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                const int i = team.league_rank();
                for (int p=0; p<phi_cg.extent(0); ++p) {
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(team, channels),
                        [=] (const int channel) {
                            Phi1_dot(i,phi_lme(p),channel) += phi_cg(p)
                                *Phi1r_dot(i,phi_row(p),channel);
                        });
                }
            });

        const auto phi_l = Phi1_l;
        const auto A1_matrix = A1_weights;
        Kokkos::parallel_for(
            "MACEField analytic A1 tangent",
            Kokkos::TeamPolicy<>(num_nodes*(lmax+1), Kokkos::AUTO),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                const int i = team.league_rank()/(lmax+1);
                const int l = team.league_rank()%(lmax+1);
                int lme = 0;
                int multiplicity = 0;
                for (int p=0; p<phi_l.extent(0); ++p) {
                    if (phi_l(p) < l)
                        lme += 2*phi_l(p)+1;
                    if (phi_l(p) == l)
                        ++multiplicity;
                }
                auto input = Kokkos::View<Precision**,Kokkos::LayoutRight,
                    Kokkos::MemoryUnmanaged>(
                        &Phi1_dot(i,lme,0), 2*l+1, multiplicity*channels);
                auto output = Kokkos::subview(
                    A1_dot, i, Kokkos::make_pair(l*l,l*(l+2)+1), Kokkos::ALL);
                KokkosBatched::TeamGemm<
                    Kokkos::TeamPolicy<>::member_type,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Algo::Gemm::Blocked>::invoke(
                        team, 1.0, input, A1_matrix(l), 0.0, output);
            });

        const auto A1_scale_values = A1_spline_values;
        const auto A1_scale_splines = A1_splines;
        const auto A1_response_type_to_active = type_to_active;
        const int A1_response_active_type_count = num_active_types;
        const bool A1_scale_recompute = use_mh0_adjoint_reuse();
        if (A1_scaled) {
            Kokkos::parallel_for(
                "MACEField analytic A1 tangent scaling",
                Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                    const int i = team.league_rank();
                    double scale;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, num_neigh(i)),
                        [=] (const int j, double& sum) {
                            const int edge = first_neigh(i)+j;
                            if (A1_scale_recompute) {
                                const int type_i = A1_response_type_to_active(
                                    node_types(i));
                                const int type_j = A1_response_type_to_active(
                                    neigh_types(edge));
                                const int edge_type = type_i <= type_j
                                    ? type_i*(2*A1_response_active_type_count-type_i-1)/2
                                        +type_j
                                    : type_j*(2*A1_response_active_type_count-type_j-1)/2
                                        +type_i;
                                sum += A1_scale_splines.evaluate_function(
                                    edge_type, r(edge), 0);
                            } else {
                                sum += A1_scale_values(edge,0);
                            }
                        }, scale);
                    scale += 1.0;
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, lm_count*channels),
                        [=] (const int index) {
                            A1_dot(i,index/channels,index%channels) /= scale;
                        });
                });
        }

        Kokkos::fence();
        Phi1r_dot = decltype(Phi1r_dot)();
        Phi1_dot = decltype(Phi1_dot)();

        M1_dot = decltype(M1_dot)(
            "MACEField M1_dot", num_nodes, channels);
        Kokkos::deep_copy(M1_dot, 0.0);
        const auto M1_spec = M1_poly_spec;
        const auto M1_coeff = M1_poly_coeff;
        const auto M1_A1 = A1;
        if (m1_polynomial_policy == M1PolynomialPolicy::recompute) {
            const int multiplication_nodes = M1_spec.extent(0);
            const int tile_channels =
                macefield_response_m1_recompute_tile_channels();
            const int vector_length = m1_recompute_vector_length();
            const int scratch_level = m1_recompute_scratch_level();
            using TeamMember = typename Kokkos::TeamPolicy<>::member_type;
            using ScratchView = Kokkos::View<
                Precision**, Kokkos::LayoutRight,
                typename TeamMember::scratch_memory_space,
                Kokkos::MemoryUnmanaged>;
            auto policy = vector_length == 1
                ? Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 1)
                : Kokkos::TeamPolicy<>(num_nodes, 1, vector_length);
            policy.set_scratch_size(
                scratch_level, Kokkos::PerTeam(
                    2*sizeof(Precision)*polynomial_nodes*tile_channels));
            Kokkos::parallel_for(
                "MACEField analytic M1 directional recompute", policy,
                KOKKOS_LAMBDA (const TeamMember& member) {
                    const std::size_t node = static_cast<std::size_t>(
                        member.league_rank());
                    ScratchView workspace(
                        member.team_scratch(scratch_level),
                        2*polynomial_nodes, tile_channels);
                    for (int channel_begin=0; channel_begin<channels;
                         channel_begin += tile_channels) {
                        const int active_channels = Kokkos::min(
                            tile_channels, channels-channel_begin);
                        Kokkos::parallel_for(
                            Kokkos::TeamVectorRange(member, active_channels),
                            [=] (const int lane) {
                                const int channel = channel_begin+lane;
                                Precision output = 0;
                                for (int p=0; p<lm_count; ++p) {
                                    workspace(p,lane) =
                                        M1_A1(node,p,channel);
                                    workspace(polynomial_nodes+p,lane) =
                                        A1_dot(node,p,channel);
                                    output += M1_coeff(
                                        node_types(node),p,channel)
                                        *workspace(polynomial_nodes+p,lane);
                                }
                                for (int p=0; p<multiplication_nodes; ++p) {
                                    const int p0 = M1_spec(p,0);
                                    const int p1 = M1_spec(p,1);
                                    const int output_node = lm_count+p;
                                    workspace(output_node,lane) =
                                        workspace(p0,lane)*workspace(p1,lane);
                                    workspace(polynomial_nodes+output_node,lane) =
                                        workspace(polynomial_nodes+p0,lane)
                                            *workspace(p1,lane)
                                        +workspace(p0,lane)
                                            *workspace(polynomial_nodes+p1,lane);
                                    output += M1_coeff(
                                        node_types(node),output_node,channel)
                                        *workspace(
                                            polynomial_nodes+output_node,lane);
                                }
                                M1_dot(node,channel) = output;
                            });
                        member.team_barrier();
                    }
                });
            macefield_response_m1_recompute_forward_launch_count += 1;
        } else {
            M1_value_dot = decltype(M1_value_dot)(
                "MACEField M1 value_dot", num_nodes, polynomial_nodes, channels);
            M1_gradient = decltype(M1_gradient)(
                "MACEField M1 gradient", num_nodes, polynomial_nodes, channels);
            M1_gradient_dot = decltype(M1_gradient_dot)(
                "MACEField M1 gradient_dot", num_nodes, polynomial_nodes, channels);
            const auto M1_values = M1_poly_values;
            Kokkos::parallel_for(
                "MACEField analytic M1 directional graph",
                Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                    const int i = team.league_rank();
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorMDRange<
                            Kokkos::Rank<2,Kokkos::Iterate::Right>,
                            Kokkos::TeamPolicy<>::member_type>(
                                team, lm_count, channels),
                        [=] (const int lm, const int channel) {
                            M1_value_dot(i,lm,channel) = A1_dot(i,lm,channel);
                            Kokkos::atomic_add(
                                &M1_dot(i,channel),
                                M1_coeff(node_types(i),lm,channel)
                                    *M1_value_dot(i,lm,channel));
                        });
                    team.team_barrier();
                    for (int p=0; p<M1_spec.extent(0); ++p) {
                        const int p0 = M1_spec(p,0);
                        const int p1 = M1_spec(p,1);
                        Kokkos::parallel_for(
                            Kokkos::TeamVectorRange(team, channels),
                            [=] (const int channel) {
                                const int node = lm_count+p;
                                M1_value_dot(i,node,channel) =
                                    M1_value_dot(i,p0,channel)*M1_values(i,p1,channel)
                                    +M1_values(i,p0,channel)*M1_value_dot(i,p1,channel);
                                M1_dot(i,channel) +=
                                    M1_coeff(node_types(i),node,channel)
                                    *M1_value_dot(i,node,channel);
                            });
                    }
                    team.team_barrier();
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorMDRange<
                            Kokkos::Rank<2,Kokkos::Iterate::Right>,
                            Kokkos::TeamPolicy<>::member_type>(
                                team, polynomial_nodes, channels),
                        [=] (const int node, const int channel) {
                            M1_gradient(i,node,channel) =
                                M1_coeff(node_types(i),node,channel);
                            M1_gradient_dot(i,node,channel) = 0.0;
                        });
                    team.team_barrier();
                    for (int p=M1_spec.extent(0)-1; p>=0; --p) {
                        const int p0 = M1_spec(p,0);
                        const int p1 = M1_spec(p,1);
                        const int node = lm_count+p;
                        Kokkos::parallel_for(
                            Kokkos::TeamVectorRange(team, channels),
                            [=] (const int channel) {
                                const Precision adjoint = M1_gradient(i,node,channel);
                                const Precision adjoint_dot =
                                    M1_gradient_dot(i,node,channel);
                                M1_gradient_dot(i,p0,channel) +=
                                    adjoint_dot*M1_values(i,p1,channel)
                                    +adjoint*M1_value_dot(i,p1,channel);
                                M1_gradient_dot(i,p1,channel) +=
                                    adjoint_dot*M1_values(i,p0,channel)
                                    +adjoint*M1_value_dot(i,p0,channel);
                                M1_gradient(i,p0,channel) +=
                                    adjoint*M1_values(i,p1,channel);
                                M1_gradient(i,p1,channel) +=
                                    adjoint*M1_values(i,p0,channel);
                            });
                    }
                });

            Kokkos::fence();
            M1_value_dot = decltype(M1_value_dot)();
        }

        const auto H2_from_H1 = H2_weights_for_H1;
        const auto H2_from_M1 = H2_weights_for_M1;
        H2_dot = decltype(H2_dot)("MACEField H2_dot", num_nodes, channels);
        Kokkos::parallel_for(
            "MACEField analytic H2 tangent",
            Kokkos::MDRangePolicy<
                Kokkos::Rank<2,Kokkos::Iterate::Right>,
                Kokkos::IndexType<std::size_t>>(
                {0,0}, {num_nodes,channels}),
            KOKKOS_LAMBDA (
                const std::size_t i,
                const std::size_t output) {
                double value = 0.0;
                for (int input=0; input<channels; ++input) {
                    value += H2_from_H1(node_types(i),input*channels+output)
                        *static_cast<double>(H1_dot(i,0,input));
                    value += H2_from_M1(input*channels+output)
                        *static_cast<double>(M1_dot(i,input));
                }
                H2_dot(i,output) = value;
            });

        readout_output = decltype(readout_output)(
            "MACEField directional readout", num_nodes);
        H2_adj_local = decltype(H2_adj_local)(
            "MACEField H2_adj", num_nodes, channels);
        H2_adj_dot = decltype(H2_adj_dot)(
            "MACEField H2_adj_dot", num_nodes, channels);
        auto H2_input = Kokkos::subview(H2, Kokkos::make_pair(0,num_nodes), Kokkos::ALL);
        if (readout_recompute)
            readout_2.evaluate_gradient_directional_recompute(
                H2_input, H2_dot, readout_output, H2_adj_local, H2_adj_dot);
        else
            readout_2.evaluate_gradient_directional(
                H2_input, H2_dot, readout_output, H2_adj_local, H2_adj_dot);

        Kokkos::fence();
        M1_dot = decltype(M1_dot)();
        H2_dot = decltype(H2_dot)();
        readout_output = decltype(readout_output)();

        H1_adj_local = decltype(H1_adj_local)(
            "MACEField H1_adj", num_nodes, LM_count, channels);
        H1_adj_dot = decltype(H1_adj_dot)(
            "MACEField H1_adj_dot", num_nodes, LM_count, channels);
        M1_adj_local = decltype(M1_adj_local)(
            "MACEField M1_adj", num_nodes, channels);
        M1_adj_dot = decltype(M1_adj_dot)(
            "MACEField M1_adj_dot", num_nodes, channels);
        Kokkos::deep_copy(H1_adj_local, 0.0);
        Kokkos::deep_copy(H1_adj_dot, 0.0);
        const auto readout1 = readout_1_weights;
        Kokkos::parallel_for(
            "MACEField analytic H2 reverse tangent",
            Kokkos::MDRangePolicy<
                Kokkos::Rank<2,Kokkos::Iterate::Right>,
                Kokkos::IndexType<std::size_t>>(
                {0,0}, {num_nodes,channels}),
            KOKKOS_LAMBDA (
                const std::size_t i,
                const std::size_t input) {
                Precision h1_adjoint = static_cast<Precision>(readout1(input));
                Precision h1_adjoint_dot = 0.0;
                Precision m1_adjoint = 0.0;
                Precision m1_adjoint_dot = 0.0;
                for (int output=0; output<channels; ++output) {
                    const double h1_weight =
                        H2_from_H1(node_types(i),input*channels+output);
                    const double m1_weight = H2_from_M1(input*channels+output);
                    h1_adjoint += static_cast<Precision>(
                        h1_weight*H2_adj_local(i,output));
                    h1_adjoint_dot += static_cast<Precision>(
                        h1_weight*H2_adj_dot(i,output));
                    m1_adjoint += static_cast<Precision>(
                        m1_weight*H2_adj_local(i,output));
                    m1_adjoint_dot += static_cast<Precision>(
                        m1_weight*H2_adj_dot(i,output));
                }
                H1_adj_local(i,0,input) = h1_adjoint;
                H1_adj_dot(i,0,input) = h1_adjoint_dot;
                M1_adj_local(i,input) = m1_adjoint;
                M1_adj_dot(i,input) = m1_adjoint_dot;
            });

        Kokkos::fence();
        H2_adj_local = decltype(H2_adj_local)();
        H2_adj_dot = decltype(H2_adj_dot)();

        A1_adj_local = decltype(A1_adj_local)(
            "MACEField A1_adj", num_nodes, lm_count, channels);
        A1_adj_dot = decltype(A1_adj_dot)(
            "MACEField A1_adj_dot", num_nodes, lm_count, channels);
        if (m1_polynomial_policy == M1PolynomialPolicy::recompute) {
            const int multiplication_nodes = M1_spec.extent(0);
            const int tile_channels =
                macefield_response_m1_recompute_tile_channels();
            const int vector_length = m1_recompute_vector_length();
            const int scratch_level = m1_recompute_scratch_level();
            using TeamMember = typename Kokkos::TeamPolicy<>::member_type;
            using ScratchView = Kokkos::View<
                Precision**, Kokkos::LayoutRight,
                typename TeamMember::scratch_memory_space,
                Kokkos::MemoryUnmanaged>;
            auto policy = vector_length == 1
                ? Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 1)
                : Kokkos::TeamPolicy<>(num_nodes, 1, vector_length);
            policy.set_scratch_size(
                scratch_level, Kokkos::PerTeam(
                    4*sizeof(Precision)*polynomial_nodes*tile_channels));
            Kokkos::parallel_for(
                "MACEField analytic M1 reverse tangent recompute", policy,
                KOKKOS_LAMBDA (const TeamMember& member) {
                    const std::size_t node = static_cast<std::size_t>(
                        member.league_rank());
                    ScratchView workspace(
                        member.team_scratch(scratch_level),
                        4*polynomial_nodes, tile_channels);
                    for (int channel_begin=0; channel_begin<channels;
                         channel_begin += tile_channels) {
                        const int active_channels = Kokkos::min(
                            tile_channels, channels-channel_begin);
                        Kokkos::parallel_for(
                            Kokkos::TeamVectorRange(member, active_channels),
                            [=] (const int lane) {
                                const int channel = channel_begin+lane;
                                for (int p=0; p<lm_count; ++p) {
                                    workspace(p,lane) = M1_A1(node,p,channel);
                                    workspace(polynomial_nodes+p,lane) =
                                        A1_dot(node,p,channel);
                                }
                                for (int p=0; p<multiplication_nodes; ++p) {
                                    const int p0 = M1_spec(p,0);
                                    const int p1 = M1_spec(p,1);
                                    const int output_node = lm_count+p;
                                    workspace(output_node,lane) =
                                        workspace(p0,lane)*workspace(p1,lane);
                                    workspace(polynomial_nodes+output_node,lane) =
                                        workspace(polynomial_nodes+p0,lane)
                                            *workspace(p1,lane)
                                        +workspace(p0,lane)
                                            *workspace(polynomial_nodes+p1,lane);
                                }
                                for (int p=0; p<polynomial_nodes; ++p) {
                                    workspace(2*polynomial_nodes+p,lane) =
                                        M1_coeff(node_types(node),p,channel);
                                    workspace(3*polynomial_nodes+p,lane) = 0;
                                }
                                for (int p=multiplication_nodes-1; p>=0; --p) {
                                    const int p0 = M1_spec(p,0);
                                    const int p1 = M1_spec(p,1);
                                    const int output_node = lm_count+p;
                                    const Precision gradient = workspace(
                                        2*polynomial_nodes+output_node,lane);
                                    const Precision gradient_dot = workspace(
                                        3*polynomial_nodes+output_node,lane);
                                    workspace(3*polynomial_nodes+p0,lane) +=
                                        gradient_dot*workspace(p1,lane)
                                        +gradient*workspace(
                                            polynomial_nodes+p1,lane);
                                    workspace(3*polynomial_nodes+p1,lane) +=
                                        gradient_dot*workspace(p0,lane)
                                        +gradient*workspace(
                                            polynomial_nodes+p0,lane);
                                    workspace(2*polynomial_nodes+p0,lane) +=
                                        gradient*workspace(p1,lane);
                                    workspace(2*polynomial_nodes+p1,lane) +=
                                        gradient*workspace(p0,lane);
                                }
                                for (int lm=0; lm<lm_count; ++lm) {
                                    const Precision gradient = workspace(
                                        2*polynomial_nodes+lm,lane);
                                    A1_adj_local(node,lm,channel) = gradient
                                        *M1_adj_local(node,channel);
                                    A1_adj_dot(node,lm,channel) = workspace(
                                        3*polynomial_nodes+lm,lane)
                                        *M1_adj_local(node,channel)
                                        +gradient*M1_adj_dot(node,channel);
                                }
                            });
                        member.team_barrier();
                    }
                });
            macefield_response_m1_recompute_reverse_launch_count += 1;
        } else {
            Kokkos::parallel_for(
                "MACEField analytic M1 reverse tangent",
                Kokkos::MDRangePolicy<
                    Kokkos::Rank<3,Kokkos::Iterate::Right>,
                    Kokkos::IndexType<std::size_t>>(
                    {0,0,0}, {num_nodes,lm_count,channels}),
                KOKKOS_LAMBDA (
                    const std::size_t i,
                    const std::size_t lm,
                    const std::size_t channel) {
                    const Precision gradient = M1_gradient(i,lm,channel);
                    A1_adj_local(i,lm,channel) =
                        gradient*M1_adj_local(i,channel);
                    A1_adj_dot(i,lm,channel) =
                        M1_gradient_dot(i,lm,channel)*M1_adj_local(i,channel)
                        +gradient*M1_adj_dot(i,channel);
                });
        }

        Kokkos::fence();
        M1_gradient = decltype(M1_gradient)();
        M1_gradient_dot = decltype(M1_gradient_dot)();
        M1_adj_local = decltype(M1_adj_local)();
        M1_adj_dot = decltype(M1_adj_dot)();

        const auto A1_base = A1;
        const auto A1_scale_derivs = A1_spline_derivs;
        if (A1_scaled) {
            Kokkos::parallel_for(
                "MACEField analytic A1 scaled reverse tangent",
                Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                    const int i = team.league_rank();
                    double scale;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, num_neigh(i)),
                        [=] (const int j, double& sum) {
                            const int edge = first_neigh(i)+j;
                            if (A1_scale_recompute) {
                                const int type_i = A1_response_type_to_active(
                                    node_types(i));
                                const int type_j = A1_response_type_to_active(
                                    neigh_types(edge));
                                const int edge_type = type_i <= type_j
                                    ? type_i*(2*A1_response_active_type_count-type_i-1)/2
                                        +type_j
                                    : type_j*(2*A1_response_active_type_count-type_j-1)/2
                                        +type_i;
                                sum += A1_scale_splines.evaluate_function(
                                    edge_type, r(edge), 0);
                            } else {
                                sum += A1_scale_values(edge,0);
                            }
                        }, scale);
                    scale += 1.0;
                    double contraction;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, lm_count*channels),
                        [=] (const int index, double& sum) {
                            const int lm = index/channels;
                            const int channel = index%channels;
                            sum += static_cast<double>(A1_adj_dot(i,lm,channel))
                                *static_cast<double>(A1_base(i,lm,channel));
                            sum += static_cast<double>(A1_adj_local(i,lm,channel))
                                *static_cast<double>(A1_dot(i,lm,channel));
                        }, contraction);
                    if (include_force_derivative) {
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(team, num_neigh(i)),
                            [=] (const int j) {
                            const std::size_t edge = static_cast<std::size_t>(
                                first_neigh(i)+j);
                            const std::size_t coordinate_offset = 3*edge;
                            const std::size_t force_offset =
                                seed*coordinate_count+coordinate_offset;
                            double scale_value;
                            double scale_derivative;
                            if (A1_scale_recompute) {
                                const int type_i = A1_response_type_to_active(
                                    node_types(i));
                                const int type_j = A1_response_type_to_active(
                                    neigh_types(edge));
                                const int edge_type = type_i <= type_j
                                    ? type_i*(2*A1_response_active_type_count-type_i-1)/2
                                        +type_j
                                    : type_j*(2*A1_response_active_type_count-type_j-1)/2
                                        +type_i;
                                A1_scale_splines.evaluate_function(
                                    edge_type, r(edge), 0,
                                    scale_value, scale_derivative);
                            } else {
                                scale_derivative = A1_scale_derivs(edge,0);
                            }
                            const double factor = contraction/scale
                                *scale_derivative;
                            response_forces(force_offset) +=
                                factor*(compact_geometry
                                    ? static_cast<double>(
                                        unit_direction(coordinate_offset))
                                    : xyz(coordinate_offset)/r(edge));
                            response_forces(force_offset+1) +=
                                factor*(compact_geometry
                                    ? static_cast<double>(
                                        unit_direction(coordinate_offset+1))
                                    : xyz(coordinate_offset+1)/r(edge));
                            response_forces(force_offset+2) +=
                                factor*(compact_geometry
                                    ? static_cast<double>(
                                        unit_direction(coordinate_offset+2))
                                    : xyz(coordinate_offset+2)/r(edge));
                            });
                    }
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, lm_count*channels),
                        [=] (const int index) {
                            const int lm = index/channels;
                            const int channel = index%channels;
                            A1_adj_local(i,lm,channel) /= scale;
                            A1_adj_dot(i,lm,channel) /= scale;
                        });
                });
        }

        Kokkos::fence();
        A1_dot = decltype(A1_dot)();

        const auto A1_matrix_trans = A1_weights_trans;
        dPhi1_local = decltype(dPhi1_local)(
            "MACEField dPhi1", num_nodes, phi_outputs, channels);
        dPhi1_dot = decltype(dPhi1_dot)(
            "MACEField dPhi1_dot", num_nodes, phi_outputs, channels);
        dPhi1r_local = decltype(dPhi1r_local)(
            "MACEField dPhi1r", num_nodes, phi_rows, channels);
        dPhi1r_dot = decltype(dPhi1r_dot)(
            "MACEField dPhi1r_dot", num_nodes, phi_rows, channels);
        Kokkos::parallel_for(
            "MACEField analytic A1 reverse",
            Kokkos::TeamPolicy<>(num_nodes*(lmax+1), Kokkos::AUTO),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                const int i = team.league_rank()/(lmax+1);
                const int l = team.league_rank()%(lmax+1);
                int lme = 0;
                int multiplicity = 0;
                for (int p=0; p<phi_l.extent(0); ++p) {
                    if (phi_l(p) < l)
                        lme += 2*phi_l(p)+1;
                    if (phi_l(p) == l)
                        ++multiplicity;
                }
                auto base_input = Kokkos::subview(
                    A1_adj_local, i,
                    Kokkos::make_pair(l*l,l*l+2*l+1), Kokkos::ALL);
                auto dot_input = Kokkos::subview(
                    A1_adj_dot, i,
                    Kokkos::make_pair(l*l,l*l+2*l+1), Kokkos::ALL);
                auto base_output = Kokkos::View<Precision**,Kokkos::LayoutRight,
                    Kokkos::MemoryUnmanaged>(
                        &dPhi1_local(i,lme,0), 2*l+1, multiplicity*channels);
                auto dot_output = Kokkos::View<Precision**,Kokkos::LayoutRight,
                    Kokkos::MemoryUnmanaged>(
                        &dPhi1_dot(i,lme,0), 2*l+1, multiplicity*channels);
                KokkosBatched::TeamGemm<
                    Kokkos::TeamPolicy<>::member_type,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Algo::Gemm::Blocked>::invoke(
                        team, 1.0, base_input, A1_matrix_trans(l),
                        0.0, base_output);
                team.team_barrier();
                KokkosBatched::TeamGemm<
                    Kokkos::TeamPolicy<>::member_type,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Algo::Gemm::Blocked>::invoke(
                        team, 1.0, dot_input, A1_matrix_trans(l),
                        0.0, dot_output);
            });

        Kokkos::fence();
        A1_adj_local = decltype(A1_adj_local)();
        A1_adj_dot = decltype(A1_adj_dot)();

        Kokkos::deep_copy(dPhi1r_local, 0.0);
        Kokkos::deep_copy(dPhi1r_dot, 0.0);
        Kokkos::parallel_for(
            "MACEField analytic CG reverse tangent",
            Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                const int i = team.league_rank();
                for (int p=0; p<phi_cg.extent(0); ++p) {
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(team, channels),
                        [=] (const int channel) {
                            const int row = phi_row(p);
                            const int output = phi_lme(p);
                            dPhi1r_local(i,row,channel) +=
                                phi_cg(p)*dPhi1_local(i,output,channel);
                            dPhi1r_dot(i,row,channel) +=
                                phi_cg(p)*dPhi1_dot(i,output,channel);
                        });
                }
            });

        Kokkos::fence();
        dPhi1_local = decltype(dPhi1_local)();
        dPhi1_dot = decltype(dPhi1_dot)();

        const auto radial1_deriv = R1_deriv;
        const auto harmonics_grad = Y_grad;
        const bool recompute_harmonic_gradients =
            include_force_derivative && harmonics_grad.data() == nullptr;
        const auto H1_base = H1;
        const int phi_paths = Phi1_l.extent(0);
        const auto path_row_offsets = Phi1_path_row_offsets;
        using ResponseTeamMember =
            typename Kokkos::TeamPolicy<>::member_type;
        using ResponseScratchView = Kokkos::View<
            Precision*, typename ResponseTeamMember::scratch_memory_space,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
#ifdef KOKKOS_ENABLE_CUDA
        const auto edge_receivers = response_edge_receivers;
        if (response_mode != MACEStreamedEdgesMode::materialized
            && supports_fused_streamed_reverse()
            && edge_receivers.extent(0) == response_edges) {
            macefield_response_phi1_fused_launch_count += 1;
            constexpr int edges_per_team = 8;
            auto phi_fused_policy = Kokkos::TeamPolicy<>(
                static_cast<int>(
                    (response_edges+edges_per_team-1)/edges_per_team),
                edges_per_team, 32);
            if (recompute_harmonic_gradients)
                phi_fused_policy.set_scratch_size(
                    0, Kokkos::PerTeam(
                        edges_per_team*3*16*sizeof(Precision)));
            Kokkos::parallel_for(
                "MACEField analytic Phi1 reverse tangent fused",
                phi_fused_policy,
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                    const std::size_t edge_begin =
                        static_cast<std::size_t>(team.league_rank())
                        *edges_per_team;
                    const int edge_count = static_cast<int>(Kokkos::min(
                        static_cast<std::size_t>(edges_per_team),
                        response_edges-edge_begin));
                    ResponseScratchView direct_gradient_storage(
                        team.team_scratch(0),
                        recompute_harmonic_gradients
                            ? edges_per_team*3*16 : 0);
                    const int edge_offset = team.team_rank();
                    const bool edge_active = edge_offset < edge_count;
                    const std::size_t edge = edge_begin+edge_offset;
                    Precision* direct_gradients =
                        recompute_harmonic_gradients
                        ? direct_gradient_storage.data()+edge_offset*3*16
                        : nullptr;
                    if (recompute_harmonic_gradients) {
                        Kokkos::single(Kokkos::PerThread(team), [=]() {
                            if (!edge_active)
                                return;
                            const std::size_t coordinate_offset = 3*edge;
                            const Precision direction[3] = {
                                compact_geometry
                                    ? unit_direction(coordinate_offset)
                                    : static_cast<Precision>(
                                        xyz(coordinate_offset)/r(edge)),
                                compact_geometry
                                    ? unit_direction(coordinate_offset+1)
                                    : static_cast<Precision>(
                                        xyz(coordinate_offset+1)/r(edge)),
                                compact_geometry
                                    ? unit_direction(coordinate_offset+2)
                                    : static_cast<Precision>(
                                        xyz(coordinate_offset+2)/r(edge))};
                            symmetrix::
                                normalized_spherical_harmonic_gradients_from_direction<3>(
                                    direction,
                                    static_cast<Precision>(r(edge)),
                                    direct_gradients);
                        });
                        team.team_barrier();
                    }
                    if (!edge_active)
                        return;
                    const std::size_t coordinate_offset = 3*edge;
                    const std::size_t harmonic_offset =
                        edge*static_cast<std::size_t>(lm_count);
                    const std::size_t gradient_offset = 3*harmonic_offset;
                    const std::size_t force_offset =
                        seed*coordinate_count+coordinate_offset;
                    const int i = edge_receivers(edge);
                    const int neighbor = neigh_indices(edge);
                    Precision x_over_r = 0;
                    Precision y_over_r = 0;
                    Precision z_over_r = 0;
                    if (include_force_derivative) {
                        x_over_r = compact_geometry
                            ? unit_direction(coordinate_offset)
                            : static_cast<Precision>(
                                xyz(coordinate_offset)/r(edge));
                        y_over_r = compact_geometry
                            ? unit_direction(coordinate_offset+1)
                            : static_cast<Precision>(
                                xyz(coordinate_offset+1)/r(edge));
                        z_over_r = compact_geometry
                            ? unit_direction(coordinate_offset+2)
                            : static_cast<Precision>(
                                xyz(coordinate_offset+2)/r(edge));
                    }
                    Precision force_x, force_y, force_z;
                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team, channels),
                        [=] (const int channel,
                             Precision& local_x,
                             Precision& local_y,
                             Precision& local_z) {
                            Precision base_contributions[
                                streamed_fused_max_num_LM] = {};
                            Precision dot_contributions[
                                streamed_fused_max_num_LM] = {};
                            for (int path=0; path<phi_paths; ++path) {
                                const Precision radial =
                                    radial1(edge,path*channels+channel);
                                Precision radial_derivative = 0;
                                if (include_force_derivative)
                                    radial_derivative = radial1_deriv(
                                        edge,path*channels+channel);
                                const int row_begin = path_row_offsets(path);
                                const int row_end = path_row_offsets(path+1);
                                for (int row=row_begin; row<row_end; ++row) {
                                    const int lm1 = phi_lm1(row);
                                    const int lm2 = phi_lm2(row);
                                    const Precision y =
                                        harmonics(harmonic_offset+lm1);
                                    const Precision base_adjoint =
                                        dPhi1r_local(i,row,channel);
                                    const Precision adjoint_dot =
                                        dPhi1r_dot(i,row,channel);
                                    if (include_force_derivative) {
                                        const Precision base_feature =
                                            H1_base(neighbor,lm2,channel);
                                        const Precision feature_dot =
                                            H1_dot(neighbor,lm2,channel);
                                        const Precision radial_base =
                                            radial_derivative*y*base_feature;
                                        const Precision radial_dot =
                                            radial_derivative*y*feature_dot;
                                        const Precision angular_base =
                                            radial*base_feature;
                                        const Precision angular_dot =
                                            radial*feature_dot;
                                        const Precision gradient_x =
                                            recompute_harmonic_gradients
                                            ? direct_gradients[lm1]
                                            : harmonics_grad(
                                                gradient_offset+lm1);
                                        const Precision gradient_y =
                                            recompute_harmonic_gradients
                                            ? direct_gradients[lm_count+lm1]
                                            : harmonics_grad(
                                                gradient_offset+lm_count+lm1);
                                        const Precision gradient_z =
                                            recompute_harmonic_gradients
                                            ? direct_gradients[2*lm_count+lm1]
                                            : harmonics_grad(
                                                gradient_offset+2*lm_count+lm1);
                                        local_x -= adjoint_dot*(
                                            radial_base*x_over_r
                                            +angular_base*gradient_x);
                                        local_x -= base_adjoint*(
                                            radial_dot*x_over_r
                                            +angular_dot*gradient_x);
                                        local_y -= adjoint_dot*(
                                            radial_base*y_over_r
                                            +angular_base*gradient_y);
                                        local_y -= base_adjoint*(
                                            radial_dot*y_over_r
                                            +angular_dot*gradient_y);
                                        local_z -= adjoint_dot*(
                                            radial_base*z_over_r
                                            +angular_base*gradient_z);
                                        local_z -= base_adjoint*(
                                            radial_dot*z_over_r
                                            +angular_dot*gradient_z);
                                    }
                                    const Precision source_factor = radial*y;
                                    base_contributions[lm2] +=
                                        source_factor*base_adjoint;
                                    dot_contributions[lm2] +=
                                        source_factor*adjoint_dot;
                                }
                            }
                            for (int lm2=0; lm2<LM_count; ++lm2) {
                                Kokkos::atomic_add(
                                    &H1_adj_local(neighbor,lm2,channel),
                                    base_contributions[lm2]);
                                Kokkos::atomic_add(
                                    &H1_adj_dot(neighbor,lm2,channel),
                                    dot_contributions[lm2]);
                            }
                        }, force_x, force_y, force_z);
                    if (include_force_derivative) {
                        Kokkos::single(Kokkos::PerThread(team), [=]() {
                            response_forces(force_offset) +=
                                static_cast<double>(force_x);
                            response_forces(force_offset+1) +=
                                static_cast<double>(force_y);
                            response_forces(force_offset+2) +=
                                static_cast<double>(force_z);
                        });
                    }
                });
        } else
#endif
        {
            macefield_response_phi1_generic_launch_count += 1;
            auto phi_reverse_policy = Kokkos::TeamPolicy<>(
                num_nodes, Kokkos::AUTO, 32);
            if (recompute_harmonic_gradients)
                phi_reverse_policy.set_scratch_size(
                    0, Kokkos::PerTeam(3*16*sizeof(Precision)));
            Kokkos::parallel_for(
                "MACEField analytic Phi1 reverse tangent",
                phi_reverse_policy,
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                ResponseScratchView direct_gradient_storage(
                    team.team_scratch(0),
                    recompute_harmonic_gradients ? 3*16 : 0);
                const int i = team.league_rank();
                const int edge_begin = first_neigh(i);
                for (int j=0; j<num_neigh(i); ++j) {
                    const std::size_t edge =
                        static_cast<std::size_t>(edge_begin+j);
                    const std::size_t coordinate_offset = 3*edge;
                    const std::size_t harmonic_offset =
                        edge*static_cast<std::size_t>(lm_count);
                    const std::size_t gradient_offset = 3*harmonic_offset;
                    const std::size_t force_offset =
                        seed*coordinate_count+coordinate_offset;
                    const int neighbor = neigh_indices(edge);
                    const double direction_x = include_force_derivative
                        ? (compact_geometry
                            ? static_cast<double>(unit_direction(coordinate_offset))
                            : xyz(coordinate_offset)/r(edge))
                        : 0.0;
                    const double direction_y = include_force_derivative
                        ? (compact_geometry
                            ? static_cast<double>(unit_direction(coordinate_offset+1))
                            : xyz(coordinate_offset+1)/r(edge))
                        : 0.0;
                    const double direction_z = include_force_derivative
                        ? (compact_geometry
                            ? static_cast<double>(unit_direction(coordinate_offset+2))
                            : xyz(coordinate_offset+2)/r(edge))
                        : 0.0;
                    if (recompute_harmonic_gradients) {
                        Kokkos::single(Kokkos::PerTeam(team), [=]() {
                            const Precision direction[3] = {
                                static_cast<Precision>(direction_x),
                                static_cast<Precision>(direction_y),
                                static_cast<Precision>(direction_z)};
                            symmetrix::
                                normalized_spherical_harmonic_gradients_from_direction<3>(
                                    direction,
                                    static_cast<Precision>(r(edge)),
                                    direct_gradient_storage.data());
                        });
                        team.team_barrier();
                    }
                    double force_x, force_y, force_z;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, phi_rows),
                        [=] (const int row,
                             double& local_x,
                             double& local_y,
                             double& local_z) {
                            const int lm1 = phi_lm1(row);
                            const int lm2 = phi_lm2(row);
                            const int path = phi_path(row);
                            const Precision y = harmonics(harmonic_offset+lm1);
                            double row_x, row_y, row_z;
                            Kokkos::parallel_reduce(
                                Kokkos::ThreadVectorRange(team, channels),
                                [=] (const int channel,
                                     double& channel_x,
                                     double& channel_y,
                                     double& channel_z) {
                                    const Precision radial =
                                        radial1(edge,path*channels+channel);
                                    Precision radial_derivative = 0;
                                    if (include_force_derivative) {
                                        radial_derivative = radial1_deriv(
                                            edge,path*channels+channel);
                                    }
                                    const Precision base_adjoint =
                                        dPhi1r_local(i,row,channel);
                                    const Precision adjoint_dot =
                                        dPhi1r_dot(i,row,channel);
                                    if (include_force_derivative) {
                                        const Precision base_feature =
                                            H1_base(neighbor,lm2,channel);
                                        const Precision feature_dot =
                                            H1_dot(neighbor,lm2,channel);
                                        const double radial_base =
                                            static_cast<double>(
                                                radial_derivative*y
                                                *base_feature);
                                        const double radial_dot =
                                            static_cast<double>(
                                                radial_derivative*y
                                                *feature_dot);
                                        const double angular_base =
                                            static_cast<double>(
                                                radial*base_feature);
                                        const double angular_dot =
                                            static_cast<double>(
                                                radial*feature_dot);
                                        const double base_adj =
                                            static_cast<double>(base_adjoint);
                                        const double dot_adj =
                                            static_cast<double>(adjoint_dot);
                                        const double gradient_x =
                                            recompute_harmonic_gradients
                                            ? static_cast<double>(
                                                direct_gradient_storage(lm1))
                                            : harmonics_grad(
                                                gradient_offset+lm1);
                                        const double gradient_y =
                                            recompute_harmonic_gradients
                                            ? static_cast<double>(
                                                direct_gradient_storage(
                                                    lm_count+lm1))
                                            : harmonics_grad(
                                                gradient_offset+lm_count+lm1);
                                        const double gradient_z =
                                            recompute_harmonic_gradients
                                            ? static_cast<double>(
                                                direct_gradient_storage(
                                                    2*lm_count+lm1))
                                            : harmonics_grad(
                                                gradient_offset+2*lm_count+lm1);
                                        channel_x -= dot_adj*(
                                            radial_base*direction_x
                                            +angular_base*gradient_x);
                                        channel_x -= base_adj*(
                                            radial_dot*direction_x
                                            +angular_dot*gradient_x);
                                        channel_y -= dot_adj*(
                                            radial_base*direction_y
                                            +angular_base*gradient_y);
                                        channel_y -= base_adj*(
                                            radial_dot*direction_y
                                            +angular_dot*gradient_y);
                                        channel_z -= dot_adj*(
                                            radial_base*direction_z
                                            +angular_base*gradient_z);
                                        channel_z -= base_adj*(
                                            radial_dot*direction_z
                                            +angular_dot*gradient_z);
                                    }
                                    const Precision source_factor = radial*y;
                                    Kokkos::atomic_add(
                                        &H1_adj_local(neighbor,lm2,channel),
                                        source_factor*base_adjoint);
                                    Kokkos::atomic_add(
                                        &H1_adj_dot(neighbor,lm2,channel),
                                        source_factor*adjoint_dot);
                                }, row_x, row_y, row_z);
                            local_x += row_x;
                            local_y += row_y;
                            local_z += row_z;
                        }, force_x, force_y, force_z);
                    if (include_force_derivative) {
                        Kokkos::single(Kokkos::PerTeam(team), [=]() {
                            response_forces(force_offset) += force_x;
                            response_forces(force_offset+1) += force_y;
                            response_forces(force_offset+2) += force_z;
                        });
                    }
                    if (recompute_harmonic_gradients)
                        team.team_barrier();
                }
                });
        }

        Kokkos::fence();
        dPhi1r_local = decltype(dPhi1r_local)();
        dPhi1r_dot = decltype(dPhi1r_dot)();
        H1_dot = decltype(H1_dot)();

        H1_product_adj_dot = decltype(H1_product_adj_dot)(
            "MACEField product adjoint tangent", num_nodes, LM_count, channels);
        Kokkos::parallel_reduce(
            "MACEField analytic field reverse tangent",
            Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(
                0, static_cast<std::size_t>(num_nodes)*LM_count*channels),
            AnalyticFieldReverseReducer<Precision>{
                channels,
                LM_count,
                field_entry_count,
                seed,
                H1_adj_local,
                H1_adj_dot,
                H1_before_field,
                H1_product_adj_dot,
                H1_up_weights,
                field_input_lm,
                field_component,
                field_output_lm,
                field_path,
                field_coefficient,
                field_up_matrix,
                electric_field,
                response_hessian
            });

        Kokkos::fence();
        H1_adj_local = decltype(H1_adj_local)();
        H1_adj_dot = decltype(H1_adj_dot)();
        if (!include_force_derivative) {
            H1_product_adj_dot = decltype(H1_product_adj_dot)();
            continue;
        }

        const auto product_weights = H1_product_weights;
        Kokkos::parallel_for(
            "MACEField analytic H1 product reverse tangent",
            Kokkos::TeamPolicy<>(num_nodes*(Lmax+1), Kokkos::AUTO),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                const int i = team.league_rank()/(Lmax+1);
                const int l = team.league_rank()%(Lmax+1);
                auto input = Kokkos::subview(
                    H1_product_adj_dot, i,
                    Kokkos::make_pair(l*l,l*(l+2)+1), Kokkos::ALL);
                auto weights = Kokkos::subview(
                    product_weights, l, Kokkos::ALL, Kokkos::ALL);
                auto output = Kokkos::subview(
                    M0_adj_dot, i,
                    Kokkos::make_pair(l*l,l*(l+2)+1), Kokkos::ALL);
                KokkosBatched::TeamGemm<
                    Kokkos::TeamPolicy<>::member_type,
                    KokkosBatched::Trans::NoTranspose,
                    KokkosBatched::Trans::Transpose,
                    KokkosBatched::Algo::Gemm::Unblocked>::invoke(
                        team, 1.0, input, weights, 0.0, output);
            });

        Kokkos::fence();
        H1_product_adj_dot = decltype(H1_product_adj_dot)();

        Kokkos::deep_copy(A0_adj_dot, 0.0);
        if (use_m0_module()) {
            const auto execution_space = Kokkos::DefaultExecutionSpace();
            launch_M0_module_reverse(
                execution_space, num_nodes, node_types, A0, M0_adj_dot,
                A0_adj_dot, mh0_a0_scale_adjoint, false);
        } else {
            const auto M0_specs = M0_poly_spec;
            const auto M0_coeffs = M0_poly_coeff;
            const auto M0_values = M0_poly_values;
            for (int LM=0; LM<LM_count; ++LM) {
                const auto spec = M0_specs(LM);
                const auto coefficients = M0_coeffs(LM);
                const auto values = M0_values(LM);
                const int graph_nodes = coefficients.extent(1);
                Kokkos::View<Precision***,Kokkos::LayoutRight> graph_adj_dot(
                    "MACEField M0 graph adjoint tangent",
                    num_nodes, graph_nodes, channels);
                Kokkos::parallel_for(
                    "MACEField analytic M0 reverse tangent",
                    Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO, 32),
                    KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                        const int i = team.league_rank();
                        Kokkos::parallel_for(
                            Kokkos::TeamVectorMDRange<
                                Kokkos::Rank<2,Kokkos::Iterate::Right>,
                                Kokkos::TeamPolicy<>::member_type>(
                                    team, graph_nodes, channels),
                            [=] (const int node, const int channel) {
                                graph_adj_dot(i,node,channel) =
                                    coefficients(node_types(i),node,channel)
                                    *M0_adj_dot(i,LM,channel);
                            });
                        team.team_barrier();
                        for (int p=spec.extent(0)-1; p>=0; --p) {
                            const int p0 = spec(p,0);
                            const int p1 = spec(p,1);
                            const int node = lm_count+p;
                            Kokkos::parallel_for(
                                Kokkos::TeamVectorRange(team, channels),
                                [=] (const int channel) {
                                    const Precision adjoint =
                                        graph_adj_dot(i,node,channel);
                                    graph_adj_dot(i,p0,channel) +=
                                        adjoint*values(i,p1,channel);
                                    graph_adj_dot(i,p1,channel) +=
                                        adjoint*values(i,p0,channel);
                                });
                        }
                        team.team_barrier();
                        Kokkos::parallel_for(
                            Kokkos::TeamVectorMDRange<
                                Kokkos::Rank<2,Kokkos::Iterate::Right>,
                                Kokkos::TeamPolicy<>::member_type>(
                                    team, lm_count, channels),
                            [=] (const int lm, const int channel) {
                                Kokkos::atomic_add(
                                    &A0_adj_dot(i,lm,channel),
                                    graph_adj_dot(i,lm,channel));
                            });
                    });
            }
        }

        const auto A0_base = A0;
        const auto A0_scale_values = A0_spline_values;
        const auto A0_scale_derivs = A0_spline_derivs;
        const auto A0_scale_splines = A0_splines;
        const auto response_type_to_active = type_to_active;
        const int response_active_type_count = num_active_types;
        const bool A0_scale_recompute =
            response_mode != MACEStreamedEdgesMode::materialized;
        if (A0_scaled) {
            Kokkos::parallel_for(
                "MACEField analytic A0 scaled reverse tangent",
                Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                    const int i = team.league_rank();
                    double scale;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, num_neigh(i)),
                        [=] (const int j, double& sum) {
                            const int edge = first_neigh(i)+j;
                            if (A0_scale_recompute) {
                                const int type_i = response_type_to_active(
                                    node_types(i));
                                const int type_j = response_type_to_active(
                                    neigh_types(edge));
                                const int edge_type = type_i <= type_j
                                    ? type_i*(2*response_active_type_count-type_i-1)/2
                                        +type_j
                                    : type_j*(2*response_active_type_count-type_j-1)/2
                                        +type_i;
                                sum += A0_scale_splines.evaluate_function(
                                    edge_type, r(edge), 0);
                            } else {
                                sum += A0_scale_values(edge,0);
                            }
                        }, scale);
                    scale += 1.0;
                    double contraction;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, lm_count*channels),
                        [=] (const int index, double& sum) {
                            const int lm = index/channels;
                            const int channel = index%channels;
                            sum += static_cast<double>(A0_adj_dot(i,lm,channel))
                                *static_cast<double>(A0_base(i,lm,channel));
                        }, contraction);
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, num_neigh(i)),
                        [=] (const int j) {
                            const std::size_t edge = static_cast<std::size_t>(
                                first_neigh(i)+j);
                            const std::size_t coordinate_offset = 3*edge;
                            const std::size_t force_offset =
                                seed*coordinate_count+coordinate_offset;
                            double derivative;
                            if (A0_scale_recompute) {
                                const int type_i = response_type_to_active(
                                    node_types(i));
                                const int type_j = response_type_to_active(
                                    neigh_types(edge));
                                const int edge_type = type_i <= type_j
                                    ? type_i*(2*response_active_type_count-type_i-1)/2
                                        +type_j
                                    : type_j*(2*response_active_type_count-type_j-1)/2
                                        +type_i;
                                double value;
                                A0_scale_splines.evaluate_function(
                                    edge_type, r(edge), 0, value, derivative);
                            } else {
                                derivative = A0_scale_derivs(edge,0);
                            }
                            const double factor = contraction/scale*derivative;
                            response_forces(force_offset) +=
                                factor*(compact_geometry
                                    ? static_cast<double>(
                                        unit_direction(coordinate_offset))
                                    : xyz(coordinate_offset)/r(edge));
                            response_forces(force_offset+1) +=
                                factor*(compact_geometry
                                    ? static_cast<double>(
                                        unit_direction(coordinate_offset+1))
                                    : xyz(coordinate_offset+1)/r(edge));
                            response_forces(force_offset+2) +=
                                factor*(compact_geometry
                                    ? static_cast<double>(
                                        unit_direction(coordinate_offset+2))
                                    : xyz(coordinate_offset+2)/r(edge));
                        });
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, lm_count*channels),
                        [=] (const int index) {
                            A0_adj_dot(i,index/channels,index%channels) /= scale;
                        });
                });
        }

        const auto radial0 = R0;
        const auto radial0_deriv = R0_deriv;
#ifdef KOKKOS_ENABLE_CUDA
        if (response_mode != MACEStreamedEdgesMode::materialized
            && response_edge_receivers.extent(0) == response_edges) {
            macefield_response_a0_fused_launch_count += 1;
            constexpr int edges_per_team = 8;
            const auto edge_receivers = response_edge_receivers;
            auto a0_fused_policy = Kokkos::TeamPolicy<>(
                static_cast<int>(
                    (response_edges+edges_per_team-1)/edges_per_team),
                edges_per_team, 32);
            if (recompute_harmonic_gradients)
                a0_fused_policy.set_scratch_size(
                    0, Kokkos::PerTeam(
                        edges_per_team*3*16*sizeof(Precision)));
            Kokkos::parallel_for(
                "MACEField analytic A0 reverse tangent fused",
                a0_fused_policy,
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                    const std::size_t edge_begin =
                        static_cast<std::size_t>(team.league_rank())
                        *edges_per_team;
                    const int edge_count = static_cast<int>(Kokkos::min(
                        static_cast<std::size_t>(edges_per_team),
                        response_edges-edge_begin));
                    ResponseScratchView direct_gradient_storage(
                        team.team_scratch(0),
                        recompute_harmonic_gradients
                            ? edges_per_team*3*16 : 0);
                    const int edge_offset = team.team_rank();
                    const bool edge_active = edge_offset < edge_count;
                    const std::size_t edge = edge_begin+edge_offset;
                    Precision* direct_gradients =
                        recompute_harmonic_gradients
                        ? direct_gradient_storage.data()+edge_offset*3*16
                        : nullptr;
                    if (recompute_harmonic_gradients) {
                        Kokkos::single(Kokkos::PerThread(team), [=]() {
                            if (!edge_active)
                                return;
                            const std::size_t coordinate_offset = 3*edge;
                            const Precision direction[3] = {
                                compact_geometry
                                    ? unit_direction(coordinate_offset)
                                    : static_cast<Precision>(
                                        xyz(coordinate_offset)/r(edge)),
                                compact_geometry
                                    ? unit_direction(coordinate_offset+1)
                                    : static_cast<Precision>(
                                        xyz(coordinate_offset+1)/r(edge)),
                                compact_geometry
                                    ? unit_direction(coordinate_offset+2)
                                    : static_cast<Precision>(
                                        xyz(coordinate_offset+2)/r(edge))};
                            symmetrix::
                                normalized_spherical_harmonic_gradients_from_direction<3>(
                                    direction,
                                    static_cast<Precision>(r(edge)),
                                    direct_gradients);
                        });
                        team.team_barrier();
                    }
                    if (!edge_active)
                        return;
                    const std::size_t coordinate_offset = 3*edge;
                    const std::size_t harmonic_offset =
                        edge*static_cast<std::size_t>(lm_count);
                    const std::size_t gradient_offset = 3*harmonic_offset;
                    const std::size_t force_offset =
                        seed*coordinate_count+coordinate_offset;
                    const int i = edge_receivers(edge);
                    const Precision x_over_r = compact_geometry
                        ? unit_direction(coordinate_offset)
                        : static_cast<Precision>(
                            xyz(coordinate_offset)/r(edge));
                    const Precision y_over_r = compact_geometry
                        ? unit_direction(coordinate_offset+1)
                        : static_cast<Precision>(
                            xyz(coordinate_offset+1)/r(edge));
                    const Precision z_over_r = compact_geometry
                        ? unit_direction(coordinate_offset+2)
                        : static_cast<Precision>(
                            xyz(coordinate_offset+2)/r(edge));
                    Precision force_x, force_y, force_z;
                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team, channels),
                        [=] (const int channel,
                             Precision& local_x,
                             Precision& local_y,
                             Precision& local_z) {
                            for (int l=0; l<=lmax; ++l) {
                                const Precision radial =
                                    radial0(edge,l*channels+channel);
                                const Precision radial_derivative =
                                    radial0_deriv(edge,l*channels+channel);
                                for (int lm=l*l; lm<(l+1)*(l+1); ++lm) {
                                    const Precision adjoint =
                                        A0_adj_dot(i,lm,channel);
                                    const Precision radial_force =
                                        radial_derivative
                                        *harmonics(harmonic_offset+lm)*adjoint;
                                    const Precision angular_force =
                                        radial*adjoint;
                                    const Precision gradient_x =
                                        recompute_harmonic_gradients
                                        ? direct_gradients[lm]
                                        : harmonics_grad(gradient_offset+lm);
                                    const Precision gradient_y =
                                        recompute_harmonic_gradients
                                        ? direct_gradients[lm_count+lm]
                                        : harmonics_grad(
                                            gradient_offset+lm_count+lm);
                                    const Precision gradient_z =
                                        recompute_harmonic_gradients
                                        ? direct_gradients[2*lm_count+lm]
                                        : harmonics_grad(
                                            gradient_offset+2*lm_count+lm);
                                    local_x -= radial_force*x_over_r
                                        +angular_force*gradient_x;
                                    local_y -= radial_force*y_over_r
                                        +angular_force*gradient_y;
                                    local_z -= radial_force*z_over_r
                                        +angular_force*gradient_z;
                                }
                            }
                        }, force_x, force_y, force_z);
                    Kokkos::single(Kokkos::PerThread(team), [=]() {
                        response_forces(force_offset) +=
                            static_cast<double>(force_x);
                        response_forces(force_offset+1) +=
                            static_cast<double>(force_y);
                        response_forces(force_offset+2) +=
                            static_cast<double>(force_z);
                    });
                });
        } else
#endif
        {
            macefield_response_a0_generic_launch_count += 1;
            auto A0_reverse_policy = Kokkos::TeamPolicy<>(
                num_nodes, Kokkos::AUTO, 32);
            if (recompute_harmonic_gradients)
                A0_reverse_policy.set_scratch_size(
                    0, Kokkos::PerTeam(3*16*sizeof(Precision)));
            Kokkos::parallel_for(
                "MACEField analytic A0 reverse tangent",
                A0_reverse_policy,
                KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team) {
                ResponseScratchView direct_gradient_storage(
                    team.team_scratch(0),
                    recompute_harmonic_gradients ? 3*16 : 0);
                const int i = team.league_rank();
                const int edge_begin = first_neigh(i);
                for (int j=0; j<num_neigh(i); ++j) {
                    const std::size_t edge =
                        static_cast<std::size_t>(edge_begin+j);
                    const std::size_t coordinate_offset = 3*edge;
                    const std::size_t harmonic_offset =
                        edge*static_cast<std::size_t>(lm_count);
                    const std::size_t gradient_offset = 3*harmonic_offset;
                    const std::size_t force_offset =
                        seed*coordinate_count+coordinate_offset;
                    const double direction_x = compact_geometry
                        ? static_cast<double>(unit_direction(coordinate_offset))
                        : xyz(coordinate_offset)/r(edge);
                    const double direction_y = compact_geometry
                        ? static_cast<double>(unit_direction(coordinate_offset+1))
                        : xyz(coordinate_offset+1)/r(edge);
                    const double direction_z = compact_geometry
                        ? static_cast<double>(unit_direction(coordinate_offset+2))
                        : xyz(coordinate_offset+2)/r(edge);
                    if (recompute_harmonic_gradients) {
                        Kokkos::single(Kokkos::PerTeam(team), [=]() {
                            const Precision direction[3] = {
                                static_cast<Precision>(direction_x),
                                static_cast<Precision>(direction_y),
                                static_cast<Precision>(direction_z)};
                            symmetrix::
                                normalized_spherical_harmonic_gradients_from_direction<3>(
                                    direction,
                                    static_cast<Precision>(r(edge)),
                                    direct_gradient_storage.data());
                        });
                        team.team_barrier();
                    }
                    double force_x, force_y, force_z;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, lmax+1),
                        [=] (const int l,
                             double& local_x,
                             double& local_y,
                             double& local_z) {
                            double l_x, l_y, l_z;
                            Kokkos::parallel_reduce(
                                Kokkos::ThreadVectorRange(team, channels),
                                [=] (const int channel,
                                     double& channel_x,
                                     double& channel_y,
                                     double& channel_z) {
                                    for (int lm=l*l; lm<(l+1)*(l+1); ++lm) {
                                        const double adjoint = static_cast<double>(
                                            A0_adj_dot(i,lm,channel));
                                        const double radial = static_cast<double>(
                                            radial0_deriv(edge,l*channels+channel))
                                            *static_cast<double>(
                                                harmonics(harmonic_offset+lm))
                                            *adjoint;
                                        const double angular = static_cast<double>(
                                            radial0(edge,l*channels+channel))*adjoint;
                                        const double gradient_x =
                                            recompute_harmonic_gradients
                                            ? static_cast<double>(
                                                direct_gradient_storage(lm))
                                            : harmonics_grad(
                                                gradient_offset+lm);
                                        const double gradient_y =
                                            recompute_harmonic_gradients
                                            ? static_cast<double>(
                                                direct_gradient_storage(
                                                    lm_count+lm))
                                            : harmonics_grad(
                                                gradient_offset+lm_count+lm);
                                        const double gradient_z =
                                            recompute_harmonic_gradients
                                            ? static_cast<double>(
                                                direct_gradient_storage(
                                                    2*lm_count+lm))
                                            : harmonics_grad(
                                                gradient_offset+2*lm_count+lm);
                                        channel_x -= radial*direction_x
                                            +angular*gradient_x;
                                        channel_y -= radial*direction_y
                                            +angular*gradient_y;
                                        channel_z -= radial*direction_z
                                            +angular*gradient_z;
                                    }
                                }, l_x, l_y, l_z);
                            local_x += l_x;
                            local_y += l_y;
                            local_z += l_z;
                        }, force_x, force_y, force_z);
                    Kokkos::single(Kokkos::PerTeam(team), [=]() {
                        response_forces(force_offset) += force_x;
                        response_forces(force_offset+1) += force_y;
                        response_forces(force_offset+2) += force_z;
                    });
                    if (recompute_harmonic_gradients)
                        team.team_barrier();
                }
                });
        }
        Kokkos::fence();
    }

    if (response_mode == MACEStreamedEdgesMode::generic
        || mace_uses_prepared_execution(response_mode)) {
        R1 = decltype(R1)();
        R1_deriv = decltype(R1_deriv)();
    }
    if (response_mode == MACEStreamedEdgesMode::generic
        || mace_uses_prepared_execution(response_mode)) {
        R0 = decltype(R0)();
        R0_deriv = decltype(R0_deriv)();
    }
    Kokkos::fence();
}

template <typename Precision>
void MACEKokkos<Precision>::reconstruct_mh0_field_response_state(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    if (!use_mh0_adjoint_reuse())
        return;

    compute_A0_streamed(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);

    const bool direct = use_factorized_direct_inference();
    if (direct) {
        release_factorized_stateful_workspace();
        if (use_factorized_direct_jit_forward())
            compute_Phi1_streamed_jit(
                num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
        else
            compute_Phi1_streamed(
                num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
        compute_A1(num_nodes, !use_factorized_async_inference());
    } else {
        compute_factorized(
            num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
    }
    compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M1(num_nodes, node_types);
    compute_H2(num_nodes, node_types);
    macefield_response_primal_reconstruction_count += 1;
}

template <typename Precision>
void MACEKokkos<Precision>::compute_electric_field_hessian(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    Kokkos::View<const double*> electric_field)
{
    compute_electric_field_response(
        num_nodes, node_types, num_neigh, neigh_indices, neigh_types,
        xyz, r, electric_field, true, false);
}

template <typename Precision>
void MACEKokkos<Precision>::compute_current_electric_field_hessian(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    Kokkos::View<const double*> electric_field,
    const std::uint64_t graph_generation)
{
    if (mace_uses_prepared_execution(streamed_edges)
        && (graph_generation == 0
            || graph_generation != factorized_prepared_graph_generation
            || factorized_completed_evaluation_epoch == 0
            || factorized_completed_evaluation_graph_generation
                != graph_generation))
        throw std::logic_error(
            "MACEField current response requires a completed prepared Execution "
            "evaluation for the current graph token.");
    compute_electric_field_response(
        num_nodes, node_types, num_neigh, neigh_indices, neigh_types,
        xyz, r, electric_field, false, false);
}

template <typename Precision>
void MACEKokkos<Precision>::compute_electric_field_force_derivative(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    Kokkos::View<const double*> electric_field)
{
    compute_electric_field_response(
        num_nodes, node_types, num_neigh, neigh_indices, neigh_types,
        xyz, r, electric_field, true, true);
}

template <typename Precision>
void MACEKokkos<Precision>::compute_current_electric_field_force_derivative(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    Kokkos::View<const double*> electric_field,
    const std::uint64_t graph_generation)
{
    if (mace_uses_prepared_execution(streamed_edges)
        && (graph_generation == 0
            || graph_generation != factorized_prepared_graph_generation
            || factorized_completed_evaluation_epoch == 0
            || factorized_completed_evaluation_graph_generation
                != graph_generation))
        throw std::logic_error(
            "MACEField current response requires a completed prepared Execution "
            "evaluation for the current graph token.");
    compute_electric_field_response(
        num_nodes, node_types, num_neigh, neigh_indices, neigh_types,
        xyz, r, electric_field, false, true);
}

template <typename Precision>
void MACEKokkos<Precision>::compute_prepared_factorized_field_response(
    const std::uint64_t graph_generation,
    Kokkos::View<const double*> electric_field,
    const bool include_force_derivative)
{
    if (!mace_uses_prepared_execution(streamed_edges)
        || graph_generation == 0
        || graph_generation != factorized_prepared_graph_generation
        || factorized_completed_evaluation_epoch == 0
        || factorized_completed_evaluation_graph_generation != graph_generation)
        throw std::logic_error(
            "MACEField current response requires a completed prepared Execution "
            "evaluation for the current graph token.");
    const std::size_t num_edges = execution_prepared_neigh_indices.extent(0);
    const bool compact_geometry =
        edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64;
    compact_edge_geometry_active = compact_geometry;
    try {
        compute_electric_field_response(
            static_cast<int>(execution_prepared_node_types.extent(0)),
            execution_prepared_node_types,
            execution_prepared_num_neigh,
            execution_prepared_neigh_indices,
            execution_prepared_neigh_types,
            execution_prepared_xyz,
            Kokkos::subview(
                execution_prepared_r,
                Kokkos::make_pair(std::size_t(0), num_edges)),
            electric_field,
            false,
            include_force_derivative);
    } catch (...) {
        compact_edge_geometry_active = false;
        throw;
    }
    compact_edge_geometry_active = false;
}
