if st.sidebar.button("🚀 Auto-Fit (All 4 Parameters)", use_container_width=True):
    if len(R_fit) > 5:
        with st.spinner("Fitting all parameters..."):
            M_baryon_fit = baryonic_mass_from_components(R_fit, Vgas_fit, Vdisk_fit, Vbul_fit)
            initial_guess = [
                10**st.session_state.log_rho0,
                st.session_state.r_s,
                st.session_state.epsilon,
                st.session_state.Omega
            ]
            try:
                best_rho0, best_r_s, best_eps, best_Omega = fit_qcaus_all(
                    R_fit, Vobs_fit, errV_fit, M_baryon_fit, initial_guess
                )
                st.session_state.log_rho0 = np.log10(best_rho0)
                st.session_state.r_s = best_r_s
                st.session_state.epsilon = best_eps
                st.session_state.Omega = best_Omega
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Fit failed: {e}")
