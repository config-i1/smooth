# estimator commented out lines 2754 to 2821
adam_created_arima = filler(
            b_values['B'],
            ets_model, e_type, t_type, s_type, model_is_trendy, model_is_seasonal,
            components_number_ets, components_number_ets_non_seasonal,
            components_number_ets_seasonal, components_number_arima,
            lags, lags_model, lags_model_max,
            adam_created['mat_vt'], adam_created['mat_wt'], adam_created['mat_f'], adam_created['vec_g'],
            persistence_estimate, persistence_level_estimate, persistence_trend_estimate,
            persistence_seasonal_estimate, persistence_xreg_estimate,
            phi_estimate,
            initial_type, initial_estimate,
            initial_level_estimate, initial_trend_estimate, initial_seasonal_estimate,
            initial_arima_estimate, initial_xreg_estimate,
            arima_model, ar_estimate, ma_estimate, ar_orders, i_orders, ma_orders,
            ar_required, ma_required, arma_parameters,
            non_zero_ari, non_zero_ma, adam_created['arima_polynomials'],
            xreg_model, xreg_number,
            xreg_parameters_missing, xreg_parameters_included,
            xreg_parameters_estimated, xreg_parameters_persistence, constant_estimate
        )

        # Write down the initials in the recent profile
        profiles_recent_table[:] = adam_created_arima['mat_vt'][:, :lags_model_max]

        # Do initial fit to get the state values from the backcasting
        adam_fitted = adam_fitter_wrap(
            adam_created_arima['mat_vt'], adam_created_arima['mat_wt'], adam_created_arima['mat_f'], adam_created_arima['vec_g'],
            lags_model_all, index_lookup_table, profiles_recent_table,
            e_type, t_type, s_type, components_number_ets, components_number_ets_seasonal,
            components_number_arima, xreg_number, constant_required,
            y_in_sample, ot, True
        )

        adam_created['mat_vt'][:, :lags_model_max] = adam_fitted['mat_vt'][:, :lags_model_max]
        # Produce new initials
        b_values_new = initialiser(
            ets_model, e_type, t_type, s_type, model_is_trendy, model_is_seasonal,
            components_number_ets_non_seasonal, components_number_ets_seasonal, components_number_ets,
            lags, lags_model, lags_model_seasonal, lags_model_arima, lags_model_max,
            adam_created['mat_vt'],
            persistence_estimate, persistence_level_estimate, persistence_trend_estimate,
            persistence_seasonal_estimate, persistence_xreg_estimate,
            phi_estimate, initial_type, initial_estimate,
            initial_level_estimate, initial_trend_estimate, initial_seasonal_estimate,
            initial_arima_estimate, initial_xreg_estimate,
            arima_model, ar_required, ma_required, ar_estimate, ma_estimate, ar_orders, ma_orders,
            components_number_arima, components_names_arima, initial_arima_number,
            xreg_model, xreg_number,
            xreg_parameters_estimated, xreg_parameters_persistence,
            constant_estimate, constant_name, other_parameter_estimate
        )
        B = b_values_new['B']
        # Failsafe, just in case if the initial values contain NA / NaN
        B[np.isnan(B)] = b_values['B'][np.isnan(B)]



        # Fix for mixed ETS models producing negative values
        if (e_type == "M" and any(t in ["A", "Ad"] for t in [t_type, s_type]) or
            t_type == "M" and any(t in ["A", "Ad"] for t in [e_type, s_type]) or
            s_type == "M" and any(t in ["A", "Ad"] for t in [e_type, t_type])):
            if e_type == "M" and ("level" in B) and (B["level"] <= 0):
                B["level"] = y_in_sample[0]
            if t_type == "M" and ("trend" in B) and (B["trend"] <= 0):
                B["trend"] = 1
            seasonal_params = [p for p in B.keys() if p.startswith("seasonal")]
            if s_type == "M" and any(B[p] <= 0 for p in seasonal_params):
                for p in seasonal_params:
                    if B[p] <= 0:
                        B[p] = 1