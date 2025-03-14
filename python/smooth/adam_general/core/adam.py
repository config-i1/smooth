import time
import warnings
from core.checker import parameters_checker
from core.estimator import estimator, selector
from core.creator import creator, initialiser, architector, filler
from core.utils.ic import ic_function
from core.forecaster import preparator, forecaster
import numpy as np
import pandas as pd

from smooth.adam_general._adam_general import adam_fitter, adam_forecaster

class Adam(object):
    # Note: I dont know what else parameters to include here!
    # Will be discussed
    def __init__(self, model, lags, 
                 
                profiles_recent_provided = False,
                profiles_recent_table = None,
                orders = None,
                constant = False,
                outliers = "ignore",
                level = 0.99,
                persistence = None,
                phi = None,
                initial = None,
                distribution = "default",
                loss = "likelihood",
                
                occurrence = "none",
                ic = "AICc",
                bounds = "usual",
                silent = False,
                multisteps = None,
                lb = None,
                ub = None,
                print_level = 1,
                max_eval = None):
  
        # Start measuring the time of calculations
        self.start_time = time.time()

        self.model = model
        self.lags = lags
        self.profiles_recent_provided = profiles_recent_provided
        self.profiles_recent_table = profiles_recent_table
        self.orders = orders
        self.constant = constant
        self.outliers = outliers
        self.level = level
        self.persistence = persistence
        self.phi = phi
        self.initial = initial
        self.distribution = distribution
        self.loss = loss
        self.occurrence = occurrence
        self.ic = ic
        self.bounds = bounds
        self.silent = silent
        self.multisteps = multisteps
        self.lb = lb
        self.ub = ub
        self.print_level = print_level
        self.max_eval = max_eval

        # what else should take place in the init?

    def fit(self, ts, h = None, 
            holdout = False,
            model_do = "estimate",
            fast = False,
            models_pool = None,
            lambda_param = None,
            frequency = None):
        
        self.h = h
        self.holdout = holdout
        self.model_do = model_do
        self.fast = fast
        self.models_pool = models_pool
        self.lambda_param = lambda_param
        self.frequency = frequency

        # first checking the parameters
        # This should be more pretty
        (self.general, 
        self.observations_dict,
        self.persistence_results, 
        self.initials_results, 
        self.arima_results, 
        self.constant_dict, 
        self.model_type_dict, 
        self.components_dict, 
        self.lags_dict, 
        self.occurrence_dict, 
        self.phi_dict,
        self.explanatory_dict,
        self.params_info) = parameters_checker(ts, model=self.model,
                                        lags=self.lags,
                                        orders=self.orders,
                                        constant=self.constant,
                                        outliers=self.outliers,
                                        level=self.level,
                                        persistence=self.persistence,
                                        phi=self.phi,
                                        initial=self.initial,
                                        distribution=self.distribution,
                                        loss=self.loss,
                                        h=self.h,
                                        holdout=self.holdout,
                                        occurrence=self.occurrence,
                                        ic=self.ic,
                                        bounds=self.bounds,
                                        silent=self.silent,
                                        model_do=self.model_do,
                                        fast=self.fast,
                                        models_pool=self.models_pool,
                                        lambda_param=self.lambda_param,
                                        frequency=self.frequency)
        
        # line 534 -> use regression here

        # line 677 -> do the adam selection here

        # then lines 4033 to 4070 deals with the occurence model
        # this will also wait for a bit 

        # then I also skip the regression data on lines 4036
        # I also skip the number of parameters on line 4070
        # line 4099 we continue:
        if self.model_type_dict["model_do"] == "estimate":
            # If this is LASSO/RIDGE with lambda=1, use MSE to estimate initials only
            lambda_original = self.general['lambda']
            if self.general['loss'] in ["LASSO", "RIDGE"] and self.general['lambda'] == 1:
                if self.model_type_dict["ets_model"]:
                    # Pre-set ETS parameters
                    self.persistence_results["persistence_estimate"] = False
                    self.persistence_results["persistence_level_estimate"] = False
                    self.persistence_results["persistence_trend_estimate"] = False
                    self.persistence_results["persistence_seasonal_estimate"] = [False]
                    self.persistence_results["persistence_level"] = 0
                    self.persistence_results["persistence_trend"] = 0 
                    self.persistence_results["persistence_seasonal"] = [0]
                    # Phi
                    self.phi_dict["phi_estimate"] = False
                    self.phi_dict["phi"] = 1

                if self.model_type_dict["xreg_model"]:
                    # ETSX parameters
                    self.persistence_results["persistence_xreg_estimate"] = False
                    self.persistence_results["persistence_xreg"] = 0

                if self.model_type_dict["arima_model"]:
                    # Pre-set ARMA parameters
                    self.arima_results["ar_estimate"] = [False]
                    self.arima_results["ma_estimate"] = [False]
                    arma_parameters = []
                    j = 0
                    for i, lag in enumerate(self.lags_dict["lags"]):
                        if self.arima_results["ar_orders"][i] > 0:
                            arma_parameters.extend([1] * self.arima_results["ar_orders"][i])
                            j += self.arima_results["ar_orders"][i]
                        if self.arima_results["ma_orders"][i] > 0:
                            arma_parameters.extend([0] * self.arima_results["ma_orders"][i])
                            j += self.arima_results["ma_orders"][i]
                    self.arima_results["arma_parameters"] = arma_parameters

                self.general['lambda'] = 0

            # Estimate the model
            self.adam_estimated = estimator(
                general_dict=self.general,
                model_type_dict=self.model_type_dict,
                lags_dict=self.lags_dict,
                observations_dict=self.observations_dict,
                arima_dict=self.arima_results,
                constant_dict=self.constant_dict,
                explanatory_dict=self.explanatory_dict,
                profiles_recent_table=self.profiles_recent_table,
                profiles_recent_provided=self.profiles_recent_provided,
                persistence_dict=self.persistence_results,
                initials_dict=self.initials_results,
                occurrence_dict=self.occurrence_dict,
                phi_dict=self.phi_dict,
                components_dict=self.components_dict,
            )

            # Build the architector
            (self.model_type_dict, 
             self.components_dict, 
             self.lags_dict, 
             self.observations_dict, 
             self.profile_dict) = architector(
                model_type_dict=self.model_type_dict,
                lags_dict=self.lags_dict,
                observations_dict=self.observations_dict,
                arima_checked=self.arima_results,
                constants_checked=self.constant_dict,
                explanatory_checked=self.explanatory_dict,
                profiles_recent_table=self.profiles_recent_table,
                profiles_recent_provided=self.profiles_recent_provided
            )
            
            # Build the creator
            self.adam_created = creator(
                model_type_dict=self.model_type_dict,
                lags_dict=self.lags_dict,
                profiles_dict=self.profile_dict,
                observations_dict=self.observations_dict,
                persistence_checked=self.persistence_results,
                initials_checked=self.initials_results,
                arima_checked=self.arima_results,
                constants_checked=self.constant_dict,
                phi_dict=self.phi_dict,
                components_dict=self.components_dict,
                explanatory_checked=self.explanatory_dict
            )

            # Calculate IC
            self.ic_selection = ic_function(self.general['ic'], 
                                         self.adam_estimated['log_lik_adam_value'])
            # Update parameters number
            self.n_param_estimated = self.adam_estimated['n_param_estimated']

            # Initialize parameters_number in general if not already present
            if 'parameters_number' not in self.general:
                self.general['parameters_number'] = self.params_info['parameters_number']

            self.general['parameters_number'][0][0] = self.n_param_estimated

            # Handle likelihood loss case
            if self.general['loss'] == 'likelihood':
                if len(self.general['parameters_number'][0]) <= 3:
                    self.general['parameters_number'][0].append(1)
                else:
                    self.general['parameters_number'][0][3] = 1

            # Calculate row sums
            if len(self.general['parameters_number'][0]) <= 4:
                self.general['parameters_number'][0].append(sum(self.general['parameters_number'][0][0:4]))
                self.general['parameters_number'][1].append(sum(self.general['parameters_number'][1][0:4]))
            else:
                self.general['parameters_number'][0][4] = sum(self.general['parameters_number'][0][0:4])
                self.general['parameters_number'][1][4] = sum(self.general['parameters_number'][1][0:4])

        elif self.model_type_dict["model_do"] == "select":
            # Run model selection
            self.adam_selected = selector(
                model_type_dict=self.model_type_dict,
                phi_dict=self.phi_dict,
                general_dict=self.general, 
                lags_dict=self.lags_dict, 
                observations_dict=self.observations_dict, 
                arima_dict=self.arima_results,
                constant_dict=self.constant_dict,
                explanatory_dict=self.explanatory_dict,
                occurrence_dict=self.occurrence_dict,
                components_dict=self.components_dict,
                profiles_recent_table=self.profiles_recent_table,
                profiles_recent_provided=self.profiles_recent_provided,
                persistence_results=self.persistence_results,
                initials_results=self.initials_results,
                criterion=self.ic,
                silent=self.silent
            )


            # Get selection results
            self.ic_selection = self.adam_selected['ic_selection']
            
            self.results = self.adam_selected['results']

            # Find best model
            self.best_model = min(self.ic_selection.items(), key=lambda x: x[1])[0]
            self.best_id = next(i for i, result in enumerate(self.results) 
                              if result['model'] == self.best_model)

            # Update dictionaries with best model results
            self.model_type_dict = self.results[self.best_id]['model_type_dict']
            self.phi_dict = self.results[self.best_id]['phi_dict']
            self.adam_estimated = self.results[self.best_id]['adam_estimated']

            # Build the architector
            (self.model_type_dict, 
             self.components_dict, 
             self.lags_dict, 
             self.observations_dict, 
             self.profile_dict) = architector(
                model_type_dict=self.model_type_dict,
                lags_dict=self.lags_dict,
                observations_dict=self.observations_dict,
                arima_checked=self.arima_results,
                constants_checked=self.constant_dict,
                explanatory_checked=self.explanatory_dict,
                profiles_recent_table=self.profiles_recent_table,
                profiles_recent_provided=self.profiles_recent_provided
            )

            # Build the creator
            self.adam_created = creator(
                model_type_dict=self.model_type_dict,
                lags_dict=self.lags_dict,
                profiles_dict=self.profile_dict,
                observations_dict=self.observations_dict,
                persistence_checked=self.persistence_results,
                initials_checked=self.initials_results,
                arima_checked=self.arima_results,
                constants_checked=self.constant_dict,
                phi_dict=self.phi_dict,
                components_dict=self.components_dict,
                explanatory_checked=self.explanatory_dict
            )

            # Update parameters number
            self.n_param_estimated = self.adam_estimated['n_param_estimated']
            self.general['parameters_number'] = self.params_info['parameters_number']
            self.general['parameters_number'][0][0] = self.n_param_estimated

            # Handle likelihood loss case
            if self.general['loss'] == 'likelihood':
                if len(self.general['parameters_number'][0]) <= 3:
                    self.general['parameters_number'][0].append(1)
                else:
                    self.general['parameters_number'][0][3] = 1

            # Calculate row sums
            if len(self.general['parameters_number'][0]) <= 4:
                self.general['parameters_number'][0].append(sum(self.general['parameters_number'][0][0:4]))
                self.general['parameters_number'][1].append(sum(self.general['parameters_number'][1][0:4]))
            else:
                self.general['parameters_number'][0][4] = sum(self.general['parameters_number'][0][0:4])
                self.general['parameters_number'][1][4] = sum(self.general['parameters_number'][1][0:4])

        elif self.model_type_dict["model_do"] == "combine":
            # Store original model configuration
            model_original = self.model_type_dict.copy()
            
            # If models pool is not provided, create one
            if self.general['models_pool'] is None:
                # Define the whole pool of errors
                if not self.model_type_dict['allow_multiplicative']:
                    pool_errors = ['A']
                    pool_trends = ['N', 'A', 'Ad']
                    pool_seasonals = ['N', 'A']
                else:
                    pool_errors = ['A', 'M']
                    pool_trends = ['N', 'A', 'Ad', 'M', 'Md']
                    pool_seasonals = ['N', 'A', 'M']

                # If error_type is not Z, check on additive errors
                if self.model_type_dict['error_type'] != 'Z':
                    if self.model_type_dict['error_type'] == 'N':
                        pool_errors = ['N']
                    elif self.model_type_dict['error_type'] in ['A', 'X']:
                        pool_errors = ['A']
                    elif self.model_type_dict['error_type'] in ['M', 'Y']:
                        pool_errors = ['M']

                # If trend_type is not Z, create pool with specified type
                if self.model_type_dict['trend_type'] != 'Z':
                    if self.model_type_dict['trend_type'] == 'N':
                        pool_trends = ['N']
                    elif self.model_type_dict['trend_type'] == 'A':
                        pool_trends = ['Ad' if self.model_type_dict['damped'] else 'A']
                    elif self.model_type_dict['trend_type'] == 'M':
                        pool_trends = ['Md' if self.model_type_dict['damped'] else 'M']
                    elif self.model_type_dict['trend_type'] == 'X':
                        pool_trends = ['N', 'A', 'Ad']
                    elif self.model_type_dict['trend_type'] == 'Y':
                        pool_trends = ['N', 'M', 'Md']

                # If season_type is not Z, create specific pools
                if self.model_type_dict['season_type'] != 'Z':
                    if self.model_type_dict['season_type'] == 'N':
                        pool_seasonals = ['N']
                    elif self.model_type_dict['season_type'] == 'A':
                        pool_seasonals = ['A']
                    elif self.model_type_dict['season_type'] == 'X':
                        pool_seasonals = ['N', 'A']
                    elif self.model_type_dict['season_type'] == 'M':
                        pool_seasonals = ['M']
                    elif self.model_type_dict['season_type'] == 'Y':
                        pool_seasonals = ['N', 'M']

                # Create models pool by combining all possibilities
                self.general['models_pool'] = [e + t + s for e in pool_errors 
                                             for t in pool_trends 
                                             for s in pool_seasonals]

            # Run model selection
            self.adam_selected = selector(
                model_type_dict=self.model_type_dict,
                phi_dict=self.phi_dict,
                general=self.general,
                lags_dict=self.lags_dict,
                observations_dict=self.observations_dict,
                arima_results=self.arima_results,
                constant_dict=self.constant_dict,
                explanatory_dict=self.explanatory_dict,
                occurrence_dict=self.occurrence_dict,
                components_dict=self.components_dict,
                profiles_recent_table=self.profiles_recent_table,
                profiles_recent_provided=self.profiles_recent_provided,
                persistence_results=self.persistence_results,
                initials_results=self.initials_results,
                criterion="AICc",
                silent=False
            )

            # Calculate weights based on IC
            ic_best = min(self.adam_selected['ic_selection'].values())
            ic_weights = {model: np.exp(-0.5 * (ic - ic_best)) 
                         for model, ic in self.adam_selected['ic_selection'].items()}
            weights_sum = sum(ic_weights.values())
            ic_weights = {model: weight/weights_sum for model, weight in ic_weights.items()}

            # Set very small weights to 0 as a failsafe
            ic_weights = {model: 0 if weight < 1e-5 else weight 
                         for model, weight in ic_weights.items()}
            weights_sum = sum(ic_weights.values())
            ic_weights = {model: weight/weights_sum for model, weight in ic_weights.items()}
            
            # Store weights in selected results
            self.adam_selected['ic_weights'] = ic_weights

            # Process each selected model
            for i in range(len(self.adam_selected['results'])):
                result = self.adam_selected['results'][i]
                
                # Build architector for this model
                (self.model_type_dict, 
                 self.components_dict, 
                 self.lags_dict, 
                 self.observations_dict, 
                 self.profile_dict) = architector(
                    model_type_dict=result['model_type_dict'],
                    lags_dict=self.lags_dict,
                    observations_dict=self.observations_dict,
                    arima_checked=self.arima_results,
                    constants_checked=self.constant_dict,
                    explanatory_checked=self.explanatory_dict,
                    profiles_recent_table=self.profiles_recent_table,
                    profiles_recent_provided=self.profiles_recent_provided
                )

                # Store updated dictionaries in results
                self.adam_selected['results'][i].update({
                    'model_type_dict': self.model_type_dict,
                    'components_dict': self.components_dict,
                    'lags_dict': self.lags_dict,
                    'observations_dict': self.observations_dict,
                    'profile_dict': self.profile_dict
                })

                # Create matrices for this model
                self.adam_created = creator(
                    model_type_dict=self.model_type_dict,
                    lags_dict=self.lags_dict,
                    profiles_dict=self.profile_dict,
                    observations_dict=self.observations_dict,
                    persistence_checked=self.persistence_results,
                    initials_checked=self.initials_results,
                    arima_checked=self.arima_results,
                    constants_checked=self.constant_dict,
                    phi_dict=self.phi_dict,
                    components_dict=self.components_dict,
                    explanatory_checked=self.explanatory_dict
                )

                # Store created matrices
                self.adam_selected['results'][i]['adam_created'] = self.adam_created

                # Update parameters number
                n_param_estimated = result['adam_estimated']['n_param_estimated']
                self.general['parameters_number'] = self.params_info['parameters_number']
                self.general['parameters_number'][0][0] = n_param_estimated

                # Handle likelihood loss case
                if self.general['loss'] == 'likelihood':
                    if len(self.general['parameters_number'][0]) <= 3:
                        self.general['parameters_number'][0].append(1)
                    else:
                        self.general['parameters_number'][0][3] = 1

                # Calculate row sums
                if len(self.general['parameters_number'][0]) <= 4:
                    self.general['parameters_number'][0].append(sum(self.general['parameters_number'][0][0:4]))
                    self.general['parameters_number'][1].append(sum(self.general['parameters_number'][1][0:4]))
                else:
                    self.general['parameters_number'][0][4] = sum(self.general['parameters_number'][0][0:4])
                    self.general['parameters_number'][1][4] = sum(self.general['parameters_number'][1][0:4])

                # Store parameters number
                self.adam_selected['results'][i]['parameters_number'] = self.general['parameters_number']

        else:
            warnings.warn(f"Unknown model_do value: {self.model_type_dict['model_do']}. Expected one of: 'estimate', 'select', 'combine'")

        # Transform data into appropriate classes
        if isinstance(self.observations_dict['y_in_sample'], np.ndarray):
            self.y_in_sample = pd.Series(
                self.observations_dict['y_in_sample'], 
                index=pd.date_range(
                    start=self.observations_dict['y_start'], 
                    periods=len(self.observations_dict['y_in_sample']), 
                    freq=self.observations_dict['frequency']
                )
            )
            if self.general['holdout']:
                self.y_holdout = pd.Series(
                    self.observations_dict['y_holdout'],
                    index=pd.date_range(
                        start=self.observations_dict['y_forecast_start'],
                        periods=len(self.observations_dict['y_holdout']),
                        freq=self.observations_dict['frequency']
                    )
                )
        else:
            self.y_in_sample = self.observations_dict['y_in_sample'].copy()
            if self.general['holdout']:
                self.y_holdout = pd.Series(
                    self.observations_dict['y_holdout'],
                    index=self.observations_dict['y_forecast_index']
                )

        # Handle distribution selection
        if self.general['distribution'] == "default":
            if self.general['loss'] == "likelihood":
                if self.model_type_dict['error_type'] == "A":
                    self.general['distribution_new'] = "dnorm"
                elif self.model_type_dict['error_type'] == "M":
                    self.general['distribution_new'] = "dgamma"
            elif self.general['loss'] in ["MAEh", "MACE", "MAE"]:
                self.general['distribution_new'] = "dlaplace"
            elif self.general['loss'] in ["HAMh", "CHAM", "HAM"]:
                self.general['distribution_new'] = "ds"
            elif self.general['loss'] in ["MSEh", "MSCE", "MSE", "GPL"]:
                self.general['distribution_new'] = "dnorm"
        else:
            self.general['distribution_new'] = self.general['distribution']


        return self
    

    def predict(self):
        """Make predictions using the fitted model"""
        
        self.prepared_model = preparator(
            # Model info
            model_type_dict=self.model_type_dict,
            
            # Components info
            components_dict=self.components_dict,
            
            # Lags info 
            lags_dict=self.lags_dict,
            
            # Matrices from creator
            matrices_dict=self.adam_created,
            
            # Parameter dictionaries
            persistence_checked=self.persistence_results,
            initials_checked=self.initials_results,
            arima_checked=self.arima_results,
            explanatory_checked=self.explanatory_dict,
            phi_dict=self.phi_dict,
            constants_checked=self.constant_dict,
            
            # Other parameters
            observations_dict=self.observations_dict,
            occurrence_dict=self.occurrence_dict,
            general_dict=self.general,
            profiles_dict=self.profile_dict,
            
            # The parameter vector
            adam_estimated=self.adam_estimated,
            
            # Optional parameters
            bounds="usual",
            other=None
        )


        self.predictions = forecaster(
            model_prepared=self.prepared_model,
            observations_dict=self.observations_dict,
            general_dict=self.general,
            occurrence_dict=self.occurrence_dict,
            lags_dict=self.lags_dict,
            model_type_dict=self.model_type_dict,
            explanatory_checked=self.explanatory_dict,
            components_dict=self.components_dict,
            constants_checked=self.constant_dict
        )

        return self.predictions
        


        
        #return out