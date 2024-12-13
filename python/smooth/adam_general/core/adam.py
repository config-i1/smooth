import time
import warnings
from adam_profile import parameters_checker

class Adam:
    def __init__(self, model="ZXZ", lags=None, orders=None, constant=False, formula=None, 
                 regressors=["use", "select", "adapt"],
                 occurrence=["none", "auto", "fixed", "general", "odds-ratio", "inverse-odds-ratio", "direct"],
                 distribution=["default", "dnorm", "dlaplace", "ds", "dgnorm", "dlnorm", "dinvgauss", "dgamma"],
                 loss=["likelihood", "MSE", "MAE", "HAM", "LASSO", "RIDGE", "MSEh", "TMSE", "GTMSE", "MSCE"],
                 outliers=["ignore", "use", "select"], level=0.99,
                 h=0, holdout=False,
                 persistence=None, phi=None, initial=["optimal", "backcasting", "complete"], arma=None,
                 ic=["AICc", "AIC", "BIC", "BICc"], bounds=["usual", "admissible", "none"],
                 silent=True, **kwargs):
        self.model = model
        self.lags = lags
        self.orders = orders
        self.constant = constant
        self.formula = formula
        self.regressors = regressors
        self.occurrence = occurrence
        self.distribution = distribution
        self.loss = loss
        self.outliers = outliers
        self.level = level
        self.h = h
        self.holdout = holdout
        self.persistence = persistence
        self.phi = phi
        self.initial = initial
        self.arma = arma
        self.ic = ic
        self.bounds = bounds
        self.silent = silent
        self.kwargs = kwargs

        self.profiles_recent_provided = False
        self.profiles_recent_table = None
        self.initial_estimated = None
        self.B = None
        self.loss_value = None
        self.other = {}

        self.elipsis = dict()


        # If a previous model is provided as a model, write down the variables
        if isinstance(self.model, (Adam, AdamSimulation)):
            self._handle_previous_model()
        
        elif isinstance(self.model, ETS):
            self._init_from_ets()
        
        elif isinstance(self.model, str):
            pass  # Everything is okay
        else:
            warnings.warn("A model of an unknown class was provided. Switching to 'ZZZ'.", UserWarning)
            self.model = "ZZZ"

    # Check the parameters of the function and create variables based on them
    checker_return = parameters_checker(
        data=data, model=self.model, lags=self.lags, formula=self.formula,
        orders=self.orders, constant=self.constant, arma=self.arma,
        outliers=self.outliers, level=self.level,
        persistence=self.persistence, phi=self.phi, initial=self.initial,
        distribution=self.distribution, loss=self.loss, h=self.h,
        holdout=self.holdout, occurrence=self.occurrence, ic=self.ic,
        bounds=self.bounds, regressors=self.regressors, y_name=y_name,
        silent=self.silent, model_do="",
        parent_environment=locals(), ellipsis=ellipsis, fast=False
    )


    #### If select was provided in the model, do auto.adam selection ####
    # here I need to implement the auto.adam selection.
    # Will do it in the future
    # lines 690 - 700 in adam.R

    


    def _init_from_ets(self):
        self.components = self.model.components # when I do the ETS
        coeffs = self.model.coef()
        self.persistence = coeffs['persistence']
        self.phi = coeffs.get('phi')
        self.initial = coeffs['initial']
        self.lags = [1]
        
        if self.components[1] != "N":
            self.lags.append(1)
        if self.components[2] != "N":
            self.lags.append(self.model.m)

        self.model = self.model.model_type()
        self.distribution = "dnorm"
        self.loss = "likelihood"

    def coef(self):
        if isinstance(self.model, ETS):
            coefficients = {
                'persistence': self.persistence,
                'initial': self.initial
            }
            if self.phi is not None:
                coefficients['phi'] = self.phi

            # Add level, trend, and seasonal components if present
            if self.components[0] != "N":
                coefficients['level'] = self.initial[0]
            if self.components[1] != "N":
                coefficients['trend'] = self.initial[1]
            if self.components[2] != "N":
                seasonal_index = 2 if self.components[1] != "N" else 1
                coefficients['seasonal'] = self.initial[seasonal_index:]
            
            return coefficients
        return {}

    def fit(self, data):
        # Start measuring the time of calculations
        start_time = time.time()

        # Get the call information
        ellipsis = self.kwargs

        # Is it useful?
        y_name = str(data)



    def _handle_previous_model(self):
        # If this is the simulated data, extract the parameters
        # TODO: Handle simulated data case if needed
        
        self.initial = self.model.initial
        self.initial_estimated = self.model.initial_estimated
        self.distribution = self.model.distribution
        self.loss = self.model.loss
        self.persistence = self.model.persistence
        self.phi = self.model.phi
        
        if self.model.initial_type != "complete":
            self.initial = self.model.initial
        else:
            self.initial = "b"
        
        self.occurrence = self.model.occurrence
        self.ic = self.model.ic
        self.bounds = self.model.bounds
        
        # lambda for LASSO
        self.ellipsis['lambda'] = self.model.other.get('lambda')
        
        # parameters for distributions
        self.ellipsis['alpha'] = self.model.other.get('alpha')
        self.ellipsis['shape'] = self.model.other.get('shape')
        self.ellipsis['nu'] = self.model.other.get('nu')
        self.B = self.model.B
        
        self.loss_value = self.model.loss_value
        self.log_lik_adam_value = self.model.loglik()
        self.lags_model_all = self.model.model_lags()
        
        # This needs to be fixed to align properly in case of various seasonals
        self.profiles_recent_table = self.model.profile_initial
        self.profiles_recent_provided = True
        
        self.regressors = self.model.regressors
        
        if self.formula is None:
            self.formula = self.model.formula()
        
        # Parameters of the original ARIMA model
        self.lags = self.model.lags()
        self.orders = self.model.orders()
        self.constant = self.model.constant
        
        if self.constant is None:
            self.constant = False
        
        self.arma = self.model.arma
        
        self.model = self.model.model_type()
        model_do = "use"
        
        # TODO: Uncomment if needed
        # if "C" in self.model:
        #     self.initial = "o"

    # Further fitting logic goes here
    # ...
