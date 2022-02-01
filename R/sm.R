# @importFrom greybox sm
# @export
sm.adam <- function(object, model="YYY", lags=c(frequency(data)),
                    orders=list(ar=c(0),i=c(0),ma=c(0),select=FALSE),
                    constant=FALSE, formula=NULL, data=NULL,
                    regressors=c("use","select","adapt"),
                    h=0, holdout=FALSE,
                    persistence=NULL, phi=NULL, initial=c("optimal","backcasting"), arma=NULL,
                    ic=c("AICc","AIC","BIC","BICc"), bounds=c("usual","admissible","none"),
                    silent=TRUE, ...){
    # The function creates a scale model for the provided model
    # occurrence and distribution are extracted from the model.
    # loss can only be likelihood (for now)
    # outliers are not detected
    # Start measuring the time of calculations
    startTime <- Sys.time();

    distribution <- object$distribution;
    loss <- "likelihood";
    occurrence <- object$occurrence;
    outliers <- "ignore";

    cl <- object$call;
    if(is.null(data)){
        data <- object$call$data;
    }
    if(is.null(formula)){
        formula <- formula(object);
        formula[[2]] <- NULL;
    }

    # Extract the other value
    if(!is.null(other)){
        other <- switch(distribution,
                        "dgnorm"=,
                        "dlgnorm"=other$shape,
                        "dalaplace"=other$alpha,
                        "dt"=other$df,
                        NULL);
    }

    #### Ellipsis values ####
    ellipsis <- list(...);
    # Fisher Information
    if(is.null(ellipsis$FI)){
        FI <- FALSE;
    }
    else{
        FI <- ellipsis$FI;
    }
    # Starting values for the optimiser
    if(is.null(ellipsis$B)){
        B <- NULL;
    }
    else{
        B <- ellipsis$B;
    }
    # Parameters for the nloptr from the ellipsis
    if(is.null(ellipsis$xtol_rel)){
        xtol_rel <- 1E-6;
    }
    else{
        xtol_rel <- ellipsis$xtol_rel;
    }
    if(is.null(ellipsis$algorithm)){
        algorithm <- "NLOPT_LN_SBPLX";
    }
    else{
        algorithm <- ellipsis$algorithm;
    }
    if(is.null(ellipsis$maxtime)){
        maxtime <- -1;
    }
    else{
        maxtime <- ellipsis$maxtime;
    }
    if(is.null(ellipsis$xtol_abs)){
        xtol_abs <- 1E-8;
    }
    else{
        xtol_abs <- ellipsis$xtol_abs;
    }
    if(is.null(ellipsis$ftol_rel)){
        ftol_rel <- 1E-6;
    }
    else{
        ftol_rel <- ellipsis$ftol_rel;
    }
    if(is.null(ellipsis$ftol_abs)){
        ftol_abs <- 0;
    }
    else{
        ftol_abs <- ellipsis$ftol_abs;
    }
    if(is.null(ellipsis$print_level)){
        print_level <- 0;
    }
    else{
        print_level <- ellipsis$print_level;
    }
    print_level_hidden <- print_level;
    if(print_level==41){
        print_level[] <- 0;
    }
    if(is.null(ellipsis$stepSize)){
        stepSize <- .Machine$double.eps^(1/4);
    }
    else{
        stepSize <- ellipsis$stepSize;
    }

    if(is.null(cl)){
        responseName <- "y";
    }
    else{
        responseName <- formula(ellipsis$cl)[[2]];
    }

    et <- residuals(object);

    ### Call adam() with a set of values, providing custom loss, e.g.:
    # lossFunction <- -dlnorm(...)
    ## Then form fitted, recalculate logLik, get residuals



    #### The function estimates parameters of scale model ####
    CFScale <- function(B){
        scale <- fitterScale(B, distribution);
        CFValue <- -sum(switch(distribution,
                               "dnorm" = dnorm(y[otU], mean=mu[otU], sd=scale, log=TRUE),
                               "dlaplace" = dlaplace(y[otU], mu=mu[otU], scale=scale, log=TRUE),
                               "ds" = ds(y[otU], mu=mu[otU], scale=scale, log=TRUE),
                               "dgnorm" = dgnorm(y[otU], mu=mu[otU], scale=scale,
                                                 shape=other, log=TRUE),
                               "dlogis" = dlogis(y[otU], location=mu[otU], scale=scale, log=TRUE),
                               "dt" = dt(y[otU]-mu[otU], df=scale, log=TRUE),
                               "dalaplace" = dalaplace(y[otU], mu=mu[otU], scale=scale,
                                                       alpha=other, log=TRUE),
                               "dlnorm" = dlnorm(y[otU], meanlog=mu[otU], sdlog=scale, log=TRUE),
                               "dllaplace" = dlaplace(log(y[otU]), mu=mu[otU],
                                                      scale=scale, log=TRUE)-log(y[otU]),
                               "dls" = ds(log(y[otU]), mu=mu[otU], scale=scale, log=TRUE)-log(y[otU]),
                               "dlgnorm" = dgnorm(log(y[otU]), mu=mu[otU], scale=scale,
                                                  shape=other, log=TRUE)-log(y[otU]),
                               "dinvgauss" = dinvgauss(y[otU], mean=mu[otU],
                                                       dispersion=scale/mu[otU], log=TRUE),
                               "dgamma" = dgamma(y[otU], shape=1/scale,
                                                 scale=scale*mu[otU], log=TRUE)
        ));

        # The differential entropy for the models with the missing data
        if(occurrenceModel){
            CFValue[] <- CFValue + sum(switch(distribution,
                                              "dnorm" =,
                                              "dfnorm" =,
                                              "dbcnorm" =,
                                              "dlogitnorm" =,
                                              "dlnorm" = obsZero*(log(sqrt(2*pi)*scale[!otU])+0.5),
                                              "dgnorm" =,
                                              "dlgnorm" =obsZero*(1/other-
                                                                      log(other /
                                                                              (2*scale[!otU]*gamma(1/other)))),
                                              # "dinvgauss" = 0.5*(obsZero*(log(pi/2)+1+suppressWarnings(log(scale[!otU])))-
                                              #                                 sum(log(mu[!otU]))),
                                              "dinvgauss" = obsZero*(0.5*(log(pi/2)+1+suppressWarnings(log(scale[!otU])))),
                                              "dgamma" = obsZero*(1/scale[!otU] + log(scale[!otU]) +
                                                                      log(gamma(1/scale[!otU])) +
                                                                      (1-1/scale[!otU])*digamma(1/scale[!otU])),
                                              "dlaplace" =,
                                              "dllaplace" =,
                                              "ds" =,
                                              "dls" = obsZero*(2 + 2*log(2*scale[!otU])),
                                              "dalaplace" = obsZero*(1 + log(2*scale[!otU])),
                                              "dlogis" = obsZero*2,
                                              "dt" = obsZero*((scale[!otU]+1)/2 *
                                                                  (digamma((scale[!otU]+1)/2)-digamma(scale[!otU]/2)) +
                                                                  log(sqrt(scale[!otU]) * beta(scale[!otU]/2,0.5))),
                                              "dchisq" = obsZero*(log(2)*gamma(scale[!otU]/2)-
                                                                      (1-scale[!otU]/2)*digamma(scale[!otU]/2)+
                                                                      scale[!otU]/2),
                                              # "dbeta" = sum(log(beta(mu[otU],scale[!otU][otU]))-
                                              #                   (mu[otU]-1)*
                                              #                   (digamma(mu[otU])-
                                              #                        digamma(mu[otU]+scale[!otU][otU]))-
                                              #                   (scale[!otU][otU]-1)*
                                              #                   (digamma(scale[!otU][otU])-
                                              #                        digamma(mu[otU]+scale[!otU][otU]))),
                                              # This is a normal approximation of the real entropy
                                              # "dpois" = sum(0.5*log(2*pi*scale[!otU])+0.5),
                                              # "dnbinom" = obsZero*(log(sqrt(2*pi)*scale[!otU])+0.5),
                                              0
            ));
        }
        return(CFValue);
    }

    #### !!!! This needs to be double checked
    errors <- switch(distribution,
                     "dnorm"=,
                     "dlnorm"=,
                     "dbcnorm"=,
                     "dlogitnorm"=,
                     "dfnorm"=,
                     "dlogis"=,
                     "dlaplace"=,
                     "dllaplace"=,
                     "dalaplace"=abs(residuals[subset]),
                     "ds"=,
                     "dls"=residuals[subset]^2,
                     "dgnorm"=,
                     "dlgnorm"=abs(residuals[subset])^{1/other},
                     "dgamma"=abs(residuals[subset]),
                     "dinvgauss"=abs(residuals[subset]));

    errors[] <- errors / scale;

    # If formula does not have response variable, update it.
    # This is mainly needed for the proper plots and outputs
    if(length(formula)==2){
        cl$formula <- update.formula(formula,paste0(responseName,"~."))
    }

    # Form the scale object
    finalModel <- structure(list(formula=formula, coefficients=B, fitted=scale, residuals=errors,
                                 df.residual=obsInsample-nVariables, df=nVariables, call=cl, rank=nVariables,
                                 data=matrixXregScale, terms=dataTerms, logLik=-CFValue,
                                 occurrence=occurrence, subset=subset, other=ellipsis, B=B, FI=FI,
                                 distribution=distribution, other=other, loss="likelihood",
                                 timeElapsed=Sys.time()-startTime),
                            class=c("scale","alm","greybox"));
    return(finalModel);
}
