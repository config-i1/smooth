#' @export
nobs.vsmooth <- function(object, ...){
    return(nrow(object$fitted));
}

#' @export
sigma.vsmooth <- function(object, ...){
    return(object$Sigma);
}

#### Extraction of parameters of models ####
#' @export
coef.vsmooth <- function(object, ...){

    parameters <- object$coefficients;

    return(parameters);
}

#' @export
modelType.vsmooth <- function(object, ...){
    model <- object$model;
    modelType <- NA;
    if(!is.null(model)){
        if(gregexpr("VES",model)!=-1){
            modelType <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
        }
    }

    return(modelType);
}

#### Prints of vsmooth ####
#' @export
print.vsmooth <- function(x, ...){
    holdout <- any(!is.na(x$holdout));
    intervals <- any(!is.na(x$lower));

    if(all(holdout,intervals)){
        insideintervals <- sum((x$holdout <= x$upper) & (x$holdout >= x$lower)) / length(x$forecast) * 100;
    }
    else{
        insideintervals <- NULL;
    }

    intervalsType <- x$intervals;

    cat(paste0("Time elapsed: ",round(as.numeric(x$timeElapsed,units="secs"),2)," seconds\n"));
    cat(paste0("Model estimated: ",x$model,"\n"));
    if(!is.null(x$nParam)){
        if(x$nParam==1){
            cat(paste0(x$nParam," parameter was estimated in the process\n"));
        }
        else{
            cat(paste0(x$nParam," parameters were estimated in the process\n"));
        }
    }

    cat(paste0("Cost function type: ",x$cfType))
    if(!is.null(x$cf)){
        cat(paste0("; Cost function value: ",round(x$cf,3),"\n"));
    }
    else{
        cat("\n");
    }

    cat("\nInformation criteria:\n");
    print(x$ICs);

    if(intervals){
        if(x$intervals=="p"){
            intervalsType <- "parametric";
        }
        else if(x$intervals=="sp"){
            intervalsType <- "semiparametric";
        }
        else if(x$intervals=="np"){
            intervalsType <- "nonparametric";
        }
        else if(x$intervals=="a"){
            intervalsType <- "asymmetric";
        }
        cat(paste0(x$level*100,"% ",intervalsType," prediction intervals were constructed\n"));
    }

}

#### Summary of objects ####
#' @export
summary.vsmooth <- function(object, ...){
    print(object);
}
