ssxreg <- function(data, xreg=NULL, go.wild=FALSE,
                   persistenceX=NULL, transitionX=NULL, initialX=NULL,
                   obs, obs.all, obs.xt, maxlag=1, h=1, silent=FALSE){
# The function does general checks needed for exogenouse variables and returns the list of necessary parameters

    if(!is.null(xreg)){
        if(any(is.na(xreg)) & silent==FALSE){
            message("The exogenous variables contain NAs! This may lead to problems during estimation and forecast.");
        }
##### The case with vectors and ts objects, but not matrices
        if(is.vector(xreg) | (is.ts(xreg) & !is.matrix(xreg))){
# Check if xreg contains something meaningful
            if(all(xreg[1:obs]==xreg[1])){
                warning("The exogenous variable has no variability. Cannot do anything with that, so dropping out xreg.",
                        call.=FALSE, immediate.=TRUE);
                xreg <- NULL;
            }
            else{
                if(length(xreg)!=obs & length(xreg)!=obs.all){
                    stop("The length of xreg does not correspond to either in-sample or the whole series lengths. Aborting!", call.=F);
                }
                if(length(xreg)==obs){
                    if(silent==FALSE){
                        message("No exogenous are provided for the holdout sample. Using Naive as a forecast.");
                    }
                    xreg <- c(as.vector(xreg),rep(xreg[obs],h));
                }
# Number of exogenous variables
                n.exovars <- 1;
# Define matrix w for exogenous variables
                matxt <- matrix(xreg,ncol=1);
# Define the second matat to fill in the coefs of the exogenous vars
                matat <- matrix(NA,obs.xt,1);
                exocomponent.names <- "exogenous";
# Fill in the initial values for exogenous coefs using OLS
                matat[1:maxlag,] <- cov(data[1:obs],xreg[1:obs])/var(xreg[1:obs]);
            }
        }
##### The case with matrices and data frames
        else if(is.matrix(xreg) | is.data.frame(xreg)){
            checkvariability <- apply(xreg[1:obs,]==rep(xreg[1,],each=obs),2,all);
            if(any(checkvariability)){
                if(all(checkvariability)){
                    warning("All exogenous variables have no variability. Cannot do anything with that, so dropping out xreg.",
                            call.=FALSE, immediate.=TRUE);
                    xreg <- NULL;
                }
                else{
                    warning("Some exogenous variables do not have any variability. Dropping them out.",
                            call.=FALSE, immediate.=TRUE);
                    xreg <- as.matrix(xreg[,!checkvariability]);
                }
            }

            if(!is.null(xreg)){
                if(nrow(xreg)!=obs & nrow(xreg)!=obs.all){
                    stop("The length of xreg does not correspond to either in-sample or the whole series lengths. Aborting!",call.=F);
                }
                if(nrow(xreg)==obs){
                    if(silent==FALSE){
	                    message("No exogenous are provided for the holdout sample. Using Naive as a forecast.");
                    }
                    for(j in 1:h){
                    xreg <- rbind(xreg,xreg[obs,]);
                    }
                }
# mat.x is needed for the initial values of coefs estimation using OLS
                mat.x <- as.matrix(cbind(rep(1,obs.all),xreg));
                n.exovars <- ncol(xreg);
# Define the second matat to fill in the coefs of the exogenous vars
                matat <- matrix(NA,obs.xt,n.exovars);
                exocomponent.names <- paste0("x",c(1:n.exovars));
# Define matrix w for exogenous variables
                matxt <- as.matrix(xreg);
# Fill in the initial values for exogenous coefs using OLS
                matat[1:maxlag,] <- rep(t(solve(t(mat.x[1:obs,]) %*% mat.x[1:obs,],tol=1e-50) %*%
                                                  t(mat.x[1:obs,]) %*% data[1:obs])[2:(n.exovars+1)],
                                          each=maxlag);
                colnames(matat) <- colnames(xreg);
            }
        }
        else{
            stop("Unknown format of xreg. Should be either vector or matrix. Aborting!",call.=F);
        }
        estimate.xreg <- TRUE;
        colnames(matat) <- exocomponent.names;

# Check the provided initialX vector
        if(!is.null(initialX)){
            if(!is.numeric(initialX) & !is.vector(initialX) & !is.matrix(initialX)){
                stop("The initials for exogenous are not a numeric vector or a matrix!", call.=FALSE);
            }
            else{
                if(length(initialX) != n.exovars){
                    stop("The size of initial vector for exogenous is wrong! It should correspond to the number of exogenous variables.", call.=FALSE);
                }
                else{
                    matat[1:maxlag,] <- as.vector(rep(initialX,each=maxlag));
                    estimate.initialX <- FALSE;
                }
            }
        }
        else{
            estimate.initialX <- TRUE;
        }
    }

##### In case we changed xreg to null or if it was like that...
    if(is.null(xreg)){
# "1" is needed for the final forecast simplification
        n.exovars <- 1;
        matxt <- matrix(1,obs.xt,1);
        matat <- matrix(0,obs.xt,1);
        matFX <- matrix(1,1,1);
        vecgX <- matrix(0,1,1);
        estimate.xreg <- FALSE;
        estimate.FX <- FALSE;
        estimate.gX <- FALSE;
        estimate.initialX <- FALSE;
    }

# Now check transition and persistence of exogenous variables
    if(estimate.xreg==TRUE & go.wild==TRUE){
# First - transition matrix
        if(!is.null(transitionX)){
            if(!is.numeric(transitionX) & !is.vector(transitionX) & !is.matrix(transitionX)){
                stop("The transition matrix for exogenous is not a numeric vector or matrix!", call.=FALSE);
            }
            else{
                if(length(transitionX) != n.exovars^2){
                    stop("The size of transition matrix for exogenous is wrong! It should correspond to the number of exogenous variables.", call.=FALSE);
                }
                else{
                    matFX <- matrix(transitionX,n.exovars,n.exovars);
                    estimate.FX <- FALSE;
                }
            }
        }
        else{
            matFX <- diag(n.exovars);
            estimate.FX <- TRUE;
        }
# Now - persistence vector
        if(!is.null(persistenceX)){
            if(!is.numeric(persistenceX) & !is.vector(persistenceX) & !is.matrix(persistenceX)){
                stop("The transition matrix for exogenous is not a numeric vector or matrix!", call.=FALSE);
            }
            else{
                if(length(persistenceX) != n.exovars){
                    stop("The size of persistence vector for exogenous is wrong! It should correspond to the number of exogenous variables.", call.=FALSE);
                }
                else{
                    vecgX <- matrix(persistenceX,n.exovars,1);
                    estimate.gX <- FALSE;
                }
            }
        }
        else{
            vecgX <- matrix(0,n.exovars,1);
            estimate.gX <- TRUE;
        }
    }
    else if(estimate.xreg==TRUE & go.wild==FALSE){
        matFX <- diag(n.exovars);
        estimate.FX <- FALSE;

        vecgX <- matrix(0,n.exovars,1);
        estimate.gX <- FALSE;
    }

    return(list(n.exovars=n.exovars, matxt=matxt, matat=matat, matFX=matFX, vecgX=vecgX,
                estimate.xreg=estimate.xreg, estimate.FX=estimate.FX,
                estimate.gX=estimate.gX, estimate.initialX=estimate.initialX))
}
