covarAnal <- function(lagsModel, h, measurement, transition, persistence, s2){
    # Function returns analytical conditional h-steps ahead covariance matrix
    # This is used in covar() method and in the construction of parametric prediction intervals
    covarMat <- diag(h);
    if(h > min(lagsModel)){
            lagsUnique <- unique(lagsModel);
            steps <- sort(lagsUnique[lagsUnique<=h]);
            stepsNumber <- length(steps);
            nComponents <- nrow(transition);
            arrayTransition <- array(0,c(nComponents,nComponents,stepsNumber));
            arrayMeasurement <- array(0,c(1,nComponents,stepsNumber));
            for(i in 1:stepsNumber){
                arrayTransition[,lagsModel==steps[i],i] <- transition[,lagsModel==steps[i]];
                arrayMeasurement[,lagsModel==steps[i],i] <- measurement[,lagsModel==steps[i]];
            }
            cValues <- rep(0,h);

            # Prepare transition array
            transitionPowered <- array(0,c(nComponents,nComponents,h,stepsNumber));
            transitionPowered[,,1:min(steps),] <- diag(nComponents);

            # Generate values for the transition matrix
            for(i in (min(steps)+1):h){
                for(k in 1:sum(steps<i)){
                    # This needs to be produced only for the lower lag.
                    # Then it will be reused for the higher ones.
                    if(k==1){
                        for(j in 1:sum(steps<i)){
                            if(((i-steps[k])/steps[j]>1)){
                                transitionNew <- arrayTransition[,,j];
                            }
                            else{
                                transitionNew <- diag(nComponents);
                            }

                            # If this is a zero matrix, do simple multiplication
                            if(all(transitionPowered[,,i,k]==0)){
                                transitionPowered[,,i,k] <- (transitionNew %*%
                                                              transitionPowered[,,i-steps[j],k]);
                            }
                            else{
                                # Check that the multiplication is not an identity matrix
                                if(!all((transitionNew %*% transitionPowered[,,i-steps[j],k])==diag(nComponents))){
                                    transitionPowered[,,i,k] <- transitionPowered[,,i,k] + (transitionNew %*%
                                                                                          transitionPowered[,,i-steps[j],k]);
                                }
                            }
                        }
                    }
                    # Copy the structure from the lower lags
                    else{
                        transitionPowered[,,i,k] <- transitionPowered[,,i-steps[k]+1,1];
                    }
                    # Generate values of cj
                    cValues[i] <- cValues[i] + arrayMeasurement[,,k] %*% transitionPowered[,,i,k] %*% persistence;
                }
            }

            # Fill in diagonals
            for(i in 2:h){
                covarMat[i,i] <- covarMat[i-1,i-1] + cValues[i]^2;
            }
            # Fill in off-diagonals
            for(i in 1:h){
                for(j in 1:h){
                    if(i==j){
                        next;
                    }
                    else if(i==1){
                        covarMat[i,j] = cValues[j];
                    }
                    else if(i>j){
                        covarMat[i,j] <- covarMat[j,i];
                    }
                    else{
                        covarMat[i,j] = covarMat[i-1,j-1] + covarMat[1,j] * covarMat[1,i];
                    }
                }
            }
        }
        # Multiply the matrix by the one-step-ahead variance
        covarMat <- covarMat * s2;

        return(covarMat);
}

adamVarAnal <- function(lagsModel, h, measurement, transition, persistence, s2){
    #### The function returns variances for the multiplicative error ETS models
    # Prepare the necessary parameters
    lagsUnique <- unique(lagsModel);
    steps <- sort(lagsUnique[lagsUnique<=h]);
    stepsNumber <- length(steps);
    nComponents <- nrow(transition);
    k <- length(persistence);

    # Prepare the persistence array and measurement matrix
    arrayPersistenceQ <- array(0,c(nComponents,nComponents,stepsNumber));
    matrixMeasurement <- matrix(measurement,1,nComponents);

    # Form partial matrices for different steps
    for(i in 1:stepsNumber){
        arrayPersistenceQ[,lagsModel==steps[i],i] <- diag(as.vector(persistence),k,k)[,lagsModel==steps[i]];
    }

    ## The matrices that will be used in the loop
    matrixPersistenceQ <- matrix(0,nComponents,nComponents);
    # The interrim value (Q), which accumulates the values before taking logs
    IQ <- vector("numeric",1);
    Ik <- diag(k);

    # The vector of variances
    varMat <- rep(0, h);

    # Start the loop for varMat
    for(i in 2:h){
        IQ[] <- 0;
        # Form the correct interrim Q that will be used for variances
        for(k in 1:sum(steps<i)){
            matrixPersistenceQ[] <- arrayPersistenceQ[,,k];
            IQ[] <- IQ[] + sum(diag((matrixPowerWrap(Ik + matrixPowerWrap(matrixPersistenceQ,2)*s2,
                                                     ceiling(i/lagsUnique[k])-1) - Ik)));
        }
        varMat[i] <- log(IQ);
    }
    varMat[] <- exp(varMat)*(1+s2);
    varMat[1] <- varMat[1] - 1;
    varMat[-1] <- varMat[-1] + s2;

    return(varMat);
}
