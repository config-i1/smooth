sim.ces <- function(seasonality=c("N","S","P","F"), seas.freq=1,
             persistence=NULL, initial=NULL, initial.season=NULL,
             obs=10, nseries=1,silent=FALSE,
             randomizer=c("rnorm","runif","rbeta","rt"),
             ...){
# Function simulates the data using CES state-space framework
#
# seasonality - the type of seasonality to produce.
# seas.freq - the frequency of the data. In the case of seasonal models must be >1.
# persistence - is the persistence vector, that includes the smoothing parameters.
#    If NULL it will be generated.
# initial - the vector of initial states,
#    If NULL it will be generated.
# initial.season - the vector of initial states for seasonal coefficients (first m - real, the other m - imaginary).
#    If NULL it will be generated.
# obs - the number of observations in each time series.
# nseries - the number of series needed to be generated.
# silent - if TRUE no output is given.
# randomizer - the type of the random number generator function
# ... - the parameters passed to the randomizer.

    randomizer <- randomizer[1]
    seasonality <- seasonality[1]

# In the case of wrong nseries, make it natural number. The same is for obs and seas.freq.
    nseries <- abs(round(nseries,0))
    obs <- abs(round(obs,0))
    seas.freq <- abs(round(seas.freq,0))
    
# Check the used model and estimate the length of needed persistence vector.
    if(seasonality!="N" & seasonality!="S" & seasonality!="P" & seasonality!="F"){
        stop("Wrong seasonality type!",call.=FALSE)
    }
    else{
      if(seasonality!="N" & seas.freq==1){
        stop("Can't simulate seasonal data with seas.freq=1!",call.=FALSE)
      }
    }

    if(seasonality=="N"){
# number of smoothing parameters
        persistence.length <- 2
# number of non-seasonal components
        n.components <- 2
# w matrix
        mat.w <- diag(1,2)
# lag vector for the components of the model
        lags <- c(1,1)
# names of the components
        component.names <- c("level","potential")
# the inner frequency of the model
        model.freq <- 1
# number of seasonal components
        n.seas.components <- 0
    }
    else if(seasonality=="S"){
        persistence.length <- 2
        n.components <- 0
        mat.w <- diag(1,2)
        lags <- c(seas.freq,seas.freq)
        component.names <- c("level.s","potential.s")
        model.freq <- seas.freq
        n.seas.components <- 2
    }
    else if(seasonality=="P"){
        persistence.length <- 3
        n.components <- 2
        mat.w <- cbind(diag(1,2),c(1,0))
        lags <- c(1,1,seas.freq)
        component.names <- c("level","potential","seasonal")
        model.freq <- seas.freq
        n.seas.components <- 1
    }
    else if(seasonality=="F"){
        persistence.length <- 4
        n.components <- 2
        mat.w <- cbind(diag(1,2),diag(1,2))
        lags <- c(1,1,seas.freq,seas.freq)
        component.names <- c("level","potential","level.s","potential.s")
        model.freq <- seas.freq
        n.seas.components <- 2
    }

# If the chosen randomizer is not rnorm, rt and runif and no parameters are provided, change to rnorm.
    if(randomizer!="rnorm" & randomizer!="rt" & randomizer!="runif" & (any(names(match.call(expand.dots=FALSE)[-1]) == "...")==FALSE)){
      warning(paste0("The chosen randomizer - ",randomizer," - needs some arbitrary parameters! Changing to 'rnorm' now."),call.=FALSE)
      randomizer = "rnorm"
    }

# Check the persistence vector length
    if(!is.null(persistence)){
        if(persistence.length != length(persistence)){
            message("The length of persistence vector does not correspond to the chosen model!")
            message("Falling back to random number generator in... now!")
            persistence <- NULL
        }
        if(((persistence[1]-2.5)^2 + persistence[2]^2 <= 1.25) |
                  ((persistence[1]-0.5)^2 + (persistence[2]-1)^2 <= 0.25) |
                  ((persistence[1]-1.5)^2 + (persistence[2]-0.5)^2 >= 1.5)){
            message("ATTENTION! The model is unstable with the defined persistence vector!")
        }
    }

# Check the inital vector length
    if(!is.null(initial)){
        if(length(initial)>2){
            message("The length of the initial value is wrong! It should not be greater than 2.")
            message("The initials will be chosen randomly!")
            initial <- NULL
        }
    }

# Check the initial seasonal vector length
    if(seasonality!="N" & !is.null(initial.season)){
        if(model.freq!=(length(initial.season)/n.seas.components)){
            message("The length of seasonal initial states does not correspond to the chosen frequency!")
            message("Falling back to random number generator in... now!")
            initial.season <- NULL
        }
    }

##### Start filling in the matrices #####
# Create vector for the whole series
    z <- rep(NA,obs)
# Create vector for the real series
    y <- rep(NA,obs)
# Create vector for the imaginary series
    p <- rep(NA,obs)

# Create the matrix of state vectors
    mat.xt <- matrix(NA,nrow=(obs+model.freq),ncol=persistence.length)
    colnames(mat.xt) <- component.names

# If the initials are provided, write them down
    if(seasonality!="S" & !is.null(initial)){
        mat.xt[1:model.freq,1:n.components] <- rep(initial,each=model.freq)
    }

# If the seasonal model is chosen, fill in the first "seas.freq" values of seasonal component.
    if(seasonality!="N" & !is.null(initial.season)){
        mat.xt[1:model.freq,(n.components+1):(n.components+n.seas.components)] <- initial.season
    }

    if(nseries > 1){
# The array of the components
        arr.xt <- array(NA,c(obs+model.freq,persistence.length,nseries))
        dimnames(arr.xt)[[2]] <- c(component.names)
# The matrix of the final data
        mat.yt <- matrix(NA,obs,nseries)
# The matrix of the final imaginary data
        mat.pt <- matrix(NA,obs,nseries)
# The matrix of the error term
        mat.errors <- matrix(NA,obs,nseries)
# The matrix of smoothing parameters
        mat.g <- matrix(NA,nseries,persistence.length)
        colnames(mat.g) <- c(component.names)
# The vector of likelihoods
        vec.likelihood <- rep(NA,nseries)

        if(silent == FALSE){
          cat("Series simulated:  ")
        }
    }

##### Start the loop #####
for(k in 1:nseries){

# If the persistence is NULL or was of the wrong length, generate the values
    if(is.null(persistence)){
# Generate persistence values using batmanplot
        vec.g <- rep(NA,persistence.length);
        vec.g <- runif(persistence.length,1.5-sqrt(1.25),1.5+sqrt(1.25))
        while(((vec.g[1]-2.5)^2 + vec.g[2]^2 <= 1.25) |
                ((vec.g[1]-0.5)^2 + (vec.g[2]-1)^2 <= 0.25) |
                ((vec.g[1]-1.5)^2 + (vec.g[2]-0.5)^2 >= 1.5)){
            vec.g[2] <- runif(1,0.5-sqrt(1.5-(vec.g[1]-1.5)^2),0.5+sqrt(1.5-(vec.g[1]-1.5)^2))
        }

        if(persistence.length==3){
          vec.g[3] <- runif(1,0,1)
        }
        else if(persistence.length>3){
            while(((vec.g[3]-2.5)^2 + vec.g[4]^2 <= 1.25) |
                  ((vec.g[3]-0.5)^2 + (vec.g[4]-1)^2 <= 0.25) |
                  ((vec.g[3]-1.5)^2 + (vec.g[4]-0.5)^2 >= 1.5)){
                vec.g[4] <- runif(1,0.5-sqrt(1.5-(vec.g[3]-1.5)^2),0.5+sqrt(1.5-(vec.g[3]-1.5)^2))
            }
        }
    }
    else{
        vec.g <- persistence
    }

# Define transition matrix F
    mat.F <- matrix(c(1,1,-(1-vec.g[2]),(1-vec.g[1])),2,2)

    if(seasonality=="P"){
        mat.F <- cbind(mat.F,c(0,0))
        mat.F <- rbind(mat.F,c(0,0,1))
    }
    else if(seasonality=="F"){
        mat.F <- cbind(mat.F,c(0,0),c(0,0))
        mat.F <- rbind(mat.F,c(0,0,1,-(1-vec.g[4])),c(0,0,1,(1-vec.g[3])))
    }

# Generate initial states of level and trend if they were not supplied
    if(is.null(initial) & seasonality!="S"){
        mat.xt[1:model.freq,1] <- runif(1,-1000,1000)
        mat.xt[1:model.freq,2] <- mat.xt[1,1] / ((vec.g[1]!=0)*vec.g[1] + (vec.g[1]==0)*1)
    }

# Generate seasonal states if they were not supplied
    if(is.null(initial.season) & seasonality!="N"){
        if(seasonality=="S"){
            mat.xt[1:model.freq,1] <- runif(model.freq,-500,500)
            mat.xt[1:model.freq,1] <- mat.xt[1:model.freq,1] - mean(mat.xt[1:model.freq,1])
            mat.xt[1:model.freq,2] <- mat.xt[1:model.freq,1] / ((vec.g[1]!=0)*vec.g[1] + (vec.g[1]==0)*1)
        }
        else if(seasonality=="P"){
            mat.xt[1:model.freq,3] <- runif(model.freq,-500,500)
            mat.xt[1:model.freq,3] <- mat.xt[1:model.freq,3] - mean(mat.xt[1:model.freq,3])
        }
        else if(seasonality=="F"){
            mat.xt[1:model.freq,3] <- runif(model.freq,-500,500)
            mat.xt[1:model.freq,3] <- mat.xt[1:model.freq,3] - mean(mat.xt[1:model.freq,3])
            mat.xt[1:model.freq,4] <- mat.xt[1:model.freq,3] / ((vec.g[3]!=0)*vec.g[3] + (vec.g[3]==0)*1)
        }
    }

# Check if any argument was passed in dots
    if(any(names(match.call(expand.dots=FALSE)[-1]) == "...")==FALSE){
# Create vector of the errors
        if(randomizer=="rnorm" | randomizer=="runif"){
          errors <- eval(parse(text=paste0(randomizer,"(n=",obs,")")))
        }
        else if(randomizer=="rt"){
# The degrees of freedom are df = n - k.
          errors <- rt(obs,obs-(persistence.length + n.components + model.freq*n.seas.components))
        }

# Center errors just in case
        errors <- errors - mean(errors)
# Change variance to make some sense. Errors should not be rediculously high and not too low.
        errors <- errors * sqrt(abs(mat.xt[1,1]))
    }
# If arguments are passed, use them.
    else{
        errors <- eval(parse(text=paste0(randomizer,"(n=",obs,",", toString(as.character(list(...))),")")))

        if(randomizer=="rbeta"){
# Center the errors around 0.5
          errors <- errors - 0.5
# Make a meaningful variance of data. Something resembling to var=1.
          errors <- errors / sqrt(var(errors)) * sqrt(abs(mat.xt[1,1]))
        }
        else if(randomizer=="rt"){
# Make a meaningful variance of data.
          errors <- errors * sqrt(abs(mat.xt[1,1]))
        }

# Center errors in case all of them are positive or negative to get rid of systematic bias.
        if(all(errors>0) | all(errors<0)){
            errors <- errors - mean(errors)
        }
    }

###### Simulate the data #####
    j <- model.freq+1
    while(j<=(obs+model.freq)){
        z.mid <- mat.w %*% mat.xt[cbind((j-lags),c(1:persistence.length))] + errors[j-model.freq]
        z[j-model.freq] <- complex(real=z.mid[1],imaginary=z.mid[2])
        mat.xt[j,] <- mat.F %*% mat.xt[cbind((j-lags),c(1:persistence.length))] + vec.g * errors[j-model.freq]
        j <- j + 1
    }

    y <- Re(z)
    p <- Im(z)

    likelihood <- -obs/2 *(log(2*pi*exp(1)) + log(mean(errors^2)))

    if(nseries > 1){
        mat.yt[,k] <- y
        mat.pt[,k] <- p
        mat.errors[,k] <- errors
        arr.xt[,,k] <- mat.xt
        mat.g[k,] <- vec.g
        vec.likelihood[k] <- likelihood

# Print the number of processed series
        if (silent == FALSE){
          if(k<=10){
              cat("\b")
          }
          else if(k>10 & k<=100){
              cat("\b")
              cat("\b")
          }
          else if(k>100 & k<=1000){
              cat("\b")
              cat("\b")
              cat("\b")
          }
          else if(k>1000 & k<=10000){
              cat("\b")
              cat("\b")
              cat("\b")
              cat("\b")
          }
          else if(k>10000 & k<=100000){
              cat("\b")
              cat("\b")
              cat("\b")
              cat("\b")
              cat("\b")
          }
          else{
              cat("\b")
              cat("\b")
              cat("\b")
              cat("\b")
              cat("\b")
              cat("\b")            
          }
          cat(k)
        }
    }
}

    if(nseries==1){
        y <- ts(y,frequency=seas.freq)
        p <- ts(p,frequency=seas.freq)
        errors <- ts(errors,frequency=seas.freq)
        mat.xt <- ts(mat.xt,frequency=seas.freq,start=c(0,seas.freq-model.freq+1))

        return(list(data=y,ip=p,states=mat.xt,persistence=vec.g,residuals=errors,model=seasonality,likelihood=likelihood))
    }
    else{
        mat.yt <- ts(mat.yt,frequency=seas.freq)
        mat.pt <- ts(mat.yt,frequency=seas.freq)
        mat.errors <- ts(mat.errors,frequency=seas.freq)
        return(list(data=mat.yt,ip=mat.pt,states=arr.xt,persistence=mat.g,residuals=mat.errors,model=seasonality,likelihood=vec.likelihood))
    }
}