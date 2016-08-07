utils::globalVariables(c("estimate.initial","estimate.measurement","estimate.initial","estimate.transition",
                         "estimate.persistence","obs.xt"));

ges <- function(data, orders=c(2), lags=c(1), initial=c("optimal","backcasting"),
                persistence=NULL, transition=NULL, measurement=NULL,
                CF.type=c("MSE","MAE","HAM","MLSTFE","TFL","MSTFE","MSEh"),
                h=10, holdout=FALSE, intervals=FALSE, int.w=0.95,
                int.type=c("parametric","semiparametric","nonparametric","asymmetric"),
                intermittent=c("auto","none","fixed","croston","tsb"),
                bounds=c("admissible","none"), silent=c("none","all","graph","legend","output"),
                xreg=NULL, initialX=NULL, go.wild=FALSE, persistenceX=NULL, transitionX=NULL, ...){
# General Exponential Smoothing function. Crazy thing...
#
#    Copyright (C) 2016  Ivan Svetunkov

# Start measuring the time of calculations
    start.time <- Sys.time();

##### Set environment for ssinput and make all the checks #####
    environment(ssinput) <- environment();
    ssinput(modelType="ges",ParentEnvironment=environment());

##### !!!Temporary override for intermitency!!! #####
    if(intermittent=="a"){
        intermittent <- "n"
    }

##### Prepare exogenous variables #####
    xregdata <- ssxreg(data=data, xreg=xreg, go.wild=go.wild,
                       persistenceX=persistenceX, transitionX=transitionX, initialX=initialX,
                       obs=obs, obs.all=obs.all, obs.xt=obs.xt, maxlag=maxlag, h=h, silent=silent.text);
    n.exovars <- xregdata$n.exovars;
    matxt <- xregdata$matxt;
    matat <- xregdata$matat;
    matFX <- xregdata$matFX;
    vecgX <- xregdata$vecgX;
    estimate.xreg <- xregdata$estimate.xreg;
    estimate.FX <- xregdata$estimate.FX;
    estimate.gX <- xregdata$estimate.gX;
    estimate.initialX <- xregdata$estimate.initialX;

# 1 stands for the variance
    n.param <- 1 + n.components*estimate.measurement + n.components*(fittertype=="o")*estimate.initial +
        n.components^2*estimate.transition + orders %*% lags * estimate.persistence +
        estimate.initialX*n.exovars + estimate.FX*(n.exovars^2) + estimate.gX*(n.exovars);

# These three are needed in order to use ssgeneralfun.cpp functions
    Etype <- "A";
    Ttype <- "N";
    Stype <- "N";

elements.ges <- function(C){
    n.coef <- 0;
    if(estimate.measurement==TRUE){
        matw <- matrix(C[n.coef+(1:n.components)],1,n.components);
        n.coef <- n.coef + n.components;
    }
    else{
        matw <- matrix(measurement,1,n.components);
    }

    if(estimate.transition==TRUE){
        matF <- matrix(C[n.coef+(1:(n.components^2))],n.components,n.components);
        n.coef <- n.coef + n.components^2;
    }
    else{
        matF <- matrix(transition,n.components,n.components);
    }

    if(estimate.persistence==TRUE){
        vecg <- matrix(C[n.coef+(1:n.components)],n.components,1);
        n.coef <- n.coef + n.components;
    }
    else{
        vecg <- matrix(persistence,n.components,1);
    }

    if(fittertype=="o"){
        if(estimate.initial==TRUE){
            vtvalues <- C[n.coef+(1:(orders %*% lags))];
            n.coef <- n.coef + orders %*% lags;

        }
        else{
            vtvalues <- initial;
        }

        vt <- matrix(NA,maxlag,n.components);
        for(i in 1:n.components){
            vt[(maxlag - modellags + 1)[i]:maxlag,i] <- vtvalues[((cumsum(c(0,modellags))[i]+1):cumsum(c(0,modellags))[i+1])];
            vt[is.na(vt[1:maxlag,i]),i] <- rep(rev(vt[(maxlag - modellags + 1)[i]:maxlag,i]),
                                               ceiling((maxlag - modellags + 1) / modellags)[i])[is.na(vt[1:maxlag,i])];
        }
    }
    else{
        vt <- matvt[1:maxlag,n.components];
    }

# If exogenous are included
    if(estimate.xreg==TRUE){
        at <- matrix(NA,maxlag,n.exovars);
        if(estimate.initialX==TRUE){
            at[,] <- rep(C[n.coef+(1:n.exovars)],each=maxlag);
            n.coef <- n.coef + n.exovars;
        }
        else{
            at <- matat[1:maxlag,];
        }
        if(estimate.FX==TRUE){
            matFX <- matrix(C[n.coef+(1:(n.exovars^2))],n.exovars,n.exovars);
            n.coef <- n.coef + n.exovars^2;
        }

        if(estimate.gX==TRUE){
            vecgX <- matrix(C[n.coef+(1:n.exovars)],n.exovars,1);
            n.coef <- n.coef + n.exovars;
        }
    }
    else{
        at <- matrix(0,maxlag,n.exovars);
    }

    return(list(matw=matw,matF=matF,vecg=vecg,vt=vt,at=at,matFX=matFX,vecgX=vecgX));
}

# Cost function for GES
CF <- function(C){
    elements <- elements.ges(C);
    matw <- elements$matw;
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:maxlag,] <- elements$vt;
    matat[1:maxlag,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

    CF.res <- costfunc(matvt, matF, matw, y, vecg,
                       h, modellags, Etype, Ttype, Stype,
                       multisteps, CF.type, normalizer, fittertype,
                       matxt, matat, matFX, vecgX, ot,
                       bounds);

    return(CF.res);
}

# Likelihood function
Likelihood.value <- function(C){
    if(CF.type=="TFL"){
        return(obs.ot*log(iprob)*(h^multisteps)
               -obs.ot/2 *((h^multisteps)*log(2*pi*exp(1)) + CF(C)));
    }
    else{
        return(obs.ot*log(iprob)
               -obs.ot/2 *(log(2*pi*exp(1)) + log(CF(C))));
    }
}

##### Start the calculations #####
    matvt <- matrix(NA,nrow=obs.xt,ncol=n.components);
    y.fit <- rep(NA,obs);
    errors <- rep(NA,obs);

    if(multisteps==TRUE){
        normalizer <- sum(abs(diff(c(y))))/(obs-1);
    }
    else{
        normalizer <- 0;
    }

# If there is something to optimise, let's do it.
    if((estimate.initial==TRUE) | (estimate.measurement==TRUE) | (estimate.transition==TRUE) | (estimate.persistence==TRUE) |
       (estimate.xreg==TRUE) | (estimate.FX==TRUE) | (estimate.gX==TRUE)){
# Initial values of matvt
        slope <- cov(yot[1:min(12,obs.ot),],c(1:min(12,obs.ot)))/var(c(1:min(12,obs.ot)));
        intercept <- sum(yot[1:min(12,obs.ot),])/min(12,obs.ot) - slope * (sum(c(1:min(12,obs.ot)))/min(12,obs.ot) - 1);

        C <- NULL;
# matw, matF, vecg, vt
        if(estimate.measurement==TRUE){
            C <- c(C,rep(1,n.components));
        }
        if(estimate.transition==TRUE){
            C <- c(C,rep(1,n.components^2));
        }
        if(estimate.persistence==TRUE){
            C <- c(C,rep(0.1,n.components));
        }
        if(estimate.initial==TRUE){
            if(fittertype=="o"){
                C <- c(C,intercept);
                if((orders %*% lags)>1){
                    C <- c(C,slope);
                }
                if((orders %*% lags)>2){
                    C <- c(C,yot[1:(orders %*% lags-2),]);
                }
            }
            else{
                vtvalues <- intercept;
                if((orders %*% lags)>1){
                    vtvalues <- c(vtvalues,slope);
                }
                if((orders %*% lags)>2){
                    vtvalues <- c(vtvalues,yot[1:(orders %*% lags-2),]);
                }

                vt <- matrix(NA,maxlag,n.components);
                for(i in 1:n.components){
                    vt[(maxlag - modellags + 1)[i]:maxlag,i] <- vtvalues[((cumsum(c(0,modellags))[i]+1):cumsum(c(0,modellags))[i+1])];
                    vt[is.na(vt[1:maxlag,i]),i] <- rep(rev(vt[(maxlag - modellags + 1)[i]:maxlag,i]),
                                                ceiling((maxlag - modellags + 1) / modellags)[i])[is.na(vt[1:maxlag,i])];
                }
                matvt[1:maxlag,] <- vt;
            }
        }

# initials, transition matrix and persistence vector
        if(estimate.xreg==TRUE){
            if(estimate.initialX==TRUE){
                C <- c(C,matat[maxlag,]);
            }
            if(go.wild==TRUE){
                if(estimate.FX==TRUE){
                    C <- c(C,c(diag(n.exovars)));
                }
                if(estimate.gX==TRUE){
                    C <- c(C,rep(0,n.exovars));
                }
            }
        }

        elements <- elements.ges(C);
        matw <- elements$matw;
        matF <- elements$matF;
        vecg <- elements$vecg;
        matvt[1:maxlag,] <- elements$vt;
        matat[1:maxlag,] <- elements$at;
        matFX <- elements$matFX;
        vecgX <- elements$vecgX;

# Optimise model. First run
        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-8, "maxeval"=5000));
        C <- res$solution;

# Optimise model. Second run
        res <- nloptr(C, CF, opts=list("algorithm"="NLOPT_LN_NELDERMEAD", "xtol_rel"=1e-10, "maxeval"=1000));
        C <- res$solution;
        CF.objective <- res$objective;
    }
    else{
# matw, matF, vecg, vt
        C <- c(measurement,
               c(transition),
               c(persistence),
               c(initial));

        C <- c(C,matat[maxlag,],
               c(transitionX),
               c(persistenceX));

        CF.objective <- CF(C);
    }

# Change the CF.type in orders to calculate likelihood correctly.
    if(multisteps==TRUE){
        CF.type <- "TFL";
    }
    else{
        CF.type <- "MSE";
    }

    if(FI==TRUE){
        FI <- hessian(Likelihood.value,C);
    }
    else{
        FI <- NA;
    }

# Prepare for fitting
    elements <- elements.ges(C);
    matw <- elements$matw;
    matF <- elements$matF;
    vecg <- elements$vecg;
    matvt[1:maxlag,] <- elements$vt;
    matat[1:maxlag,] <- elements$at;
    matFX <- elements$matFX;
    vecgX <- elements$vecgX;

    fitting <- fitterwrap(matvt, matF, matw, y, vecg,
                          modellags, Etype, Ttype, Stype, fittertype,
                          matxt, matat, matFX, vecgX, ot);
    matvt <- fitting$matvt;
    y.fit <- ts(fitting$yfit,start=start(data),frequency=frequency(data));
    matat[1:nrow(fitting$matat),] <- fitting$matat;

    errors.mat <- ts(errorerwrap(matvt, matF, matw, y,
                                 h, Etype, Ttype, Stype, modellags,
                                 matxt, matat, matFX, ot),
                     start=start(data),frequency=frequency(data));
    colnames(errors.mat) <- paste0("Error",c(1:h));
    errors <- ts(fitting$errors,start=start(data),frequency=frequency(data));

    y.for <- ts(iprob*forecasterwrap(matrix(matvt[(obs+1):(obs+maxlag),],nrow=maxlag),
                               matF, matw, h, Ttype, Stype, modellags,
                               matrix(matxt[(obs.all-h+1):(obs.all),],ncol=n.exovars),
                               matrix(matat[(obs.all-h+1):(obs.all),],ncol=n.exovars), matFX),
                start=time(data)[obs]+deltat(data),frequency=datafreq);

#    s2 <- mean(errors^2);
    s2 <- as.vector(sum((errors*ot)^2)/(obs.ot-n.param));

    if(any(is.na(y.fit),is.na(y.for))){
        message("Something went wrong during the optimisation and NAs were produced!");
        message("Please check the input and report this error if it persists to the maintainer.");
    }

    if(intervals==TRUE){
        if(h==1){
            errors.x <- as.vector(errors);
            ev <- median(errors);
        }
        else{
            errors.x <- errors.mat;
            ev <- apply(errors.mat,2,median,na.rm=TRUE);
        }
        if(int.type!="a"){
            ev <- 0;
        }

        vt <- matrix(matvt[cbind(obs-modellags,c(1:n.components))],n.components,1);

        quantvalues <- ssintervals(errors.x, ev=ev, int.w=int.w, int.type=int.type, df=(obs.ot - n.param),
                                 measurement=matw, transition=matF, persistence=vecg, s2=s2, modellags=modellags,
                                 y.for=y.for, iprob=iprob);
        y.low <- ts(c(y.for) + quantvalues$lower,start=start(y.for),frequency=frequency(data));
        y.high <- ts(c(y.for) + quantvalues$upper,start=start(y.for),frequency=frequency(data));
    }
    else{
        y.low <- NA;
        y.high <- NA;
    }

    environment(likelihoodFunction) <- environment();
    environment(ICFunction) <- environment();

    IC.values <- ICFunction(n.param=n.param+n.param.intermittent,C=C,Etype=Etype);
    ICs <- IC.values$ICs;

# Revert to the provided cost function
    CF.type <- CF.type.original

# Write down initials of states vector and exogenous
    if(estimate.initial==TRUE){
        initial <- C[2*n.components+n.components^2+(1:(orders %*% lags))];
    }
    if(estimate.initialX==TRUE){
        initialX <- matat[1,];
    }

# Fill in the rest of matvt
    matvt <- ts(matvt,start=(time(data)[1] - deltat(data)*maxlag),frequency=frequency(data));
    if(!is.null(xreg)){
        matvt <- cbind(matvt,matat);
        colnames(matvt) <- c(paste0("Component ",c(1:n.components)),colnames(matat));
    }
    else{
        colnames(matvt) <- paste0("Component ",c(1:n.components));
    }

    if(holdout==T){
        y.holdout <- ts(data[(obs+1):obs.all],start=start(y.for),frequency=frequency(data));
        errormeasures <- errorMeasurer(y.holdout,y.for,y);
    }
    else{
        y.holdout <- NA;
        errormeasures <- NA;
    }

    modelname <- paste0("GES(",paste(orders,"[",lags,"]",collapse=",",sep=""),")");

    if(silent.text==FALSE){
        if(any(abs(eigen(matF - vecg %*% matw)$values)>(1 + 1E-10))){
            if(bounds!="a"){
                message("Unstable model was estimated! Use bounds='admissible' to address this issue!");
            }
            else{
                message("Something went wrong in optimiser - unstable model was estimated! Please report this error to the maintainer.");
            }
        }
# Calculate the number os observations in the interval
        if(all(holdout==TRUE,intervals==TRUE)){
            insideintervals <- sum(as.vector(data)[(obs+1):obs.all]<=y.high &
                                as.vector(data)[(obs+1):obs.all]>=y.low)/h*100;
        }
        else{
            insideintervals <- NULL;
        }
# Print output
        ssoutput(Sys.time() - start.time, modelname, persistence=vecg, transition=matF, measurement=matw,
                phi=NULL, ARterms=NULL, MAterms=NULL, const=NULL, A=NULL, B=NULL,
                n.components=orders %*% lags, s2=s2, hadxreg=!is.null(xreg), wentwild=go.wild,
                CF.type=CF.type, CF.objective=CF.objective, intervals=intervals,
                int.type=int.type, int.w=int.w, ICs=ICs,
                holdout=holdout, insideintervals=insideintervals, errormeasures=errormeasures);
    }
# Make plot
    if(silent.graph==FALSE){
        if(intervals==TRUE){
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit, lower=y.low,upper=y.high,
                       int.w=int.w,legend=legend,main=modelname);
        }
        else{
            graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                    int.w=int.w,legend=legend,main=modelname);
        }
    }

return(list(model=modelname,states=matvt,initial=initial,measurement=matw,transition=matF,persistence=vecg,
            fitted=y.fit,forecast=y.for,lower=y.low,upper=y.high,residuals=errors,errors=errors.mat,
            actuals=data,holdout=y.holdout,
            xreg=xreg,persistenceX=vecgX,transitionX=matFX,initialX=initialX,
            ICs=ICs,CF=CF.objective,CF.type=CF.type,FI=FI,accuracy=errormeasures));
}
