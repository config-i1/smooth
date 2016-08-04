ssintermittent <- function(data, intermittent=c("fixed","croston","tsb")){
# Function checks the provided parameters and data for intermittency

}

iss <- function(data, intermittent=c("none","fixed","croston","tsb"),
                h=10, imodel=NULL, ipersistence=NULL){
# Function estimates and returns mean and variance of probability for intermittent State-Space model based on the chosen method
    intermittent <- substring(intermittent[1],1,1);
    if(all(intermittent!=c("n","f","c","t"))){
        intermittent <- "f";
    }
    y <- data;
    obs <- length(y);
    ot <- abs((y!=0)*1);
    iprob <- mean(ot);
    obs.ones <- sum(ot);
    obs.zeroes <- obs - obs.ones;
# Sizes of demand
    yot <- matrix(y[y!=0],obs.ones,1);

    if(intermittent=="f"){
        pt <- ts(matrix(rep(iprob,obs),obs,1), start=start(data), frequency=frequency(data));
        pt.for <- ts(rep(iprob,h), start=time(data)[obs]+deltat(data), frequency=frequency(data));
        errors <- ts(ot-iprob, start=start(data), frequency=frequency(data));
        return(list(fitted=pt,forecast=pt.for,variance=pt.for*(1-pt.for),
                    likelihood=0,residuals=errors,C=iprob));
    }
### Croston's method
    else if(intermittent=="c"){
# Define the matrix of states
        ivt <- matrix(rep(iprob,obs+1),obs+1,1);
# Define the matrix of actuals as intervals between demands
        zeroes <- c(0,which(y!=0),obs+1);
### With this thing we fit model of the type 1/(1+qt)
#        zeroes <- diff(zeroes)-1;
        zeroes <- diff(zeroes);
# Number of intervals in Croston
        obs.int <- length(zeroes);
        iyt <- matrix(zeroes,obs.int,1);
        if(is.null(imodel)){
            crostonModel <- es(iyt,"MNN",intervals=T,int.w=0.95,silent=TRUE,h=h,ipersistence=ipersistence,intermittent="n");
        }
        else{
            crostonModel <- es(iyt,model=imodel,intervals=T,int.w=0.95,silent=TRUE,h=h,ipersistence=ipersistence);
        }

        zeroes[length(zeroes)] <- zeroes[length(zeroes)] - 1;
        zeroes <- zeroes;
        pt <- ts(rep(1/(crostonModel$fitted),zeroes),start=start(data),frequency=frequency(data));
        pt.for <- ts(1/(crostonModel$forecast), start=time(data)[obs]+deltat(data),frequency=frequency(data));
        likelihood <- - (crostonModel$ICs["AIC"]/2 - 3);
        C <- c(crostonModel$persistence,crostonModel$states[1,]);
        names(C) <- c(paste0("persistence ",c(1:length(crostonModel$persistence))),
                      paste0("state ",c(1:length(crostonModel$states[1,]))))

        return(list(fitted=pt,forecast=pt.for,states=crostonModel$states,variance=pt.for*(1-pt.for),
                    likelihood=likelihood,residuals=crostonModel$residuals,C=C));
    }
### TSB method
    else if(intermittent=="t"){
        ivt <- matrix(rep(iprob,obs+1),obs+1,1);
        iyt <- matrix(ot,obs,1);
        modellags <- matw <- matF <- matrix(1,1,1);
        vecg <- matrix(0.01,1,1);
        errors <- matrix(NA,obs,1);
        iyt.fit <- matrix(NA,obs,1);

        if(!is.null(imodel)){
# If chosen model is "AAdN" or anything like that, we are taking the appropriate values
            if(nchar(imodel)==4){
                Etype <- substring(imodel,1,1);
                Ttype <- substring(imodel,2,2);
                Stype <- substring(imodel,4,4);
                damped <- TRUE;
                if(substring(imodel,3,3)!="d"){
                    message(paste0("You have defined a strange imodel: ",imodel));
                    sowhat(imodel);
                    imodel <- paste0(Etype,Ttype,"d",Stype);
                }
            }
            else if(nchar(imodel)==3){
                Etype <- substring(imodel,1,1);
                Ttype <- substring(imodel,2,2);
                Stype <- substring(imodel,3,3);
                damped <- FALSE;
            }
        }
        else{
            Etype <- "M";
            Ttype <- "N";
            Stype <- "N";
        }

        CF <- function(C){
            vecg[,] <- C[1];
            ivt[1,] <- C[2];

            fitting <- fitterwrap(ivt, matF, matw, iy_kappa, vecg,
                                  modellags, "M", "N", "N", "o",
                                  matrix(0,obs,1), matrix(0,obs+1,1), matrix(1,1,1), matrix(1,1,1), matrix(1,obs,1));

            iyt.fit <- fitting$yfit;
            errors <- fitting$errors;

            CF.res <- -sum(log(dbeta(iyt.fit*(1+errors),shape1=C[3],shape2=C[4])))

            return(CF.res);
        }

        kappa <- 1E-5;
        iy_kappa <- iyt*(1 - 2*kappa) + kappa;

# Smoothing parameter, initial
        C <- c(vecg[1],ivt[1],0.5,0.5);
        res <- nloptr(C, CF, lb=c(1e-10,1e-10,1e-10,1e-10), ub=c(1,1,10,10),
                      opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-6, "maxeval"=100));
        likelihood <- -res$objective;
        C <- res$solution;
        names(C) <- c("persistence","initial","shape1","shape2")

        vecg[,] <- C[1];
        ivt[1,] <- C[2];
        iy_kappa <- iyt*(1 - 2*kappa) + kappa;
        fitting <- fitterwrap(ivt, matF, matw, iy_kappa, vecg,
                              modellags, "M", "N", "N", "o",
                              matrix(0,obs,1), matrix(0,obs+1,1), matrix(1,1,1), matrix(1,1,1), matrix(1,obs,1));

        ivt <- ts(fitting$matvt,start=(time(data)[1] - deltat(data)),frequency=frequency(data));
        iyt.fit <- ts(fitting$yfit,start=start(data),frequency=frequency(data));
        errors <- ts(fitting$errors,start=start(data),frequency=frequency(data));
        iy.for <- ts(rep(iyt.fit[obs],h),
                     start=time(data)[obs]+deltat(data),frequency=frequency(data));

        return(list(fitted=iyt.fit,states=ivt,forecast=iy.for,variance=iy.for*(1-iy.for),
                    likelihood=likelihood,residuals=errors,C=C));
    }
    else{
        return(list(fitted=rep(1,obs),states=NULL,forecast=rep(1,h),variance=rep(0,h),
                    likelihood=NULL,residuals=rep(0,obs),C=NULL));
    }
}
