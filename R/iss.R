utils::globalVariables(c("y","obs"))

intermittentParametersSetter <- function(intermittent="n",...){
# Function returns basic parameters based on intermittent type
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    if(all(intermittent!=c("n","p"))){
        ot <- (y!=0)*1;
        obsNonzero <- sum(ot);
        # 1 parameter for estimating initial probability
        n.param.intermittent <- 1;
        if(intermittent=="c"){
            # In Croston we also need to estimate smoothing parameter and variance
#            n.param.intermittent <- n.param.intermittent + 2;
        }
        else if(any(intermittent==c("t","a"))){
            # In TSB we also need to estimate smoothing parameter and two parameters of distribution...
#            n.param.intermittent <- n.param.intermittent + 3;
        }
        yot <- matrix(y[y!=0],obsNonzero,1);
        pt <- matrix(mean(ot),obsInsample,1);
        pt.for <- matrix(1,h,1);
    }
    else{
        obsNonzero <- obsInsample;
    }

# If number of observations is low, set intermittency to "none"
    if(obsNonzero < 5){
        intermittent <- "n";
    }

    if(intermittent=="n"){
        ot <- rep(1,obsInsample);
        obsNonzero <- obsInsample;
        yot <- y;
        pt <- matrix(1,obsInsample,1);
        pt.for <- matrix(1,h,1);
        n.param.intermittent <- 0;
    }
    iprob <- pt[1];
    ivar <- iprob * (1-iprob);

    assign("ot",ot,ParentEnvironment);
    assign("obsNonzero",obsNonzero,ParentEnvironment);
    assign("yot",yot,ParentEnvironment);
    assign("pt",pt,ParentEnvironment);
    assign("pt.for",pt.for,ParentEnvironment);
    assign("n.param.intermittent",n.param.intermittent,ParentEnvironment);
    assign("iprob",iprob,ParentEnvironment);
    assign("ivar",ivar,ParentEnvironment);
}

intermittentMaker <- function(intermittent="n",...){
# Function returns all the necessary stuff from intermittent models
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

##### If intermittent is not auto, then work normally #####
    if(all(intermittent!=c("n","p","a"))){
        intermittent_model <- iss(y,intermittent=intermittent,h=h);
        pt[,] <- intermittent_model$fitted;
        pt.for <- intermittent_model$forecast;
        iprob <- pt.for[1];
        ivar <- intermittent_model$variance;
    }
    else{
        ivar <- 1;
    }

    assign("pt",pt,ParentEnvironment);
    assign("pt.for",pt.for,ParentEnvironment);
    assign("iprob",iprob,ParentEnvironment);
    assign("ivar",ivar,ParentEnvironment);
}

iss <- function(data, intermittent=c("none","fixed","croston","tsb"),
                h=10, imodel=NULL, ipersistence=NULL){
# Function estimates and returns mean and variance of probability for intermittent State-Space model based on the chosen method
    intermittent <- substring(intermittent[1],1,1);
    if(all(intermittent!=c("n","f","c","t"))){
        intermittent <- "f";
    }
    y <- data;
    obsInsample <- length(y);
    ot <- abs((y!=0)*1);
    iprob <- mean(ot);
    obsOnes <- sum(ot);
# Sizes of demand
    yot <- matrix(y[y!=0],obsOnes,1);

#### Fixed probability ####
    if(intermittent=="f"){
        pt <- ts(matrix(rep(iprob,obsInsample),obsInsample,1), start=start(data), frequency=frequency(data));
        pt.for <- ts(rep(iprob,h), start=time(data)[obsInsample]+deltat(data), frequency=frequency(data));
        errors <- ts(ot-iprob, start=start(data), frequency=frequency(data));

        model <- list(fitted=pt,forecast=pt.for,states=pt,variance=pt.for*(1-pt.for),
                      likelihood=0,residuals=errors,C=c(0,iprob),actuals=ot)
    }
#### Croston's method ####
    else if(intermittent=="c"){
# Define the matrix of states
        ivt <- matrix(rep(iprob,obsInsample+1),obsInsample+1,1);
# Define the matrix of actuals as intervals between demands
        zeroes <- c(0,which(y!=0),obsInsample+1);
### With this thing we fit model of the type 1/(1+qt)
#        zeroes <- diff(zeroes)-1;
        zeroes <- diff(zeroes);
# Number of intervals in Croston
        iyt <- matrix(zeroes,length(zeroes),1);
        if(is.null(imodel)){
            crostonModel <- es(iyt,"MNN",intervals=FALSE,int.w=0.95,silent=TRUE,h=h,ipersistence=ipersistence,intermittent="n");
        }
        else{
            crostonModel <- es(iyt,model=imodel,intervals=FALSE,int.w=0.95,silent=TRUE,h=h,ipersistence=ipersistence);
        }

        zeroes[length(zeroes)] <- zeroes[length(zeroes)] - 1;
        pt <- ts(rep((1-c(crostonModel$persistence)/2)/(crostonModel$fitted),zeroes),start=start(data),frequency=frequency(data));
        pt.for <- ts((1-c(crostonModel$persistence)/2)/(crostonModel$forecast), start=time(data)[obsInsample]+deltat(data),frequency=frequency(data));
        likelihood <- - (crostonModel$ICs["AIC"]/2 - 3);
        C <- c(crostonModel$persistence,crostonModel$states[1,]);
        names(C) <- c(paste0("persistence ",c(1:length(crostonModel$persistence))),
                      paste0("state ",c(1:length(crostonModel$states[1,]))))

        model <- list(fitted=pt,forecast=pt.for,states=1/crostonModel$states,variance=pt.for*(1-pt.for),
                      likelihood=likelihood,residuals=crostonModel$residuals,C=C,actuals=ot);
    }
#### TSB method ####
    else if(intermittent=="t"){
        ivt <- matrix(rep(iprob,obsInsample+1),obsInsample+1,1);
        iyt <- matrix(ot,obsInsample,1);
        modellags <- matw <- matF <- matrix(1,1,1);
        vecg <- matrix(0.01,1,1);
        errors <- matrix(NA,obsInsample,1);
        iyt.fit <- matrix(NA,obsInsample,1);

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

#### CF for beta distribution ####
        CF <- function(C){

            fitting <- fitterwrap(ivt, matF, matw, iy_kappa, vecg,
                                  modellags, "M", "N", "N", "o",
                                  matrix(0,obsInsample,1), matrix(0,obsInsample+1,1), matrix(1,1,1), matrix(1,1,1), matrix(1,obsInsample,1));

            iyt.fit <- fitting$yfit;
            errors <- fitting$errors;

            CF.res <- -(sum(log(dbeta(iyt.fit*(1+errors),shape1=C[1],shape2=C[2]))));

            return(CF.res);
        }

#### CF for initial and persistence ####
        CF2 <- function(C){
            vecg[,] <- C[1];
            ivt[1,] <- C[2];

            fitting <- fitterwrap(ivt, matF, matw, iy_kappa, vecg,
                                  modellags, "M", "N", "N", "o",
                                  matrix(0,obsInsample,1), matrix(0,obsInsample+1,1), matrix(1,1,1), matrix(1,1,1), matrix(1,obsInsample,1));

            iyt.fit <- fitting$yfit;

            # This is part of final likelihood. Otherwise it cannot be optimised...
            CF.res <- -(sum(log(iyt.fit[ot==1])) + sum(log(1-iyt.fit[ot==0])));

            return(CF.res);
        }

        kappa <- 1E-5;
        iy_kappa <- iyt*(1 - 2*kappa) + kappa;

        # Run in order to set shape1, shape2
        C <- c(0.5,0.5);
        res <- nloptr(C, CF, lb=c(1e-10,1e-10), ub=c(10,10),
                      opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-6, "maxeval"=100));
        likelihood <- -res$objective;
        C <- res$solution;

        # Another run, now to define persistence and initial
        res <- nloptr(c(vecg[1],ivt[1]), CF2, lb=c(1e-10,1e-10), ub=c(1,1),
                      opts=list("algorithm"="NLOPT_LN_BOBYQA", "xtol_rel"=1e-6, "maxeval"=100));
        C <- c(res$solution,C);

        names(C) <- c("persistence","initial","shape1","shape2")

        vecg[,] <- C[1];
        ivt[1,] <- C[2];
        iy_kappa <- iyt*(1 - 2*kappa) + kappa;
        fitting <- fitterwrap(ivt, matF, matw, iy_kappa, vecg,
                              modellags, "M", "N", "N", "o",
                              matrix(0,obsInsample,1), matrix(0,obsInsample+1,1), matrix(1,1,1), matrix(1,1,1), matrix(1,obsInsample,1));

        ivt <- ts(fitting$matvt,start=(time(data)[1] - deltat(data)),frequency=frequency(data));
        iyt.fit <- ts(fitting$yfit,start=start(data),frequency=frequency(data));
        errors <- ts(fitting$errors,start=start(data),frequency=frequency(data));
        iyt.for <- ts(rep(iyt.fit[obsInsample],h),
                     start=time(data)[obsInsample]+deltat(data),frequency=frequency(data));

        model <- list(fitted=iyt.fit,states=ivt,forecast=iyt.for,variance=iyt.for*(1-iyt.for),
                      likelihood=likelihood,residuals=errors,C=C,actuals=ot);
    }
#### None ####
    else{
        pt <- ts(rep(1,obsInsample),start=start(data),frequency=frequency(data));
        pt.for <- ts(rep(1,h), start=time(data)[obsInsample]+deltat(data),frequency=frequency(data));
        errors <- ts(rep(0,obsInsample), start=start(data), frequency=frequency(data));
        model <- list(fitted=pt,states=pt,forecast=pt.for,variance=rep(0,h),
                      likelihood=NULL,residuals=errors,C=c(0,1),actuals=pt);
    }
    model$intermittent <- intermittent;
    return(structure(model,class="iss"));
}
