#' @rdname es
#' @export
es <- function(y, model="ZZZ", persistence=NULL, phi=NULL,
               initial=c("optimal","backcasting"), initialSeason=NULL, ic=c("AICc","AIC","BIC","BICc"),
               loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE"),
               h=10, holdout=FALSE,
               # cumulative=FALSE,
               # interval=c("none","parametric","likelihood","semiparametric","nonparametric"), level=0.95,
               bounds=c("usual","admissible","none"),
               silent=TRUE,
               xreg=NULL, regressors=c("use","select"), initialX=NULL, ...){
    # Copyright (C) 2022 - Inf  Ivan Svetunkov

    # Start measuring the time of calculations
    startTime <- Sys.time();
    cl <- match.call();
    ellipsis <- list(...);

    # Check if the simulated thing is provided
    if(is.smooth.sim(y)){
        if(smoothType(y)=="ETS"){
            model <- y;
            y <- y$data;
        }
    }
    else if(is.smooth(y)){
        model <- y;
        y <- y$y;
    }

    # If a previous model provided as a model, write down the variables
    if(is.smooth(model) | is.smooth.sim(model)){
        if(smoothType(model)!="ETS"){
            stop("The provided model is not ETS.",call.=FALSE);
        }
        # If this is the simulated data, extract the parameters
        if(is.smooth.sim(model) & !is.null(dim(model$data))){
            warning("The provided model has several submodels. Choosing a random one.",call.=FALSE);
            i <- round(runif(1,1:length(model$persistence)));
            persistence <- model$persistence[,i];
            initial <- model$initial[,i];
            initialSeason <- model$initialSeason[,i];
        }
        else{
            persistence <- model$persistence;
            initial <- model$initial;
            initialSeason <- model$initialSeason;
            if(any(model$probability!=1)){
                occurrence <- "a";
            }
        }
        phi <- model$phi;
        if(is.null(xreg)){
            xreg <- model$xreg;
        }
        else{
            if(is.null(model$xreg)){
                xreg <- NULL;
            }
            else{
                if(ncol(xreg)!=ncol(model$xreg)){
                    xreg <- xreg[,colnames(model$xreg)];
                }
            }
        }

        initialX <- model$initialX;
        model <- modelType(model);
        if(any(unlist(gregexpr("C",model))!=-1)){
            initial <- "o";
        }
    }
    else if(inherits(model,"ets")){
        # Extract smoothing parameters
        i <- 1;
        persistence <- coef(model)[i];
        if(model$components[2]!="N"){
            i <- i+1;
            persistence <- c(persistence,coef(model)[i]);
            if(model$components[3]!="N"){
                i <- i+1;
                persistence <- c(persistence,coef(model)[i]);
            }
        }
        else{
            if(model$components[3]!="N"){
                i <- i+1;
                persistence <- c(persistence,coef(model)[i]);
            }
        }

        # Damping parameter
        if(model$components[4]=="TRUE"){
            i <- i+1;
            phi <- coef(model)[i];
        }

        # Initials
        i <- i+1;
        initial <- coef(model)[i];
        if(model$components[2]!="N"){
            i <- i+1;
            initial <- c(initial,coef(model)[i]);
        }

        # Initials of seasonal component
        if(model$components[3]!="N"){
            if(model$components[2]!="N"){
                initialSeason <- rev(model$states[1,-c(1:2)]);
            }
            else{
                initialSeason <- rev(model$states[1,-c(1)]);
            }
        }
        model <- modelType(model);
    }
    else if(is.character(model)){
        # Everything is okay
    }
    else{
        warning("A model of an unknown class was provided. Switching to 'ZZZ'.",call.=FALSE);
        model <- "ZZZ";
    }

    # Merge y and xreg into one data frame
    if(!is.null(xreg) && is.numeric(y)){
        data <- cbind(y=as.data.frame(y),as.data.frame(xreg));
        data <- as.matrix(data)
        data <- ts(data, start=start(y), frequency=frequency(y));
        colnames(data)[1] <- "y";
        # Give name to the explanatory variables if they do not have them
        if(is.null(names(xreg))){
            if(!is.null(ncol(xreg))){
                colnames(data)[-1] <- paste0("x",c(1:ncol(xreg)));
            }
            else{
                colnames(data)[-1] <- "x";
            }
        }
    }
    else{
        data <- y;
    }

    # Prepare initials if they are numeric
    initialValue <- vector("list",(!is.null(initial))*1 +(!is.null(initialSeason))*1 +(!is.null(initialX))*1);
    names(initialValue) <- c("level","seasonal","xreg")[c(!is.null(initial),!is.null(initialSeason),!is.null(initialX))];
    if(is.numeric(initial)){
        initialValue <- switch(length(initial),
                               "1"=list(level=initial[1]),
                               "2"=list(level=initial[1],
                                        trend=initial[2]));
    }
    if(!is.null(initialSeason)){
        initialValue$seasonal <- initialSeason;
    }
    if(!is.null(initialX)){
        initialValue$xreg <- initialX;
    }
    if(length(initialValue)==1 && is.null(initialValue$level)){
        initialValue <- initial;
    }

    # Warnings about the interval and cumulative
    if(!is.null(ellipsis$interval) && ellipsis$interval!="none"){
        warning("Parameter \"interval\" is no longer supported in es(). ",
                "Please use forecast() method to produce prediction interval.")
    }

    if(!is.null(ellipsis$cumulative) && ellipsis$cumulative!="none"){
        warning("Parameter \"cumulative\" is no longer supported in es(). ",
                "Please use forecast() method to produce cumulative values.")
    }

    ourModel <- adam(data=data, model=model, persistence=persistence, phi=phi,
                     loss=loss, h=h, holdout=holdout, initial=initialValue,
                     ic=ic, bounds=bounds, distribution="dnorm",
                     silent=silent, regressors=regressors[1], ...);
    ourModel$call <- cl;
    ourModel$timeElapsed=Sys.time()-startTime;

    return(ourModel);
}
