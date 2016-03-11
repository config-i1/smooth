ces.auto <- function(data, C=c(1.1, 1), models=c("N","F"),
                IC=c("CIC","AIC","AICc","BIC"),
                CF.type=c("MSE","MAE","HAM","trace","GV","TV","MSEh"),
                use.test=FALSE, intervals=FALSE, int.w=0.95,
                bounds=FALSE, holdout=FALSE, h=1, silent=FALSE, legend=TRUE,
                xreg=NULL){
# Function estimates several CES models in state-space form with sigma = error,
#  chooses the one with the lowest IC value and returns complex smoothing parameter
#  value, fitted values, residuals, point and interval forecasts, matrix of CES components
#  and values of information criteria

#    Copyright (C) 2015  Ivan Svetunkov

  CF.type <- CF.type[1];
# Check if CF.type is appropriate
    if(CF.type=="trace" | CF.type=="TV" | CF.type=="GV" | CF.type=="MSEh"){
        multisteps <- TRUE;
    }
    else if(CF.type=="MSE" | CF.type=="MAE" | CF.type=="HAM"){
        multisteps <- FALSE;
    }
    else{
        message(paste0("Strange cost function specified: ",CF.type,". Switching to 'MSE'."));
        CF.type <- "MSE";
        multisteps <- FALSE;
    }

# If the pool of models is wrong, fall back to default
  if(any(models!="N" & models!="S" & models!="P" & models!="F")){
    message("The pool of models includes a strange type of model! Reverting to default pool.");
    models <- c("N","S","P","F");
  }

  if(any(is.na(data))){
    if(silent==FALSE){
      message("Data contains NAs. These observations will be excluded.")
    }
    datanew <- data[!is.na(data)]
    if(is.ts(data)){
      datanew <- ts(datanew,start=start(data),frequency=frequency(data))
    }
    data <- datanew
  }

  if(frequency(data)==1){
    if(silent==FALSE){
      message("The data is not seasonal. Simple CES was the only solution here.");
    }
    models <- "N";

    ces.model <- ces(data,h=h,holdout=holdout,C=C,silent=silent,bounds=bounds,
                     seasonality=models,xreg=xreg,intervals=intervals,int.w=int.w,
                     use.test=use.test,CF.type=CF.type);
    return(ces.model);
  }

  IC <- IC[1]

  ces.model <- as.list(models);

  j <- 1;
  for(i in models){
    if(silent==FALSE){
      print(paste0("Estimating CES with seasonality = '",i,"'"));
    }
    ces.model[[j]] <- ces(data,h=h,holdout=holdout,C=C,silent=TRUE,bounds=bounds,
                     seasonality=i,xreg=xreg,intervals=intervals,int.w=int.w,
                     use.test=use.test,CF.type=CF.type);
    j <- j+1;
  }

  IC.vector <- c(1:length(models));

  for(i in 1:length(models)){
    IC.vector[i] <- ces.model[[i]]$ICs[IC];
  }

  best.model <- ces.model[[which(IC.vector==min(IC.vector))]];

  if(silent==FALSE){
    cat(" \n");
    print(paste0("The best model is with seasonality = '",models[which(IC.vector==min(IC.vector))],"'"));

# Define obs.all, the overal number of observations (in-sample + holdout)
    obs.all <- length(data) + (1 - holdout)*h;

# Define obs, the number of observations of in-sample
    obs <- length(data) - holdout*h;

    y.fit <- best.model$fitted;
    y.for <- best.model$forecast;
    y.high <- best.model$upper;
    y.low <- best.model$lower;

    print(paste0("a0 + ia1: ",best.model$A));
    if(models[which(IC.vector==min(IC.vector))]=="F"){
      print(paste0("b0 + ib1: ",best.model$B));
    }
    else if(models[which(IC.vector==min(IC.vector))]=="P"){
      print(paste0("b: ",best.model$B));
    }

    if(multisteps==FALSE){
      CF.type <- "1 step ahead";
    }
    print(paste0("Cost function used: ",CF.type));
    print(paste0("AIC: ",round(best.model$ICs["AIC"],3),"; AICc: ", round(best.model$ICs["AICc"],3),
                 "; BIC: ", round(best.model$ICs["BIC"],3), "; CIC: ", round(best.model$ICs["CIC"],3)));
    if(intervals==TRUE){
        print(paste0(int.w*100,"% intervals were constructed"));
        graphmaker(actuals=data,forecast=y.for,fitted=y.fit,
                   lower=y.low,upper=y.high,int.w=int.w,legend=legend);
    }
    else{
        graphmaker(actuals=data,forecast=y.for,fitted=y.fit,legend=legend);
    }
    if(holdout==T){
        errormeasures <- best.model$accuracy;
        if(intervals==TRUE){
            print(paste0(round(sum(as.vector(data)[(obs+1):obs.all]<y.high &
                    as.vector(data)[(obs+1):obs.all]>y.low)/h*100,0),
                    "% of values are in the interval"));
        }
        print(paste(paste0("MPE: ",errormeasures["MPE"]*100,"%"),
                    paste0("MAPE: ",errormeasures["MAPE"]*100,"%"),
                    paste0("SMAPE: ",errormeasures["SMAPE"]*100,"%"),sep="; "));
        print(paste(paste0("MASE: ",errormeasures["MASE"]),
                    paste0("MASALE: ",errormeasures["MASALE"]*100,"%"),sep="; "));
    }
  }

  return(best.model);
}
