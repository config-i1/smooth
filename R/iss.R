utils::globalVariables(c("yInSample","obs","occurrenceModelProvided","occurrenceModel","occurrenceModel"))

intermittentParametersSetter <- function(occurrence="n",...){
# Function returns basic parameters based on occurrence type
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    if(all(occurrence!=c("n","p"))){
        ot <- (yInSample!=0)*1;
        obsNonzero <- sum(ot);
        obsZero <- obsInSample - obsNonzero;
        # 1 parameter for estimating initial probability. Works for the fixed probability model
        nParamOccurrence <- 1;
        if(any(occurrence==c("o","i","d"))){
            # The minimum number of parameters for these models is 2: level, alpha
            nParamOccurrence <- nParamOccurrence + 1;
        }
        else if(any(occurrence==c("g","a"))){
            # In "general" and "auto" the max number is 4
            nParamOccurrence <- nParamOccurrence + 3;
        }
        # Demand sizes
        yot <- matrix(yInSample[yInSample!=0],obsNonzero,1);
        if(!occurrenceModelProvided){
            pFitted <- matrix(mean(ot),obsInSample,1);
            pForecast <- matrix(1,h,1);
        }
        else{
            if(length(fitted(occurrenceModel))>obsInSample){
                pFitted <- matrix(fitted(occurrenceModel)[1:obsInSample],obsInSample,1);
            }
            else if(length(fitted(occurrenceModel))<obsInSample){
                pFitted <- matrix(c(fitted(occurrenceModel),
                               rep(fitted(occurrenceModel)[length(fitted(occurrenceModel))],obsInSample-length(fitted(occurrenceModel)))),
                             obsInSample,1);
            }
            else{
                pFitted <- matrix(fitted(occurrenceModel),obsInSample,1);
            }

            if(length(occurrenceModel$forecast)>=h){
                pForecast <- matrix(occurrenceModel$forecast[1:h],h,1);
            }
            else{
                pForecast <- matrix(c(occurrenceModel$forecast,
                                   rep(occurrenceModel$forecast[1],h-length(occurrenceModel$forecast))),h,1);
            }

        }
    }
    else{
        obsNonzero <- obsInSample;
        obsZero <- 0;
    }

    if(occurrence=="n"){
        ot <- rep(1,obsInSample);
        obsNonzero <- obsInSample;
        yot <- yInSample;
        pFitted <- matrix(1,obsInSample,1);
        pForecast <- matrix(1,h,1);
        nParamOccurrence <- 0;
    }
    ot <- ts(ot,start=dataStart,frequency=dataFreq);

    assign("ot",ot,ParentEnvironment);
    assign("obsNonzero",obsNonzero,ParentEnvironment);
    assign("obsZero",obsZero,ParentEnvironment);
    assign("yot",yot,ParentEnvironment);
    assign("pFitted",pFitted,ParentEnvironment);
    assign("pForecast",pForecast,ParentEnvironment);
    assign("nParamOccurrence",nParamOccurrence,ParentEnvironment);
}

intermittentMaker <- function(occurrence="n",...){
# Function returns all the necessary stuff from occurrence models
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

##### If occurrence is not absent or provided, then work normally #####
    if(all(occurrence!=c("n","p"))){
        if(!occurrenceModelProvided){
            occurrenceModel <- oes(ot, model=occurrenceModel, occurrence=occurrence, h=h);
        }
        else{
            occurrenceModel <- oes(ot, model=occurrenceModel, h=h);
        }
        nParamOccurrence <- nparam(occurrenceModel);
        pFitted[,] <- fitted(occurrenceModel);
        pForecast <- occurrenceModel$forecast;
        occurrence <- occurrenceModel$occurrence;
    }
    else{
        occurrenceModel <- NULL;
        nParamOccurrence <- 0;
    }

    assign("occurrence",occurrence,ParentEnvironment);
    assign("pFitted",pFitted,ParentEnvironment);
    assign("pForecast",pForecast,ParentEnvironment);
    assign("nParamOccurrence",nParamOccurrence,ParentEnvironment);
    assign("occurrenceModel",occurrenceModel,ParentEnvironment);
}
