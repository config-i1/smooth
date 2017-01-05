stepwise <- function(data, ic=c("AIC","AICc","BIC"), silent=TRUE, df=NULL){
##### Function that selects variables based on IC and using partial correlations
    ourData <- data;
    if(is.null(df)){
        df <- 0;
    }
    if(!is.data.frame(ourData)){
        ourData <- as.data.frame(ourData);
    }
    ic <- ic[1];
    if(ic=="AIC"){
        IC <- AIC;
    }
    else if(ic=="AICc"){
        IC <- AICc;
    }
    else if(ic=="BIC"){
        IC <- BIC;
    }
    ourncols <- ncol(ourData) - 1;
    bestICNotFound <- TRUE;
    testFormula <- paste0(colnames(ourData)[1],"~ 1");
    testModel <- lm(as.formula(testFormula),data=ourData);
    logLikValue <- logLik(testModel);
    attributes(logLikValue)$df <- attributes(logLikValue)$df + df;
    currentIC <- bestIC <- IC(logLikValue);
    ourData <- cbind(ourData,residuals(testModel));
    colnames(ourData)[ncol(ourData)] <- "const resid";
    bestFormula <- testFormula;
    if(!silent){
        cat(testFormula); cat(", "); cat(currentIC); cat("\n\n");
    }

    while(bestICNotFound){
        ourCorrelation <- cor(ourData);
        ourCorrelation <- ourCorrelation[-1,-1];
        ourCorrelation <- ourCorrelation[nrow(ourCorrelation),];
        ourCorrelation <- ourCorrelation[1:ourncols];
        newElement <- which(abs(ourCorrelation)==max(abs(ourCorrelation)))[1];
        newElement <- names(ourCorrelation)[newElement];
        if(any(newElement==all.vars(as.formula(bestFormula)))){
            bestICNotFound <- FALSE;
            break;
        }
        testFormula <- paste0(testFormula,"+",newElement);
        testModel <- lm(as.formula(testFormula),data=ourData);
        logLikValue <- logLik(testModel);
        attributes(logLikValue)$df <- attributes(logLikValue)$df + df;
        currentIC <- IC(logLikValue);
        if(!silent){
            cat(testFormula); cat(", "); cat(currentIC); cat("\n");
            cat(round(ourCorrelation,3)); cat("\n\n");
        }
        if(currentIC >= bestIC){
            bestICNotFound <- FALSE;
        }
        else{
            bestIC <- currentIC;
            bestFormula <- testFormula;
            ourData <- cbind(ourData,residuals(testModel));
            colnames(ourData)[ncol(ourData)] <- paste0(newElement," resid");
        }
    }
    bestModel <- lm(as.formula(bestFormula),data=ourData);
    # attributes(bestModel)$ic <- bestIC;

    return(model=bestModel);
}
