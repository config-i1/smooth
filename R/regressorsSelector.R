regressorsSelector <- function(data, ic=c("AIC","AICc","BIC"), silent=TRUE){
    ##### Function that selects variables based on AIC and using partial correlations
    ourData <- data;
    ourncols <- ncol(ourData) - 1;
    bestICNotFound <- TRUE;
    testFormula <- paste0(colnames(ourData)[1],"~ 1");
    testModel <- lm(as.formula(testFormula),data=ourData);
    currentIC <- bestIC <- AIC(testModel);
    ourData <- cbind(ourData,residuals(testModel));
    colnames(ourData)[ncol(ourData)] <- "const resid";
    bestFormula <- testFormula;

    while(bestICNotFound){
        ourCorrelation <- cor(ourData);
        ourCorrelation <- ourCorrelation[-1,-1];
        ourCorrelation <- ourCorrelation[nrow(ourCorrelation),1:ourncols];
        newElement <- which(abs(ourCorrelation)==max(abs(ourCorrelation)))[1];
        newElement <- names(ourCorrelation)[newElement];
        testFormula <- paste0(testFormula,"+",newElement);
        testModel <- lm(as.formula(testFormula),data=ourData);
        currentIC <- AIC(testModel);
        if(!silent){
            cat(testFormula); cat(", "); cat(currentIC); cat("\n");
            cat(round(ourCorrelation,3)); cat("\n\n");
        }
        if(currentIC > bestIC){
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
    return(list(model=bestModel,formula=as.formula(bestFormula)));
}
