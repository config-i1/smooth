#' Stepwise selection of regressors
#'
#' Function selects variables that give linear regression with the lowest
#' information criteria. The selection is done stepwise (forward) based on
#' partial correlations. This should be a simpler and faster implementation
#' than step() function from `stats' package.
#'
#' The algorithm uses lm() to fit different models and cor() to select the next
#' regressor in the sequence.
#'
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param data Data frame containing dependant variable in the first column and
#' the others in the rest.
#' @param ic Information criterion to use.
#' @param silent If \code{silent=FALSE}, then nothing is silent, everything is
#' printed out. \code{silent=TRUE} means that nothing is produced.
#' @param df Number of degrees of freedom to add (should be used if stepwise is
#' used on residuals).
#'
#' @return Function returns \code{model} - the final model of the class "lm".
#'
#' @seealso \code{\link[stats]{step}, \link[smooth]{xregExpander}}
#'
#' @keywords stepwise linear regression
#'
#' @examples
#'
#' xreg <- cbind(rnorm(100,10,3),rnorm(100,50,5))
#' xreg <- cbind(100+0.5*xreg[,1]-0.75*xreg[,2]+rnorm(100,0,3),xreg,rnorm(100,300,10))
#' colnames(xreg) <- c("y","x1","x2","Noise")
#'
#' stepwise(xreg)
#'
#' @export stepwise
stepwise <- function(data, ic=c("AICc","AIC","BIC"), silent=TRUE, df=NULL){
##### Function that selects variables based on IC and using partial correlations
    ourData <- data;
    obs <- nrow(ourData)
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
        if(attributes(logLikValue)$df >= (obs+1)){
            if(!silent){
                warning("Number of degrees of freedom is greater than number of observations. Cannot proceed.");
            }
            bestICNotFound <- FALSE;
            break;
        }

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

    return(model=bestModel);
}
