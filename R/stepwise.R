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
    ourData <- ourData[apply(!is.na(ourData),1,all),]
    obs <- nrow(ourData)
    if(is.null(df)){
        df <- 0;
    }
    if(!is.data.frame(ourData)){
        ourData <- as.data.frame(ourData);
    }
    # Select IC
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
    # Run the simplest model y = const
    testFormula <- paste0(colnames(ourData)[1],"~ 1");
    testModel <- lm(as.formula(testFormula),data=ourData);
    # Write down the logLik and take df into account
    logLikValue <- logLik(testModel);
    attributes(logLikValue)$df <- attributes(logLikValue)$df + df;
    # Write down the IC
    currentIC <- bestIC <- IC(logLikValue);
    # Add residuals to the ourData
    ourData <- cbind(ourData,residuals(testModel));
    colnames(ourData)[ncol(ourData)] <- "const resid";
    bestFormula <- testFormula;
    if(!silent){
        cat(testFormula); cat(", "); cat(currentIC); cat("\n\n");
    }

    # Start the loop
    while(bestICNotFound){
        ourCorrelation <- cor(ourData,use="complete.obs");
        # Extract the last row of the correlation matrix
        ourCorrelation <- ourCorrelation[-1,-1];
        ourCorrelation <- ourCorrelation[nrow(ourCorrelation),];
        ourCorrelation <- ourCorrelation[1:ourncols];
        # Find the highest correlation coefficient
        newElement <- which(abs(ourCorrelation)==max(abs(ourCorrelation)))[1];
        newElement <- names(ourCorrelation)[newElement];
        # If the newElement is the same as before, stop
        if(any(newElement==all.vars(as.formula(bestFormula)))){
            bestICNotFound <- FALSE;
            break;
        }
        # Include the new element in the original model
        testFormula <- paste0(testFormula,"+",newElement);
        testModel <- lm(as.formula(testFormula),data=ourData);
        # Modify logLik
        logLikValue <- logLik(testModel);
        attributes(logLikValue)$df <- attributes(logLikValue)$df + df;
        if(attributes(logLikValue)$df >= (obs+1)){
            if(!silent){
                warning("Number of degrees of freedom is greater than number of observations. Cannot proceed.");
            }
            bestICNotFound <- FALSE;
            break;
        }

        # Calculate the IC
        currentIC <- IC(logLikValue);
        if(!silent){
            cat(testFormula); cat(", "); cat(currentIC); cat("\n");
            cat(round(ourCorrelation,3)); cat("\n\n");
        }
        # If IC is greater than the previous, then the previous model is the best
        if(currentIC >= bestIC){
            bestICNotFound <- FALSE;
        }
        else{
            bestIC <- currentIC;
            bestFormula <- testFormula;
            ourData[,ncol(ourData)] <- residuals(testModel);
        }
    }

    # Create an object of the same name as the original data
    # If it was a call on its own, make it one string
    assign(paste0(deparse(substitute(data)),collapse=""),as.data.frame(data));
    # Remove "1+" from the best formula
    bestFormula <- sub(" 1+", "", bestFormula,fixed=T);

    bestModel <- do.call("lm", list(formula=as.formula(bestFormula),
                                    data=substitute(data)));

    return(model=bestModel);
}
