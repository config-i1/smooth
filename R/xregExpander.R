#' Exogenous variables expander
#'
#' Function expands the provided matrix or vector of variables, producing
#' values with lags and leads specified by \code{lags} variable.
#'
#' This function could be handy when you want to check if lags and leads
#' of a variable influence the dependent variable. Can be used together
#' with \code{xregDo="select"} in \link[smooth]{es}, \link[smooth]{ces},
#' \link[smooth]{ges} and \link[smooth]{ssarima}. All the missing values
#' in the beginning and at the end of lagged series are substituted by
#' mean forecasts produced using \link[smooth]{es}.
#'
#' @param xreg Vector / matrix / data.frame, containing variables that need
#' to be expanded. In case of vector / matrix it is recommended to provide
#' \code{ts} object, so the frequency of the data is taken into account.
#' @param lags Vector of lags / leads that we need to have. Negative values
#' mean lags, positive ones mean leads.
#' @param silent If \code{silent=FALSE}, then the progress is printed out.
#' Otherwise the function won't print anything in the console.
#'
#' @return \code{ts} matrix with the expanded variables is returned.
#'
#' @author Ivan Svetunkov, \email{ivan@svetunkov.ru}
#'
#' @seealso \code{\link[smooth]{es}, \link[smooth]{stepwise}}
#'
#' @keywords regression smooth
#'
#' @examples
#' # Create matrix of two variables, make it ts object and expand it
#' x <- cbind(rnorm(100,100,1),rnorm(100,50,3))
#' x <- ts(x,frequency=12)
#' xregExpander(x)
#' @export xregExpander

xregExpander <- function(xreg, lags=c(-frequency(xreg):frequency(xreg)),
                         silent=TRUE){

    lagsOriginal <- lags;
    # Remove zero from lags
    lags <- lags[lags!=0]
    lagsLengthAll <- length(lags);
    if(lagsLengthAll==0){
        warning("You have not specified any leads or lags.",call.=FALSE);
        return(xreg);
    }
    # Form leads
    leads <- lags[lags>0];
    leadsLength <- length(leads);
    if(leadsLength!=0){
        maxLead <- max(leads);
    }
    else{
        maxLead <- 0;
    }

    # Form proper lags
    lags <- abs(lags[lags<0]);
    lagsLength <- length(lags);
    if(lagsLength!=0){
        maxLag <- max(lags);
    }
    else{
        maxLag <- 0;
    }

    if(!silent){
        cat("Preparing matrices...    ");
    }

    if(is.data.frame(xreg)){
        xreg <- as.matrix(xreg);
    }

    if(!is.matrix(xreg) & (is.vector(xreg) | is.ts(xreg))){
        xregNames <- names(xreg)
        if(is.null(xregNames)){
            xregNames <- "x";
        }
        xreg <- ts(matrix(xreg),start=start(xreg),frequency=frequency(xreg));
        colnames(xreg) <- xregNames;
    }

    if(is.matrix(xreg)){
        xregStart <- start(xreg);
        xregFrequency <- frequency(xreg);
        xregNames <- colnames(xreg);
        if(is.null(xregNames)){
            xregNames <- paste0("x",1:ncol(xreg));
        }
        obs <- nrow(xreg);
        nExovars <- ncol(xreg);
        xregNew <- matrix(NA,obs,(lagsLengthAll+1)*nExovars);
        xregNew <- ts(xregNew,start=xregStart,frequency=xregFrequency);

        for(i in 1:nExovars){
            if(!silent){
                if(i==1){
                    cat("\b");
                }
                cat(paste0(rep("\b",nchar(round((i-1)/nExovars,2)*100)+1),collapse=""));
                cat(paste0(round(i/nExovars,2)*100,"%"));
            }
            chosenColumn <- (lagsLengthAll+1)*(i-1);
            xregNew[,chosenColumn+1] <- xregData <- xreg[,i];
            xregCurrentName <- xregNames[i];
            colnames(xregNew)[(lagsLengthAll+1)*(i-1)+1] <- xregCurrentName;
            xregDataNew <- xregData;
            if(leadsLength!=0){
            # Produce forecasts for leads
            # If this is a binary variable, use iss function.
                if(all((xregData==0) | (xregData==1))){
                    xregModel <- suppressWarnings(iss(xregData,model="XXX", h=maxLead,intermittent="l"));
                }
                else{
                    xregModel <- suppressWarnings(es(xregData,h=maxLead,intermittent="a"));
                }
                xregDataNew <- c(xregDataNew,xregModel$forecast);
            }
            if(lagsLength!=0){
            # Produce reversed forecasts for lags
                if(leadsLength!=0){
                    # If this is a binary variable, use iss function.
                    if(all((xregData==0) | (xregData==1))){
                        xregModel <- suppressWarnings(iss(rev(xregData), model=xregModel$model, intermittent=xregModel$intermittent,
                                                          persistence=xregModel$persistence, h=maxLag));
                    }
                    else{
                        xregModel <- suppressWarnings(es(rev(xregData), model=modelType(xregModel), persistence=xregModel$persistence,
                                                         intermittent=xregModel$intermittent, imodel=xregModel$imodel, h=maxLag));
                    }
                }
                else{
                    # If this is a binary variable, use iss function.
                    if(all((xregData==0) | (xregData==1))){
                        xregModel <- suppressWarnings(iss(rev(xregData),model="XXX", h=maxLag,intermittent="l"));
                    }
                    else{
                        xregModel <- suppressWarnings(es(rev(xregData),h=maxLag,intermittent="a"));
                    }
                }
                xregDataNew <- c(rev(xregModel$forecast),xregDataNew);
            }

            if(any(lagsOriginal<0)){
                for(j in 1:lagsLength){
                    xregNew[,chosenColumn+1+j] <- xregDataNew[1:obs-lags[j]+maxLag];
                    colnames(xregNew)[chosenColumn+1+j] <- paste0(xregCurrentName,"Lag",lags[j]);
                }
            }
            if(any(lagsOriginal>0)){
                for(j in 1:leadsLength){
                    xregNew[,chosenColumn+1+lagsLength+j] <- xregDataNew[1:obs+leads[j]+maxLag];
                    colnames(xregNew)[chosenColumn+1+lagsLength+j] <- paste0(xregCurrentName,"Lead",leads[j]);
                }
            }
        }
        if(!silent){
            cat(paste0(rep("\b",4),collapse=""));
            cat(" Done! \n");
        }
    }
    return(xregNew);
}
