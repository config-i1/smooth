#' Linear graph construction function
#'
#' The function makes a standard linear graph using at least actuals and
#' forecasts.
#'
#' Function uses the provided data to construct a linear graph. It is strongly
#' advised to use \code{ts} function to define the start of each of the
#' vectors. Otherwise the data may be plotted in a wrong way.
#'
#' @param actuals The vector of actual series.
#' @param forecast The vector of forecasts. Should be ts object that start at
#' the end of \code{fitted} values.
#' @param fitted The vector of fitted values.
#' @param lower The vector of lower bound values of a prediction interval.
#' Should be ts object that start at the end of \code{fitted} values.
#' @param upper The vector of upper bound values of a prediction interval.
#' Should be ts object that start at the end of \code{fitted} values.
#' @param level The width of the prediction interval.
#' @param legend If \code{TRUE}, the legend is drawn.
#' @param cumulative If \code{TRUE}, then the forecast is treated as
#' cumulative and value per period is plotted.
#' @param vline Whether to draw the vertical line, splitting the in-sample
#' and the holdout sample.
#' @param ... Other parameters passed to \code{plot()} function.
#' @return Function does not return anything.
#' @author Ivan Svetunkov
#' @seealso \code{\link[stats]{ts}}
#' @keywords plots linear graph
#' @examples
#'
#' x <- rnorm(100,0,1)
#' values <- es(x,model="ANN",silent=TRUE,intervals=TRUE,level=0.95)
#'
#' graphmaker(x,values$forecast,values$fitted)
#' graphmaker(x,values$forecast,values$fitted,legend=FALSE)
#' graphmaker(x,values$forecast,values$fitted,values$lower,values$upper,level=0.95)
#' graphmaker(x,values$forecast,values$fitted,values$lower,values$upper,level=0.95,legend=FALSE)
#'
#' actuals <- c(1:10)
#' forecast <- ts(c(11:15),start=end(actuals)[1]+end(actuals)[2]*deltat(actuals),
#'                frequency=frequency(actuals))
#' graphmaker(actuals,forecast)
#'
#' @export graphmaker
graphmaker <- function(actuals, forecast, fitted=NULL, lower=NULL, upper=NULL,
                       level=NULL, legend=TRUE, cumulative=FALSE, vline=TRUE, ...){
# Function constructs the universal linear graph for any model

    ellipsis <- list(...);

##### Make legend change if the fitted is provided or not!
    if(!is.null(lower) | !is.null(upper)){
        intervals <- TRUE;
        if(is.null(level)){
            message("The width of prediction intervals is not provided to graphmaker! Assuming 95%.");
            level <- 0.95;
        }
    }
    else{
        lower <- NA;
        upper <- NA;
        intervals <- FALSE;
    }
    if(is.null(fitted)){
        fitted <- NA;
    }
    h <- length(forecast);

    if(all(is.na(forecast))){
        h <- 0;
    }

    if(cumulative){
        pointForecastLabel <- "Point forecast per period";
    }
    else{
        pointForecastLabel <- "Point forecast";
    }

# Write down the default values of par
    parDefault <- par(no.readonly = TRUE);

    if(legend==TRUE){
        layout(matrix(c(1,2),2,1),heights=c(0.86,0.14));
        if(is.null(ellipsis$main)){
            parMar <- c(2,3,2,1);
        }
        else{
            parMar <- c(2,3,3,1);
        }
    }
    else{
        if(is.null(ellipsis$main)){
            parMar <- c(3,3,2,1);
        }
        else{
            parMar <- c(3,3,3,1);
        }
    }

# Estimate plot range
    if(is.null(ellipsis$ylim)){
        ellipsis$ylim <- range(min(actuals[!is.na(actuals)],fitted[!is.na(fitted)],
                                   forecast[!is.na(forecast)],lower[!is.na(lower)],upper[!is.na(upper)]),
                               max(actuals[!is.na(actuals)],fitted[!is.na(fitted)],
                                   forecast[!is.na(forecast)],lower[!is.na(lower)],upper[!is.na(upper)]));
    }

    if(is.null(ellipsis$xlim)){
        ellipsis$xlim <- range(time(actuals)[1],time(forecast)[max(h,1)]);
    }

    if(!is.null(ellipsis$main) & cumulative){
        ellipsis$main <- paste0(ellipsis$main,", cumulative forecast");
    }

    if(is.null(ellipsis$type)){
        ellipsis$type <- "l";
    }

    if(is.null(ellipsis$xlab)){
        ellipsis$xlab <- "";
    }
    else{
        parMar[1] <- parMar[1] + 1;
    }

    if(is.null(ellipsis$ylab)){
        ellipsis$ylab <- "";
    }
    else{
        parMar[2] <- parMar[2] + 1;
    }

    ellipsis$x <- actuals;

    par(mar=parMar);
    do.call(plot, ellipsis);
    # plot(actuals, type="l", xlim=range(time(actuals)[1],time(forecast)[max(h,1)]),
         # ylim=plot.range, xlab="", ylab="", main=main);
    if(any(!is.na(fitted))){
        lines(fitted,col="purple",lwd=2,lty=2);
    }
    if(vline){
        abline(v=deltat(forecast)*(start(forecast)[2]-2)+start(forecast)[1],col="red",lwd=2);
    }

    if(intervals){
        if(h>1){
            lines(lower,col="darkgrey",lwd=3,lty=2);
            lines(upper,col="darkgrey",lwd=3,lty=2);
# Draw the nice areas between the borders
            polygon(c(seq(deltat(upper)*(start(upper)[2]-1)+start(upper)[1],deltat(upper)*(end(upper)[2]-1)+end(upper)[1],deltat(upper)),
                    rev(seq(deltat(lower)*(start(lower)[2]-1)+start(lower)[1],deltat(lower)*(end(lower)[2]-1)+end(lower)[1],deltat(lower)))),
                    c(as.vector(upper), rev(as.vector(lower))), col = "lightgray", border=NA, density=10);

            lines(forecast,col="blue",lwd=2);
#If legend is needed do the stuff...
            if(legend==TRUE){
                par(cex=0.75,mar=rep(0.1,4),bty="n",xaxt="n",yaxt="n")
                plot(0,0,col="white")
                legend(x="bottom",
                       legend=c("Series","Fitted values",pointForecastLabel,paste0(level*100,"% prediction interval"),"Forecast origin"),
                       col=c("black","purple","blue","darkgrey","red"),
                       lwd=c(1,2,2,3,2),
                       lty=c(1,2,1,2,1),ncol=3);
            }
        }
        else{
            points(lower,col="darkgrey",lwd=3,pch=4);
            points(upper,col="darkgrey",lwd=3,pch=4);
            points(forecast,col="blue",lwd=2,pch=4);

            if(legend==TRUE){
                par(cex=0.75,mar=rep(0.1,4),bty="n",xaxt="n",yaxt="n")
                plot(0,0,col="white")
                legend(x="bottom",
                       legend=c("Series","Fitted values",pointForecastLabel,paste0(level*100,"% prediction interval"),"Forecast origin"),
                       col=c("black","purple","blue","darkgrey","red"),
                       lwd=c(1,2,2,3,2),
                       lty=c(1,2,NA,NA,1),
                       pch=c(NA,NA,4,4,NA),ncol=3)
            }
        }
    }
    else{
        if(h>1){
            lines(forecast,col="blue",lwd=2);

            if(legend==TRUE){
                par(cex=0.75,mar=rep(0.1,4),bty="n",xaxt="n",yaxt="n")
                plot(0,0,col="white")
                legend(x="bottom",
                       legend=c("Series","Fitted values",pointForecastLabel,"Forecast origin"),
                       col=c("black","purple","blue","red"),
                       lwd=c(1,2,2,2),
                       lty=c(1,2,1,1),ncol=2);
            }
        }
        else{
            points(forecast,col="blue",lwd=2,pch=4);
            if(legend==TRUE){
                par(cex=0.75,mar=rep(0.1,4),bty="n",xaxt="n",yaxt="n")
                plot(0,0,col="white")
                legend(x="bottom",
                       legend=c("Series","Fitted values",pointForecastLabel,"Forecast origin"),
                       col=c("black","purple","blue","red"),
                       lwd=c(1,2,2,2),
                       lty=c(1,2,NA,1),
                       pch=c(NA,NA,4,NA),ncol=2);
            }
        }
    }

    par(parDefault);
}
