graphmaker <- function(actuals,forecast,fitted=NULL,lower=NULL,upper=NULL,int.w=NULL,legend=TRUE){
# Function constructs the universal linear graph for any model
##### Make legend change if the fitted is provided or not!
    if(!is.null(lower) | !is.null(upper)){
        intervals <- TRUE;
        if(is.null(int.w)){
          message("The width of prediction intervals is not provided to graphmaker!");
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
    h <- length(forecast)

# Write down the default values of par
    def.par <- par(no.readonly = TRUE);

# Estimate plot range
    plot.range <- range(min(actuals[!is.na(actuals)],fitted[!is.na(fitted)],
                            forecast[!is.na(forecast)],lower[!is.na(lower)],upper[!is.na(upper)]),
                        max(actuals[!is.na(actuals)],fitted[!is.na(fitted)],
                            forecast[!is.na(forecast)],lower[!is.na(lower)],upper[!is.na(upper)]));

    if(legend==TRUE){
        layout(matrix(c(1,2),2,1),heights=c(0.86,0.14));
        par(mar=c(2,3,2,1));
    }
    else{
        par(mar=c(3,3,2,1));
    }

    plot(actuals,type="l",xlim=range(time(actuals)[1],time(forecast)[h]),
         ylim=plot.range,xlab="", ylab="");
    if(!all(is.na(fitted))){
        lines(fitted,col="purple",lwd=2,lty=2);
    }
    abline(v=deltat(forecast)*(start(forecast)[2]-2)+start(forecast)[1],col="red",lwd=2);

    if(intervals==TRUE){
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
                       legend=c("Series","Fitted values","Point forecast",paste0(int.w*100,"% prediction interval"),"Forecast origin"),
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
                       legend=c("Series","Fitted values","Point forecast",paste0(int.w*100,"% prediction interval"),"Forecast origin"),
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
                       legend=c("Series","Fitted values","Point forecast","Forecast origin"),
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
                       legend=c("Series","Fitted values","Point forecast","Forecast origin"),
                       col=c("black","purple","blue","red"),
                       lwd=c(1,2,2,2),
                       lty=c(1,2,NA,1),
                       pch=c(NA,NA,4,NA),ncol=2);
            }
        }
    }

    par(def.par)
}