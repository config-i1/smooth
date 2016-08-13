coef.es <- function(object,...)
{
    if(any(unlist(gregexpr("C",object$model))==-1)){
    # If this was normal ETS, return values
        return(c(object$persistence,object$initial,object$initial.season));
    }
    else{
    # If we did combinations, we cannot return anything
        message("Combination of models was done, so there are no coefficients to return");
        return(NULL);
    }
}

is.es <- function(x){
    inherits(x, "es");
}

summary.es <- function(object,...){
    return("Not yet implemented");
}

fitted.es <- function(object,...){
    return(object$fitted);
}

plot.es <- function(x,...){
    if(any(unlist(gregexpr("C",x$model))==-1)){
        # If this was normal ETS, return values
        plot(x$states,main="ES states");
    }
    else{
        # If we did combinations, we cannot return anything
        message("Combination of models was done. Sorry, but there is nothing to plot.");
    }
}
