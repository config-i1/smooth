depricator <- function(newValue, ellipsis, oldName){
    if(!is.null(ellipsis$data) & oldName=="data"){
        warning("You have provided 'data' parameter. This is deprecated. Please, use 'y' instead.", call.=FALSE);
        return(ellipsis$data);
    }
    else if(!is.null(ellipsis$cfType) & oldName=="cfType"){
        warning("You have provided 'cfType' parameter. This is deprecated. Please, use 'loss' instead.", call.=FALSE);
        return(ellipsis$cfType);
    }
    else if(!is.null(ellipsis$workFast) & oldName=="workFast"){
        warning("You have provided 'workFast' parameter. This is deprecated. Please, use 'fast' instead.", call.=FALSE);
        return(ellipsis$workFast);
    }
    else if(!is.null(ellipsis$lagsMax) & oldName=="lagsMax"){
        warning("You have provided 'lagsMax' parameter. This is deprecated. Please, use 'lags' instead.", call.=FALSE);
        return(ellipsis$lagsMax);
    }
    else if(!is.null(ellipsis$ordersMax) & oldName=="ordersMax"){
        warning("You have provided 'ordersMax' parameter. This is deprecated. Please, use 'orders' instead.", call.=FALSE);
        return(ellipsis$ordersMax);
    }
    else if(!is.null(ellipsis$intervals) & oldName=="intervals"){
        warning("You have provided 'intervals' parameter. This is deprecated. Please, use 'interval' (singular) instead.", call.=FALSE);
        return(ellipsis$intervals);
    }
    else{
        return(newValue);
    }
}
