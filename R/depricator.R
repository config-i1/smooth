depricator <- function(ellipsis, parameter, functionName){
    if(!is.null(ellipsis[[parameter]])){
        warning(paste0("The '",parameter,"' parameter is no longer supported in the ",functionName,"() function. ",
                       "Please, use 'adam()' function instead."), call.=FALSE, immediate.=TRUE);
        ellipsis[[parameter]] <- NULL;
    }
    return(ellipsis);
}
