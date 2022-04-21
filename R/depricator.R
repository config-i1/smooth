depricator <- function(ellipsis, parameter, functionName, newName){
    if(!is.null(ellipsis[[parameter]])){
        warning(paste0("The '",parameter,"' parameter is no longer supported in the ",functionName,"() function. ",
                       "Please, use '",newName,"' parameter instead."), call.=FALSE, immediate.=TRUE);
        ellipsis[[newName]] <- ellipsis[[parameter]];
        ellipsis[[parameter]] <- NULL;
    }
    return(ellipsis);
}
