depricator <- function(ellipsis, parameter, newName){
    if(!is.null(ellipsis[[parameter]])){
        warning(paste0("The '",parameter,"' parameter is no longer supported. ",
                       "Please, use '",newName,"' parameter instead."), call.=FALSE, immediate.=TRUE);
        ellipsis[[newName]] <- ellipsis[[parameter]];
        ellipsis[[parameter]] <- NULL;
    }
    return(ellipsis);
}
