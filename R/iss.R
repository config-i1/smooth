iss <- function(data, intermittent=c("simple","croston","tsb"),
                h=10, imodel=NULL, ipersistence=NULL){
# Function estimates and returns mean and variance of probability for intermittent State-Space model based on the chosen method
    intermittent <- substring(intermittent[1],1,1);
    y <- data;
    obs <- length(y);
    ot <- (y!=0)*1;
    iprob <- mean(ot);
    obs.ones <- sum(ot);
    obs.zeroes <- obs - obs.ones;
# Sizes of demand
    yot <- matrix(y[y!=0],obs.ones,1);

    if(intermittent=="s"){
        return(list(fitted=iprob,forecast=iprob,variance=iprob*(1-iprob)));
    }
    else if(intermittent=="c"){
# Define the matrix of states
        ivt <- matrix(rep(iprob,obs+1),obs+1,1);
# Define the matrix of actuals as intervals between demands
        zeroes <- c(0,which(y!=0),obs+1);
        zeroes <- diff(zeroes)-1;
# Number of intervals in Croston
        obs.int <- length(zeroes);
        iyt <- matrix(zeroes,obs.int,1);
        if(is.null(imodel)){
            if(any(iyt==0)){
                return(es(iyt,"ANN",intervals=T,int.w=0.95,silent=TRUE,h=h,ipersistence=ipersistence));
            }
            else{
                return(es(iyt,"MNN",intervals=T,int.w=0.95,silent=TRUE,h=h,ipersistence=ipersistence));
            }
        }
        else{
            return(es(iyt,model=imodel,intervals=T,int.w=0.95,silent=TRUE,h=h,ipersistence=ipersistence));
        }
    }
    else if(intermittent=="t"){
        warning("Sorry, TSB is not implemented yet.",call.=FALSE);
        return(NULL);
        ivt <- matrix(rep(iprob,obs+1),obs+1,1);
        iyt <- matrix(y,obs,1);
    }
}
