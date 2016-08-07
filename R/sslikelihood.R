utils::globalVariables(c("h","multisteps","obs.ot","CF.type","CF","ot","intermittent"));

likelihoodFunction <- function(C){
# This block is needed in order to make R CMD to shut up about "no visible binding..."
    if(any(intermittent==c("n","p"))){
        if(CF.type=="TFL" | CF.type=="aTFL"){
            return(- obs.ot/2 *((h^multisteps)*log(2*pi*exp(1)) + CF(C)));
        }
        else{
            return(- obs.ot/2 *(log(2*pi*exp(1)) + log(CF(C))));
        }
    }
    else{
        if(CF.type=="TFL" | CF.type=="aTFL"){
            return(sum(log(pt[ot==1]))*(h^multisteps)
                   + sum(log(1-pt[ot==0]))*(h^multisteps)
                   - obs.ot/2 * ((h^multisteps)*log(2*pi*exp(1)) + CF(C)));
        }
        else{
            return(sum(log(pt[ot==1])) + sum(log(1-pt[ot==0]))
                   - obs.ot/2 *(log(2*pi*exp(1)) + log(CF(C))));
        }
    }
}

## Function calculates ICs
ICFunction <- function(n.param=n.param,C,Etype=Etype){
# Information criteria are calculated with the constant part "log(2*pi*exp(1)*h+log(obs))*obs".
# And it is based on the mean of the sum squared residuals either than sum.
# Hyndman likelihood is: llikelihood <- obs*log(obs*CF.objective)

    llikelihood <- likelihoodFunction(C);

    AIC.coef <- 2*n.param*h^multisteps - 2*llikelihood;
# max here is needed in order to take into account cases with higher number of parameters than observations
    AICc.coef <- AIC.coef + 2 * n.param*h^multisteps * (n.param + 1) / max(obs.ot - n.param - 1,0);
    BIC.coef <- log(obs.ot)*n.param*h^multisteps - 2*llikelihood;

    ICs <- c(AIC.coef, AICc.coef, BIC.coef);
    names(ICs) <- c("AIC", "AICc", "BIC");

    return(list(llikelihood=llikelihood,ICs=ICs));
}
