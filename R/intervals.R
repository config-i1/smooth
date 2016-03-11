intervals <- function(x,ev=median(x),int.w=0.95,int.type=c("a","p","s","n")){
# Function constructs intervals based on the provided random variable.
# If x is a matrix, then it is assumed that each column has a variable that needs an interval.

##### Import here code from ges for parametric intervals in the case of additive ETS models + change input.

    int.type <- int.type[1]
    hsmN <- gamma(0.75)*pi^(-0.5)*2^(-0.75);

    if(is.matrix(x) | is.data.frame(x)){
        n.var <- ncol(x);
        if(length(ev)!=n.var & length(ev)!=1){
            stop("Provided expected value doesn't correspond to the dimension of x.", call.=FALSE);
        }
        else if(length(ev)==1){
            ev <- rep(ev,n.var);
        }
        upper <- rep(NA,n.var);
        lower <- rep(NA,n.var);

        for(i in 1:n.var){
            upper[i] <- ev[i] + qnorm((1+int.w)/2,0,1)/hsmN^2 * Re(hm(x[,i],ev[i]))^2;
            lower[i] <- ev[i] + qnorm((1-int.w)/2,0,1)/hsmN^2 * Im(hm(x[,i],ev[i]))^2;
        }
    }
    else if(is.numeric(x) & length(x)>1 & !is.array(x)){
        if(length(ev)>1){
            stop("Provided expected value doesn't correspond to the dimension of x.", call.=FALSE);
        }
        upper <- ev + qnorm((1+int.w)/2,0,1)/hsmN^2 * Re(hm(x,ev))^2;
        lower <- ev + qnorm((1-int.w)/2,0,1)/hsmN^2 * Im(hm(x,ev))^2;
    }
    else{
        stop("The provided data is not either vector or matrix. Can't do anything with it!", call.=FALSE);
    }

    return(list(upper=upper,lower=lower));
}
