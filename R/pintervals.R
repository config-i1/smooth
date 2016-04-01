pintervals <- function(errors, ev=median(errors), int.w=0.95, int.type=c("a","p","s","n"), df=NULL,
                      measurement=NULL, transition=NULL, persistence=NULL, s2=NULL, modellags=NULL,
                      y.for=rep(0,ncol(errors)), iprob=1){
# Function constructs intervals based on the provided random variable.
# If errors is a matrix, then it is assumed that each column has a variable that needs an interval.
# based on errors the horison is estimated as ncol(errors)

    matrixpower <- function(A,n){
        if(n==0){
            return(diag(nrow(A)));
        }
        else if(n==1){
            return(A);
        }
        else if(n>1){
            return(A %*% matrixpower(A, n-1));
        }
    }

    int.type <- int.type[1]
    hsmN <- gamma(0.75)*pi^(-0.5)*2^(-0.75);

    if(all(int.type!=c("a","p","s","n"))){
        stop(paste0("What do you mean by 'int.type=",int.type,"'? I can't work with this!"),call.=FALSE);
    }

    if(int.type=="p"){
        if(any(is.null(measurement),is.null(transition),is.null(persistence),is.null(s2),is.null(modellags))){
            stop("measurement, transition, persistence, s2 and modellags need to be provided in order to construct parametric intervals!",call.=FALSE);
        }

        if(any(!is.matrix(measurement),!is.matrix(transition),!is.matrix(persistence))){
            stop("measurement, transition and persistence must me matrices. Can't do stuff with what you've provided.",call.=FALSE);
        }
    }

#Function allows to estimate the coefficients of the simple quantile regression. Used in intervals construction.
quantfunc <- function(A){
    ee <- ye - (A[1] + A[2]*xe + A[3]*xe^2);
    return((1-quant)*sum(abs(ee[which(ee<0)]))+quant*sum(abs(ee[which(ee>=0)])));
}

# If degrees of freedom are provided, use Student's distribution. Otherwise stick with normal.
    if(is.null(df)){
        upperquant <- qnorm((1+int.w)/2,0,1);
        lowerquant <- qnorm((1-int.w)/2,0,1);
    }
    else{
        upperquant <- qt((1+int.w)/2,df=df);
        lowerquant <- qt((1-int.w)/2,df=df);
    }

    if(is.matrix(errors) | is.data.frame(errors)){
        n.var <- ncol(errors);
        obs <- nrow(errors);
        if(length(ev)!=n.var & length(ev)!=1){
            stop("Provided expected value doesn't correspond to the dimension of errors.", call.=FALSE);
        }
        else if(length(ev)==1){
            ev <- rep(ev,n.var);
        }

        upper <- rep(NA,n.var);
        lower <- rep(NA,n.var);

##### Asymmetric intervals using HM
        if(int.type=="a"){
            for(i in 1:n.var){
                upper[i] <- ev[i] + upperquant / hsmN^2 * Re(hm(errors[,i],ev[i]))^2;
                lower[i] <- ev[i] + lowerquant / hsmN^2 * Im(hm(errors[,i],ev[i]))^2;
            }
        }

##### Semiparametric intervals using the variance of errors
        else if(int.type=="s"){
            errors <- errors - matrix(ev,nrow=obs,ncol=n.var,byrow=T);
            upper <- ev + upperquant * sqrt(colMeans(errors^2,na.rm=T));
            lower <- ev + lowerquant * sqrt(colMeans(errors^2,na.rm=T));
        }

##### Nonparametric intervals using Taylor and Bunn, 1999
        else if(int.type=="n"){
            ye <- errors;
            xe <- matrix(c(1:n.var),byrow=TRUE,ncol=n.var,nrow=nrow(errors));
            xe <- xe[!is.na(ye)];
            ye <- ye[!is.na(ye)];

            A <- rep(1,3);
            quant <- (1+int.w)/2;
            A <- nlminb(A,quantfunc)$par;
            upper <- A[1] + A[2]*c(1:n.var) + A[3]*c(1:n.var)^2;

            A <- rep(1,3);
            quant <- (1-int.w)/2;
            A <- nlminb(A,quantfunc)$par;
            lower <- A[1] + A[2]*c(1:n.var) + A[3]*c(1:n.var)^2;
        }

##### Parametric intervals from GES
        else if(int.type=="p"){
            s2i <- iprob*(1-iprob);
            s2 <- s2 * iprob;

            n.components <- nrow(transition);
            maxlag <- max(modellags);
            h <- n.var;

# Array of variance of states
            mat.var.states <- array(0,c(n.components,n.components,h+maxlag));
            mat.var.states[,,1:maxlag] <- persistence %*% t(persistence) * s2;
            mat.var.states.lagged <- as.matrix(mat.var.states[,,1]);

# New transition and measurement for the internal use
            transitionnew <- matrix(0,n.components,n.components);
            measurementnew <- matrix(0,1,n.components);

# selectionmat is needed for the correct selection of lagged variables in the array
# newelements are needed for the correct fill in of all the previous matrices
            selectionmat <- transitionnew;
            newelements <- rep(FALSE,n.components);

# Define chunks, which correspond to the lags with h being the final one
            chuncksofhorizon <- c(1,unique(modellags),h);
            chuncksofhorizon <- sort(chuncksofhorizon);
            chuncksofhorizon <- chuncksofhorizon[chuncksofhorizon<=h];
            chuncksofhorizon <- unique(chuncksofhorizon);

# Length of the vector, excluding the h at the end
            chunkslength <- length(chuncksofhorizon) - 1;

# Vector of final variances
            vec.var <- rep(NA,h);
            newelements <- modellags<=(chuncksofhorizon[1]);
            measurementnew[,newelements] <- measurement[,newelements];
            vec.var[1:min(h,maxlag)] <- s2 + s2i * (y.for[1])^2;

            for(j in 1:chunkslength){
                selectionmat[modellags==chuncksofhorizon[j],] <- chuncksofhorizon[j];
                selectionmat[,modellags==chuncksofhorizon[j]] <- chuncksofhorizon[j];

                newelements <- modellags<=(chuncksofhorizon[j]+1);
                transitionnew[newelements,newelements] <- transition[newelements,newelements];
                measurementnew[,newelements] <- measurement[,newelements];

                for(i in (chuncksofhorizon[j]+1):chuncksofhorizon[j+1]){
                    selectionmat[modellags>chuncksofhorizon[j],] <- i;
                    selectionmat[,modellags>chuncksofhorizon[j]] <- i;

                    mat.var.states.lagged[newelements,newelements] <- mat.var.states[cbind(rep(c(1:n.components),each=n.components),
                                                              rep(c(1:n.components),n.components),
                                                              i - c(selectionmat))];

                    mat.var.states[,,i] <- transitionnew %*% mat.var.states.lagged %*% t(transitionnew) + persistence %*% t(persistence) * s2;
                    vec.var[i] <- measurementnew %*% mat.var.states.lagged %*% t(measurementnew) + s2 +
                                  s2i * (y.for[i])^2;
                }
            }

            upper <- ev + upperquant * sqrt(vec.var);
            lower <- ev + lowerquant * sqrt(vec.var);
        }
    }
    else if(is.numeric(errors) & length(errors)>1 & !is.array(errors)){
        if(length(ev)>1){
            stop("Provided expected value doesn't correspond to the dimension of errors.", call.=FALSE);
        }

        if(int.type=="a"){
            upper <- ev + upperquant / hsmN^2 * Re(hm(errors,ev))^2;
            lower <- ev + lowerquant / hsmN^2 * Im(hm(errors,ev))^2;
        }
        else if(any(int.type==c("s","p"))){
            s2i <- iprob*(1-iprob);
            newelements <- modellags<=1;
            measurement <- measurement[,newelements];
            s2i <- s2i * (y.for[1])^2;
            upper <- ev + upperquant * sqrt(mean((errors-ev)^2,na.rm=T) * iprob + s2i);
            lower <- ev + lowerquant * sqrt(mean((errors-ev)^2,na.rm=T) * iprob + s2i);
        }
        else if(int.type=="n"){
            upper <- quantile(errors,(1+int.w)/2);
            lower <- quantile(errors,(1-int.w)/2);
        }
    }
    else{
        stop("The provided data is not either vector or matrix. Can't do anything with it!", call.=FALSE);
    }

    return(list(upper=upper,lower=lower));
}
