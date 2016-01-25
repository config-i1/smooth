nus <- function(formula, data, h=10, holdout=FALSE,
                initial=c("whole","min","quarter","10 obs"),
                weights=c("dynamic","static","manual"),
                persistence=c("adaptive","optimal","optimal.all","manual"),
                filter=c("none","MAE","RMSE","optimal","dynamic"),
                weights.v=NULL, silent=FALSE, legend=TRUE){

#  Function builds regression using non-nuiform smoothing method.
#   formula - the formula of regression.
#   data - the data.frame that contains variables in formula.
#   h - the forecasting horizon.
#   holdout - the sample that should be taken from the data as a holdout.
#   silent - determines whether anykind of output should be given.
#   weigths - how the weights for coefficients should be calculated:
#         dynamic - weight for each variable depend on the change of this variable,
#         static - weigths are always equal to 1/k.
#         manual - weights are specified by the user.
#   persistence - how should be persistence calculated:
#         adaptive - persistence will depend on the distance from eta,
#         optimal - the general persistence is found during optimisation,
#   !!!!  optimal.all - the persistence is found for each of the variables during optimisation.
#   !!!!  manual - set by the user.
#   filter - how should be the filter of errors defined:
#         MAE - as Mean Absolute Error,
#         MSE - as Mean Square Error,
#         none - simply equal to 0 (which implies adaptation on every step),
#         dynamic - filter is based on the error and student distribution.
#   weights.v - the vector of weights defined if the "manual" weigths are chosen.

#  require(alabama);

  if(silent==FALSE){
# Start measuring time
    start.time <- Sys.time();
  }

  weights <- weights[1];
  persistence <- persistence[1];
  filter <- filter[1];
  initial <- initial[1];

  n.vars <- length(all.vars(formula));
  obs <- nrow(data)-h*holdout;

# Matrix of exogeneous variables used in the model construction (without the holdout)
#  used.data <- data[1:obs,all.vars(formula)];
  used.data <- cbind(rep(1,(obs+1)),data[1:(obs+1),all.vars(formula)[2:n.vars]]);
  colnames(used.data) <- c("const",all.vars(formula)[2:n.vars]);

  if(holdout==TRUE){
# Matrix with all the exogeneous variables
    all.data <- cbind(rep(1,(obs+h)),data[,all.vars(formula)[2:n.vars]]);
    colnames(all.data) <- colnames(used.data);
  }
  else{
    all.data <- used.data;
  }

# Make several checks of the provided data
  if(persistence=="optimal.all"){
    message("'optimal.all' for persistence is not implemented yet. Using 'adaptive' instead");
    persistence <- "adaptive";
  }

  if(weights=="manual"){
    if(is.null(weights.v)){
      message("No weights are provided! Please use paramter 'weights.v'. Changing to 'static'.");
      weights <- "static";
    }
    else if(length(colnames(used.data))!=length(weights.v)){
      message("Not all the needed weights are provided! Changing to 'static'.");
      message("HINT: don't forget about the weight of intercept!");
      weights <- "static";
    }
# Normalize the weights. Just in case...
    weights.v[which(weights.v!=0)] <- weights.v[which(weights.v!=0)] / sum(weights.v[which(weights.v!=0)]);
  }

  if(weights!="manual"){
    weights.v <- 1/n.vars;
  }

# a is the matrix of coefficients, where nrow <- T, ncol <- k
  a <- matrix(NA,nrow=(obs+1),ncol=n.vars);
  colnames(a) <- colnames(used.data);
  rownames(a) <- c(0:obs);

  a <- data.frame(a);

  if(initial=="whole"){
    reg.ols <- lm(formula=formula,data=data[1:obs,]);
  }
  else if(initial=="min"){
    reg.ols <- lm(formula=formula,data=data[1:n.vars,]);
  }
  else if(initial=="quarter"){
    reg.ols <- lm(formula=formula,data=data[1:(obs/4),]);
  }
  else if(initial=="10 obs"){
    reg.ols <- lm(formula=formula,data=data[1:10,]);
  }

  a.ols <- coef(reg.ols);
  if(any(is.na(a.ols))){
    message("At least one of the coefficients can not be estimated! Please change the sample size!");
    message("Let's make the bad coefficients equal to zero.")
    a.ols[which(is.na(a.ols))]<-0;
  }
  a[1,] <- a.ols;

# vector of all actual values of y
  y <- data[,all.vars(formula)[1]];

# vector of all future calculated y
  all.y.est <- rep(NA,length(y));
  y.est.ols <- all.y.est;

# The vector of calculated values of y during fitting
  y.est <- rep(NA,obs);

# The matrix of weights
  v <- matrix(NA,nrow=(obs+1),ncol=n.vars);
  colnames(v) <- colnames(used.data);
  v <- data.frame(v);

  v[1,] <- weights.v;

# The matrix of persistence values, NOT including weights
  g <- matrix(NA,nrow=obs,ncol=n.vars);
  colnames(g) <- colnames(used.data);
  g <- data.frame(g);

# The vector of errors
  e <- rep(NA,obs);

# The vector for dynamic filter
  eta.all <- rep(NA,obs);

# Fill in the vector of fitted y using ols for all the data
  for(i in 1:obs){
    y.est.ols[i] <- sum(a.ols*all.data[i,]);
  }
  resids.ols <- y[1:obs] - y.est.ols[1:obs];

  MAE <- mean(abs(diff(y[1:obs])));
  RMSE <- sqrt(mean((diff(y[1:obs]))^2));

  nus.filter <- function(filter){
# Define the filter width
    if(filter=="MAE"){
      eta <- MAE;
    }
    else if(filter=="RMSE"){
      eta <- RMSE;
    }
    else if(filter=="none"){
      eta <- 0;
    }
    else if(filter=="optimal"){
      eta <- MAE;
    }
    else if(filter=="dynamic"){
      eta <- (pnorm(abs(resids.ols[1]/RMSE))-0.5)*2;
    }
    return(list(eta=eta));
  }

  nus.CF <- function(C){
### This function needs to be smaller and more general!
    if(filter=="optimal" & persistence=="optimal"){

      eta <- C[1];
      alpha <- C[2];

      for (i in 1:obs){
        y.est[i] <- sum(a[i,]*used.data[i,]);
        e[i] <- y[i] - y.est[i];

        if(abs(e[i]) > eta){
          g[i,] <- rep(alpha,n.vars);
        }
        else{
          g[i,] <- rep(0,n.vars);
        }

        if(weights=="dynamic"){
          v[i+1,] <- ((used.data[i+1,]-used.data[i,])/abs(used.data[i+1,]))/sum(abs(used.data[i+1,]-used.data[i,])/abs(used.data[i+1,]));
          v[i+1,1] <- 1 - sum(v[i+1,]);
        }
        else{
          v[i+1,] <- v[i,];
        }
        data.vec <- used.data[i,];
        data.vec[which(data.vec==0)] <- Inf;

        a[(i+1),] <- a[i,] + v[i,]*g[i,]*e[i]/data.vec;
      }
    }
    else if(filter!="optimal" & persistence=="optimal"){
      alpha <- C[1];

      for (i in 1:obs){
        y.est[i] <- sum(a[i,]*used.data[i,]);
        e[i] <- y[i] - y.est[i];

        if(filter=="dynamical"){
          print("Dynamic filter and optimal persistence are not implemented yet. Implementing the absence of filter.")
        }

        if(abs(e[i]) > eta){
          g[i,] <- rep(alpha,n.vars);
        }
        else{
          g[i,] <- rep(0,n.vars);
        }

        if(weights=="dynamic"){
#          v[i+1,] <- (used.data[i+1,]-used.data[i,])/sum(abs(used.data[i+1,]-used.data[i,]));
          v[i+1,] <- ((used.data[i+1,]-used.data[i,])/abs(used.data[i+1,]))/sum(abs(used.data[i+1,]-used.data[i,])/abs(used.data[i+1,]));
          v[i+1,1] <- 1 - sum(v[i+1,]);
        }
        else{
          v[i+1,] <- v[i,];
        }
        data.vec <- used.data[i,];
        data.vec[which(data.vec==0)] <- Inf;

        a[(i+1),] <- a[i,] + v[i,]*g[i,]*e[i]/data.vec;
      }
    }
    else if(filter=="optimal" & persistence!="optimal"){
      eta <- C[1];

      for (i in 1:obs){
        y.est[i] <- sum(a[i,]*used.data[i,]);
        e[i] <- y[i] - y.est[i];

        if(abs(e[i]) > eta){
          g[i,] <- rep(abs((abs(e[i])-eta)/e[i]),n.vars);
        }
        else{
          g[i,] <- rep(0,n.vars);
        }

        if(weights=="dynamic"){
#          v[i+1,] <- (used.data[i+1,]-used.data[i,])/sum(abs(used.data[i+1,]-used.data[i,]));
          v[i+1,] <- ((used.data[i+1,]-used.data[i,])/abs(used.data[i+1,]))/sum(abs(used.data[i+1,]-used.data[i,])/abs(used.data[i+1,]));
          v[i+1,1] <- 1 - sum(v[i+1,]);
        }
        else{
          v[i+1,] <- v[i,];
        }
        data.vec <- used.data[i,];
        data.vec[which(data.vec==0)] <- Inf;

        a[(i+1),] <- a[i,] + v[i,]*g[i,]*e[i]/data.vec;
      }
    }

    CF <- sum((y[1:obs] - y.est[1:obs])^2);

    return(list(y.est=y.est, a=a, eta=eta, v=v, g=g, CF=CF, e=e));
  }

# Function constructs the regression using NUS and found parameters
  nus.construct <- function(filter, weights, persistence, eta){
    if(filter=="dynamic"){
      eta.all[1] <- eta;
      eta <- 0;
    }
    for (i in 1:obs){
      y.est[i] <- sum(a[i,]*used.data[i,]);
      e[i] <- y[i] - y.est[i];

      if(filter=="dynamic"){
        eta.all[i+1] <- (pnorm(abs(e[i]/sqrt((eta.all[i]^2*i+e[i]^2)/(i+1))))-0.5)*2;
        g[i,] <- rep(eta.all[i],n.vars);
      }
      else{
        if(persistence=="manual"){
          g[i,] <- 0.01;
        }
        else{
          if(abs(e[i]) > eta){
            g[i,] <- rep(abs((abs(e[i])-eta)/e[i]),n.vars);
          }
          else{
            g[i,] <- rep(0,n.vars);
          }
        }
      }

      if(weights=="dynamic"){
#        v[i+1,] <- (used.data[i+1,]-used.data[i,])/sum(abs(used.data[i+1,]-used.data[i,]));
        v.current <- (used.data[i+1,]-used.data[i,]) / used.data[i+1,];
        v.current[which(is.nan(unlist(v.current)) | is.infinite(unlist(v.current)))] <- 0;
        v[i+1,] <- v.current / sum(abs(v.current));
        v[i+1,1] <- 1 - sum(v[i+1,]);
      }
      else{
        v[i+1,] <- v[i,];
      }
      data.vec <- used.data[i,];
      data.vec[which(data.vec==0)] <- Inf;

      a[(i+1),] <- a[i,] + v[i,]*g[i,]*e[i]/data.vec;

    }

    return(list(y.est=y.est, a=a, g=g, v=v, eta=eta, e=e));
  }

  eta <- nus.filter(filter=filter)$eta;

  if (filter=="optimal" | persistence=="optimal"){
# If either of variables needs to be estimated, use the optim function.
    if(filter=="optimal" & persistence=="optimal"){
      C <- c(eta,0.5);
    }
    else if(filter=="optimal" & persistence!="optimal"){
      C <- c(eta);
    }
    else if(filter!="optimal" & persistence=="optimal"){
      C <- c(0.5);
    }

# nus.CF returns several values, we need only one of them, that's why we need this
    additional.CF <- function(C){
      CF <- nus.CF(C)$CF;
      return(CF);
    }

    res <- optim(C, additional.CF, lower=rep(0,length(C)), upper=rep(Inf,length(C)), method="L-BFGS-B");
    res.param <- nus.CF(res$par);
    eta <- res.param$eta;
    v <- res.param$v;
    g <- res.param$g;
    y.est <- res.param$y.est;
    a <- res.param$a;
    e <- res.param$e;
  }
  else{
    res.param <- nus.construct(filter=filter, weights=weights, persistence=persistence, eta=eta);
    eta <- res.param$eta;
    v <- res.param$v;
    g <- res.param$g;
    y.est <- res.param$y.est;
    a <- res.param$a;
    e <- res.param$e;
  }

  last.a <- as.matrix(a[obs+1,])
  all.y.est[1:obs] <- y.est;

if(silent==FALSE){
  if(holdout==TRUE){
    for (i in (obs+1):(obs+h)){
      all.y.est[i] <- sum(last.a*all.data[i,]);
      y.est.ols[i] <- sum(a.ols*all.data[i,]);
    }

    print(paste0("NUS MASE: ",MASE(y[(obs+1):(obs+h)],all.y.est[(obs+1):(obs+h)],MAE)));
    print(paste0("OLS MASE: ",MASE(y[(obs+1):(obs+h)],y.est.ols[(obs+1):(obs+h)],MAE)));
  }

  print(paste0("Time elapsed: ",round(as.numeric(Sys.time() - start.time,units="secs"),2)," seconds"));
  par(mfcol=c(1,1))
    plot(y,type="l",ylim=range(min(y,all.y.est),max(y,all.y.est)));
#    lines(y[1:obs]);
#    lines((all.y.est+eta),col="#22aa22");
#    lines((all.y.est-eta),col="#22aa22");
    lines(all.y.est[1:obs],col="purple",lwd=2,lty=2);
#    lines(all.y.est,col="red");
    if(holdout==TRUE){
        lines(c((obs+1):(obs+h)),all.y.est[(obs+1):(obs+h)],col="blue",lwd=2);
    }
    lines(y.est.ols,col="darkgreen",lty=2);
    abline(v=obs,col="red",lwd=2)
    if(legend==TRUE){
        legend(x="bottomleft",
               legend=c("Series","Fitted values","Point forecast","OLS forecast","Forecast origin"),
               col=c("black","purple","blue","darkgreen","red"),
               lwd=c(1,2,2,1,2),
               lty=c(1,2,1,2,1));
    }
}

  return(list(y.est=all.y.est, last.a=as.matrix(a[obs+1,]), a=a, v=v, g=g, eta=eta, residuals=e));
}
