#' @importFrom stats pgamma qgamma rbinom
dgnorm <- function(x, mu = 0, alpha = 1, beta = 1,
                   log = FALSE) {
    # A failsafe for NaN / NAs of alpha / beta
    if(any(is.nan(alpha))){
        alpha[is.nan(alpha)] <- 0
    }
    if(any(is.na(alpha))){
        alpha[is.na(alpha)] <- 0
    }
    if(any(alpha<0)){
        alpha[alpha<0] <- 0
    }
    if(any(is.nan(beta))){
        beta[is.nan(beta)] <- 0
    }
    if(any(is.na(beta))){
        beta[is.na(beta)] <- 0
    }
    gnormValues <- (exp(-(abs(x-mu)/ alpha)^beta)* beta/(2*alpha*gamma(1/beta)))
    if(log){
        gnormValues[] <- log(gnormValues)
    }

    return(gnormValues)
}

pgnorm <- function(q, mu = 0, alpha = 1, beta = 1,
                   lower.tail = TRUE, log.p = FALSE) {
  # A failsafe for NaN / NAs of alpha / beta
  if(any(is.nan(alpha))){
    alpha[is.nan(alpha)] <- 0
  }
  if(any(is.na(alpha))){
    alpha[is.na(alpha)] <- 0
  }
  if(any(alpha<0)){
    alpha[alpha<0] <- 0
  }
  if(any(is.nan(beta))){
    beta[is.nan(beta)] <- 0
  }
  if(any(is.na(beta))){
    beta[is.na(beta)] <- 0
  }

  # Failsafe mechanism. If beta is too high, switch to uniform
  if(beta>100){
    return(punif(q, min=mu-alpha, mu+alpha, lower.tail=lower.tail, log.p=log.p))
  }
  else{
    p <- (1/2 + sign(q - mu[])* pgamma(abs(q - mu)^beta, shape = 1/beta, rate = (1/alpha)^beta)/2)
    if (lower.tail) {
      if (!log.p) {
        return(p)
      } else {
        return(log(p))
      }
    } else if (!lower.tail) {
      if (!log.p) {
        return(1 - p)
      } else {
        return(log(1 - p))
      }
    }
  }
}

qgnorm <- function(p, mu = 0, alpha = 1, beta = 1,
                   lower.tail = TRUE, log.p = FALSE) {
  # A failsafe for NaN / NAs of alpha / beta
  if(any(is.nan(alpha))){
    alpha[is.nan(alpha)] <- 0
  }
  if(any(is.na(alpha))){
    alpha[is.na(alpha)] <- 0
  }
  if(any(alpha<0)){
    alpha[alpha<0] <- 0
  }
  if(any(is.nan(beta))){
    beta[is.nan(beta)] <- 0
  }
  if(any(is.na(beta))){
    beta[is.na(beta)] <- 0
  }

  if (lower.tail & !log.p) {
    p <- p
  } else if (lower.tail & log.p) {
    p <- exp(p)
  } else if (!lower.tail & !log.p) {
    p <- 1 - p
  } else {
    p <- log(1 - p)
  }

  # Failsafe mechanism. If beta is too high, switch to uniform
  if(beta>100){
    gnormValues <- qunif(p, min=mu-alpha, mu+alpha);
  }
  # If it is not too bad, scale the scale parameter
  else if((1/alpha)^beta<1e-300){
    lambdaScale <- ceiling(alpha) / 10
    lambda <- (alpha/lambdaScale)^(beta)
    gnormValues <- (sign(p-0.5)*(qgamma(abs(p - 0.5)*2, shape = 1/beta, scale = lambda))^(1/beta) + mu)*lambdaScale
  }
  else{
    lambda <- alpha^(-beta)
    gnormValues <- (sign(p-0.5)*qgamma(abs(p - 0.5)*2, shape = 1/beta, scale = lambda)^(1/beta) + mu)
  }

  return(gnormValues)
}

rgnorm <- function(n, mu = 0, alpha = 1, beta = 1) {
  # A failsafe for NaN / NAs of alpha / beta
  if(any(is.nan(alpha))){
    alpha[is.nan(alpha)] <- 0
  }
  if(any(is.na(alpha))){
    alpha[is.na(alpha)] <- 0
  }
  if(any(alpha<0)){
    alpha[alpha<0] <- 0
  }
  if(any(is.nan(beta))){
    beta[is.nan(beta)] <- 0
  }
  if(any(is.na(beta))){
    beta[is.na(beta)] <- 0
  }

  gnormValues <- qgnorm(runif(n), mu=mu, alpha=alpha, beta=beta)
  # lambda <- (1/alpha)^beta
  # gnormValues <- qgamma(runif(n), shape = 1/beta, scale = alpha^beta)^(1/beta)*((-1)^rbinom(n, 1, 0.5)) + mu
  return(gnormValues)
}
