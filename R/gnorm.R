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

  lambda <- (1/alpha)^beta
  gnormValues <- (sign(p-0.5)*qgamma(abs(p - 0.5)*2, shape = 1/beta, scale = 1/lambda)^(1/beta) + mu)

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

  lambda <- (1/alpha)^beta
  gnormValues <- qgamma(runif(n), shape = 1/beta, scale = alpha^beta)^(1/beta)*((-1)^rbinom(n, 1, 0.5)) + mu
  return(gnormValues)
}
