#' @param initial Can be either character or a list, or a vector of initial states.
#' If it is character, then it can be \code{"backcasting"}, meaning that the initials of
#' dynamic part of the model are produced using backcasting procedure (advised
#' for data with high frequency), or \code{"optimal"}, meaning that all initial
#' states are optimised, or \code{"two-stage"}, meaning that optimisation is done
#' after the backcasting, refining the states. In case of backcasting, the parameters of the
#' explanatory variables are optimised. Alternatively, you can set \code{initial="complete"}
#' backcasting, which means that all states (including explanatory variables) are initialised
#' via backcasting.
