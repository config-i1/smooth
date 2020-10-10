#' Function returns the ultimate answer to any question
#'
#' You need description? So what?
#'
#' You need details? So what?
#'
#' @param ...  Any number of variables or string with a question.
#' @return It doesn't return any value, only messages. So what?
#' @template ssAuthor
#' @seealso Nowwhat (to be implemented),
#' @references \itemize{
#' \item\href{https://en.wiktionary.org/wiki/so_what}{Sowhat?}
#' \item\href{https://www.youtube.com/watch?v=FJfFZqTlWrQ}{Sowhat?}
#' \item\href{https://en.wikipedia.org/wiki/Douglas_Adams}{42}
#' }
#' @keywords sowhat 42
#' @examples
#'
#' x <- rnorm(10000,0,1);
#' sowhat(x);
#'
#' sowhat("What's the meaning of life?")
#'
#' sowhat("I don't have a girlfriend.")
#'
#' @export sowhat
sowhat <- function(...){
# Function returns ultimate answer to any question
    if(any(grepl("\\?",unlist(list(...))))){
        message("42");
    }
    message("So what?");
}

# Function is needed to ask additional question before the release
release_questions <- function(){
  c("i1: Did you check package with --use-valgrind if C++ code changed?");
}
