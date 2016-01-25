sowhat <- function(...){
# Function returns ultimate answer to any question
    if(any(grepl("\\?",unlist(list(...))))){
        message("42");
    }
    message("So what?");
}