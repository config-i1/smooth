.onAttach <- function(libname, pkgname) {
    packageStartupMessage(paste0("This is package 'smooth', v",packageVersion(pkgname)));
}
