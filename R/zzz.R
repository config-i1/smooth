.onAttach <- function(libname, pkgname) {
    startUpMessage <- paste0("This is package \"smooth\", v",packageVersion(pkgname));
    randomNumber <- ceiling(runif(1,0,100));
    if(randomNumber==1){
        startUpMessage <- paste0(startUpMessage,"\nBy the way, have you already tried adam() function from smooth?");
    }
    else if(randomNumber==2){
        startUpMessage <- paste0(startUpMessage,"\nIf you want to know more about the smooth package and forecasting, ",
                                 "you can visit my website: https://forecasting.svetunkov.ru/");
    }
    else if(randomNumber==3){
        startUpMessage <- paste0(startUpMessage,"\nHave you tried adam() yet? If you want to know more about this function, ",
                                 "you can visit my online textbook: https://www.openforecast.org/adam/");
    }
    else if(randomNumber==4){
        startUpMessage <- paste0(startUpMessage,"\nAny thought or suggestions about the package? ",
                                 "Have you found a bug? File an issue on github: https://github.com/config-i1/smooth/issues");
    }
    startUpMessage <- paste0(startUpMessage,"\n");
    packageStartupMessage(startUpMessage);
}

.onUnload <- function (libpath) {
  library.dynam.unload("smooth", libpath)
}
