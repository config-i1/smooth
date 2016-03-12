ssoutput <- function(timeelapsed, modelname, persistence=NULL, transition=NULL, measurement=NULL,
                     phi=NULL, ARterms=NULL, MAterms=NULL, const=NULL, A=NULL, B=NULL,
                     n.components=NULL, s2=NULL, hadxreg=FALSE, wentwild=FALSE,
                     CF.type="MSE", CF.objective=NULL, intervals=FALSE,
                     int.type=c("p","s","n","a"), int.w=0.95, ICs,
                     holdout=FALSE, insideintervals=NULL, errormeasures=NULL){
# Function forms the generic output for State-space models.
    cat(paste0("Time elapsed: ",round(as.numeric(timeelapsed,units="secs"),2)," seconds\n"));
    cat(paste0("Model estimated: ",modelname,"\n"));
### Stuff for ETS and GES
    if(!is.null(persistence)){
        cat(paste0("Persistence vector g: ", paste(round(persistence,3),collapse=", "),"\n"));
    }
    if(!is.null(transition)){
        cat("Transition matrix F: \n");
        print(round(transition,3));
    }
    if(!is.null(measurement)){
        cat(paste0("Measurement vector w: ",paste(round(measurement,3),collapse=", "),"\n"));
    }
    if(!is.null(phi)){
        cat(paste0("Damping parameter: ", round(phi,3),"\n"));
    }
### Stuff for ARIMA
    if(all(!is.null(ARterms),any(ARterms!=0))){
        cat("Matrix of AR terms:\n");
        print(round(ARterms,3));
    }
    if(all(!is.null(MAterms),any(MAterms!=0))){
        cat("Matrix of MA terms:\n");
        print(round(MAterms,3));
    }
    if(!is.null(const)){
        cat(paste0("Constant value is: ",round(const,3),"\n"));
    }
### Stuff for CES
    if(!is.null(A)){
        cat(paste0("a0 + ia1: ",round(A,5),"\n"));
    }
    if(!is.null(B)){
        if(is.complex(B)){
            cat(paste0("b0 + ib1: ",round(B,5),"\n"));
        }
        else{
            cat(paste0("b: ",round(B,5),"\n"));
        }
    }

    if(!is.null(n.components)){
        if(n.components==1){
            cat(paste0(n.components," initial state was estimated in the process\n"));
        }
        else{
            cat(paste0(n.components," initial states were estimated in the process\n"));
        }
    }

    if(!is.null(s2)){
        cat(paste0("Residuals standard deviation: ",round(sqrt(s2),3),"\n"));
    }

    if(hadxreg==TRUE){
        cat("Xreg coefficients were estimated");
        if(wentwild==TRUE){
            cat(" in a crazy style\n");
        }
        else{
            cat(" in a normal style\n");
        }
    }

    cat(paste0("Cost function type: ",CF.type))
    if(!is.null(CF.objective)){
        cat(paste0("; Cost function value: ",round(CF.objective,0),"\n"));
    }
    else{
        cat("\n");
    }

    cat("\nInformation criteria:\n");
    print(ICs);
    cat("\n");

    if(intervals==TRUE){
        if(int.type=="p"){
            int.type <- "parametric";
        }
        else if(int.type=="s"){
            int.type <- "semiparametric";
        }
        else if(int.type=="n"){
            int.type <- "nonparametric";
        }
        else if(int.type=="a"){
            int.type <- "asymmetric";
        }
        cat(paste0(int.w*100,"% ",int.type," prediction intervals were constructed\n"));
    }

    if(holdout==TRUE){
        if(intervals==TRUE){
            cat(paste0(round(insideintervals,0), "% of values are in the prediction interval\n"));
        }
        cat("Forecast errors:\n");
        cat(paste(paste0("MPE: ",errormeasures["MPE"]*100,"%"),
                    paste0("MAPE: ",errormeasures["MAPE"]*100,"%"),
                    paste0("SMAPE: ",errormeasures["SMAPE"]*100,"%\n"),sep="; "));
        cat(paste(paste0("MASE: ",errormeasures["MASE"]),
                    paste0("MASALE: ",errormeasures["MASALE"]*100,"%\n"),sep="; "));
    }
}
