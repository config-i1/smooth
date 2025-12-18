#include <iostream>
#include <cmath>

#include <carma>
#include <armadillo>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <adamGeneral.h>

namespace py = pybind11;

// # Fitter for univariate models
// Convert from Rcpp::List to pybind11::dict

py::dict adamFitter(arma::mat &matrixVt,
                    arma::mat const &matrixWt,
                    arma::mat &matrixF,
                    arma::vec const &vectorG,
                    arma::uvec &lags,
                    arma::umat const &indexLookupTable,
                    arma::mat profilesRecent,
                    char const &E, char const &T, char const &S,
                    unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                    unsigned int const &nArima,
                    unsigned int const &nXreg,
                    bool const &constant,
                    arma::vec const &vectorYt,
                    arma::vec const &vectorOt,
                    bool const &backcast,
                    unsigned int const &nIterations,
                    bool const &refineHead,
                    bool const &adamETS)
{
    /* # matrixVt should have a length of obs + lagsModelMax.
     * # matrixWt is a matrix with nrows = obs
     * # vecG should be a vector
     * # lags is a vector of lags
     */

    int obs = vectorYt.n_rows;
    unsigned int nETS = nNonSeasonal + nSeasonal;
    int nComponents = matrixVt.n_rows;
    int lagsModelMax = max(lags);

    // Fitted values and the residuals
    arma::vec vecYfit(obs, arma::fill::zeros);
    arma::vec vecErrors(obs, arma::fill::zeros);

    // These are objects used in backcasting.
    // Needed for some experiments.
    arma::mat &matrixFInv = matrixF;
    arma::vec const &vectorGInv = vectorG;

    // Loop for the backcast
    for (unsigned int j=1; j<=nIterations; j=j+1) {

        // Refine the head (in order for it to make sense)
        // This is only needed for ETS(*,Z,*) models, with trend.
        // This is not needed for lagsMax=1, because there is nothing to fill in
        if(refineHead && (T!='N')){
            // Record the initial profile to the first column
            matrixVt.col(0) = profilesRecent(indexLookupTable.col(0));
            if(lagsModelMax>1){
                // Update the head, but only for the trend component
                for (int i=1; i<lagsModelMax; i=i+1) {
                    profilesRecent(indexLookupTable.col(i).rows(0,1)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                   matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant).rows(0,1);
                    matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));
                }
            }
        }
        ////// Run forward
        // Loop for the model construction
        for (int i = lagsModelMax; i < obs + lagsModelMax; i = i + 1)
        {
            matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));

            /* # Measurement equation and the error term */
            vecYfit(i-lagsModelMax) = adamWvalue(profilesRecent(indexLookupTable.col(i)),
                    matrixWt.row(i-lagsModelMax), E, T, S,
                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

            // If this is zero (intermittent), then set error to zero
            if(vectorOt(i-lagsModelMax)==0){
                vecErrors(i-lagsModelMax) = 0;
            }
            else
            {
                vecErrors(i - lagsModelMax) = errorf(vectorYt(i - lagsModelMax), vecYfit(i - lagsModelMax), E);
            }

            /* # Transition equation */
            profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                                                 matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                                                      adamGvalue(profilesRecent(indexLookupTable.col(i)), matrixF, matrixWt.row(i - lagsModelMax), E, T, S,
                                                                 nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant, vectorG, vecErrors(i - lagsModelMax), vecYfit(i - lagsModelMax), adamETS);

            // If ot is fractional, amend the fitted value
            if (vectorOt(i - lagsModelMax) != 0 && vectorOt(i - lagsModelMax) != 1)
            {
                // We need this multiplication for cases, when occurrence is fractional
                vecYfit(i - lagsModelMax) = vectorOt(i - lagsModelMax) * vecYfit(i - lagsModelMax);
            }
        }

        ////// Backwards run
        if(backcast && j<(nIterations)){
            // Change the specific element in the state vector to negative
            if(T=='A'){
                profilesRecent(1) = -profilesRecent(1);
            }
            else if(T=='M'){
                profilesRecent(1) = 1/profilesRecent(1);
            }

            for (int i=obs+lagsModelMax-1; i>=lagsModelMax; i=i-1) {
                /* # Measurement equation and the error term */
                vecYfit(i-lagsModelMax) = adamWvalue(profilesRecent(indexLookupTable.col(i)),
                        matrixWt.row(i-lagsModelMax), E, T, S,
                        nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

                // If this is zero (intermittent), then set error to zero
                if(vectorOt(i-lagsModelMax)==0){
                    vecErrors(i-lagsModelMax) = 0;
                }
                else{
                    // We need this multiplication for cases, when occurrence is fractional
                    vecYfit(i-lagsModelMax) = vectorOt(i-lagsModelMax)*vecYfit(i-lagsModelMax);
                    vecErrors(i-lagsModelMax) = errorf(vectorYt(i-lagsModelMax), vecYfit(i-lagsModelMax), E);
                }

                /* # Transition equation */
                profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                                                     matrixFInv, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                                                          adamGvalue(profilesRecent(indexLookupTable.col(i)), matrixFInv,
                                                                     matrixWt.row(i - lagsModelMax), E, T, S,
                                                                     nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                                                                     vectorGInv, vecErrors(i - lagsModelMax), vecYfit(i - lagsModelMax), adamETS);
            }

            // Fill in the head of the series.
            if(refineHead){
                for (int i = lagsModelMax - 1; i >= 0; i = i - 1)
                {
                    profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                                                         matrixFInv, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);

                    // matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));
                }
            }

            // Change back the specific element in the state vector
            if(T=='A'){
                profilesRecent(1) = -profilesRecent(1);
                // Write down correct initials
                // This is needed in case the profileRecent has changed in previous lines
                // matrixVt.col(0) = profilesRecent(indexLookupTable.col(0));
            }
            else if(T=='M'){
                profilesRecent(1) = 1/profilesRecent(1);
                // Write down correct initials
                // This is needed in case the profileRecent has changed in previous lines
                // matrixVt.col(0) = profilesRecent(indexLookupTable.col(0));
            }
        }
    }

    // return List::create(Named("matVt") = matrixVt, Named("yFitted") = vecYfit,
    //                     Named("errors") = vecErrors, Named("profile") = profilesRecent);

    // Create a Python dictionary to return results
    py::dict result;
    result["matVt"] = matrixVt;
    result["yFitted"] = vecYfit;
    result["errors"] = vecErrors;
    result["profile"] = profilesRecent;

    return result;
}

/* # Function produces the point forecasts for the specified model */
arma::vec adamForecaster(arma::mat const &matrixWt,
                         arma::mat const &matrixF,
                         arma::uvec lags,
                         arma::umat const &indexLookupTable,
                         arma::mat profilesRecent,
                         char const &E,
                         char const &T,
                         char const &S,
                         unsigned int const &nNonSeasonal,
                         unsigned int const &nSeasonal,
                         unsigned int const &nArima,
                         unsigned int const &nXreg,
                         bool const &constant, unsigned int const &horizon)
{
    // unsigned int lagslength = lags.n_rows;
    unsigned int nETS = nNonSeasonal + nSeasonal;
    unsigned int nComponents = indexLookupTable.n_rows;

    arma::vec vecYfor(horizon, arma::fill::zeros);

    /* # Fill in the new xt matrix using F. Do the forecasts. */
    for (unsigned int i = 0; i < horizon; i = i + 1)
    {
        vecYfor.row(i) = adamWvalue(profilesRecent(indexLookupTable.col(i)), matrixWt.row(i), E, T, S,
                                    nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

        profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                                             matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
    }

    // return List::create(Named("matVt") = matrixVtnew, Named("yForecast") = vecYfor);
    return vecYfor;
}

// # Simulator for generating multiple forecast paths
py::dict adamSimulator(arma::cube &arrayVt, arma::mat const &matrixErrors, arma::mat const &matrixOt,
                       arma::cube const &arrayF, arma::mat const &matrixWt, arma::mat const &matrixG,
                       char const &E, char const &T, char const &S, arma::uvec &lags,
                       arma::umat const &indexLookupTable, arma::mat profilesRecent,
                       unsigned int const &nNonSeasonal, unsigned int const &nSeasonal,
                       unsigned int const &nArima, unsigned int const &nXreg, bool const &constant,
                       bool const &adamETS) {

    unsigned int obs = matrixErrors.n_rows;
    unsigned int nSeries = matrixErrors.n_cols;

    int lagsModelMax = max(lags);
    unsigned int nETS = nNonSeasonal + nSeasonal;
    int nComponents = lags.n_rows;
    int obsAll = obs + lagsModelMax;
    arma::mat profilesRecentOriginal = profilesRecent;

    double yFitted;

    arma::mat matrixVt(nComponents, obsAll, arma::fill::zeros);
    arma::mat matrixF(arrayF.n_rows, arrayF.n_cols, arma::fill::zeros);

    arma::mat matY(obs, nSeries);

    for(unsigned int i=0; i<nSeries; i=i+1){
        matrixVt = arrayVt.slice(i);
        matrixF = arrayF.slice(i);
        profilesRecent = profilesRecentOriginal;
        for(int j=lagsModelMax; j<obsAll; j=j+1) {
            /* # Measurement equation and the error term */
            yFitted = adamWvalue(profilesRecent(indexLookupTable.col(j-lagsModelMax)),
                                                         matrixWt.row(j-lagsModelMax), E, T, S,
                                                         nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                                         nComponents, constant);
            matY(j-lagsModelMax,i) = matrixOt(j-lagsModelMax,i) *
                                             (yFitted +
                                             adamRvalue(profilesRecent(indexLookupTable.col(j-lagsModelMax)),
                                                        matrixWt.row(j-lagsModelMax), E, T, S,
                                                        nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant) *
                                                            matrixErrors(j-lagsModelMax,i));

            /* # Transition equation */
            profilesRecent(indexLookupTable.col(j-lagsModelMax)) =
                                                (adamFvalue(profilesRecent(indexLookupTable.col(j-lagsModelMax)),
                                                            matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima,
                                                            nComponents, constant) +
                                                 adamGvalue(profilesRecent(indexLookupTable.col(j-lagsModelMax)),
                                                            matrixF, matrixWt.row(j-lagsModelMax),
                                                            E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nXreg,
                                                            nComponents, constant, matrixG.col(i),
                                                            matrixErrors(j-lagsModelMax,i), yFitted, adamETS));

            matrixVt.col(j) = profilesRecent(indexLookupTable.col(j-lagsModelMax));
        }
        arrayVt.slice(i) = matrixVt;
    }

    // Create a Python dictionary to return results
    py::dict result;
    result["arrayVt"] = arrayVt;
    result["matrixYt"] = matY;

    return result;
}

PYBIND11_MODULE(_adam_general, m)
{
    m.doc() = "Adam code"; // module docstring
    m.def(
        "adam_fitter",
        &adamFitter,
        "fits the adam model",
        py::arg("matrixVt"),
        py::arg("matrixWt"),
        py::arg("matrixF"),
        py::arg("vectorG"),
        py::arg("lags"),
        py::arg("indexLookupTable"),
        py::arg("profilesRecent"),
        py::arg("E"),
        py::arg("T"),
        py::arg("S"),
        py::arg("nNonSeasonal"),
        py::arg("nSeasonal"),
        py::arg("nArima"),
        py::arg("nXreg"),
        py::arg("constant"),
        py::arg("vectorYt"),
        py::arg("vectorOt"),
        py::arg("backcast"),
        py::arg("nIterations"),
        py::arg("refineHead"),
        py::arg("adamETS"));
    m.def("adam_forecaster", &adamForecaster, "forecasts the adam model",
          py::arg("matrixWt"),
          py::arg("matrixF"),
          py::arg("lags"),
          py::arg("indexLookupTable"),
          py::arg("profilesRecent"),
          py::arg("E"),
          py::arg("T"),
          py::arg("S"),
          py::arg("nNonSeasonal"),
          py::arg("nSeasonal"),
          py::arg("nArima"),
          py::arg("nXreg"),
          py::arg("constant"),
          py::arg("horizon"));
    m.def("adam_simulator", &adamSimulator, "simulates multiple forecast paths",
          py::arg("arrayVt"),
          py::arg("matrixErrors"),
          py::arg("matrixOt"),
          py::arg("arrayF"),
          py::arg("matrixWt"),
          py::arg("matrixG"),
          py::arg("E"),
          py::arg("T"),
          py::arg("S"),
          py::arg("lags"),
          py::arg("indexLookupTable"),
          py::arg("profilesRecent"),
          py::arg("nNonSeasonal"),
          py::arg("nSeasonal"),
          py::arg("nArima"),
          py::arg("nXreg"),
          py::arg("constant"),
          py::arg("adamETS"));
}
