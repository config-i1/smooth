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
                    bool const &backcast)
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


    // Loop for the backcasting
    unsigned int nIterations = 1;
    if (backcast)
    {
        nIterations = 1;
    }

    // Loop for the backcast
    for (unsigned int j = 1; j <= nIterations; j = j + 1)
    {

        // Refine the head (in order for it to make sense)
        // This is only needed for ETS(*,Z,*) models, with trend.
        if(!backcast || nArima==0){
            
            for (int i = 0; i < lagsModelMax; i = i + 1)
            {
                matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));
                profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                                                     matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);
            }
        }
        ////// Run forward
        // Loop for the model construction
        for (int i = lagsModelMax; i < obs + lagsModelMax; i = i + 1)
        {
            

            /* # Measurement equation and the error term */
            vecYfit(i - lagsModelMax) = adamWvalue(profilesRecent(indexLookupTable.col(i)),
                                                   matrixWt.row(i - lagsModelMax), E, T, S,
                                                   nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);


            // If this is zero (intermittent), then set error to zero
            if (vectorOt(i - lagsModelMax) == 0)
            {
                vecErrors(i - lagsModelMax) = 0;
            }
            else
            {
                // We need this multiplication for cases, when occurrence is fractional
                vecYfit(i - lagsModelMax) = vectorOt(i - lagsModelMax) * vecYfit(i - lagsModelMax);
                vecErrors(i - lagsModelMax) = errorf(vectorYt(i - lagsModelMax), vecYfit(i - lagsModelMax), E);
            }

            /* # Transition equation */
            profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                                                 matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                                                      adamGvalue(profilesRecent(indexLookupTable.col(i)), matrixF, matrixWt.row(i - lagsModelMax), E, T, S,
                                                                 nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant, vectorG, vecErrors(i - lagsModelMax));

            matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));
        }

        ////// Backwards run
        if (backcast && j < (nIterations))
        {
            // Change the specific element in the state vector to negative
            if (T == 'A')
            {
                profilesRecent(1) = -profilesRecent(1);
            }
            else if (T == 'M')
            {
                profilesRecent(1) = 1 / profilesRecent(1);
            }

            for (int i = obs + lagsModelMax - 1; i >= lagsModelMax; i = i - 1)
            {
                /* # Measurement equation and the error term */
                vecYfit(i - lagsModelMax) = adamWvalue(profilesRecent(indexLookupTable.col(i)),
                                                       matrixWt.row(i - lagsModelMax), E, T, S,
                                                       nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant);

                // If this is zero (intermittent), then set error to zero
                if (vectorOt(i - lagsModelMax) == 0)
                {
                    vecErrors(i - lagsModelMax) = 0;
                }
                else
                {
                    // Need to add this line to match the R version
                    vecYfit(i - lagsModelMax) = vectorOt(i - lagsModelMax) * vecYfit(i - lagsModelMax);
                    vecErrors(i - lagsModelMax) = errorf(vectorYt(i - lagsModelMax), vecYfit(i - lagsModelMax), E);
                }

                /* # Transition equation */
                profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                                                     matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant) +
                                                          adamGvalue(profilesRecent(indexLookupTable.col(i)), matrixF,
                                                                     matrixWt.row(i - lagsModelMax), E, T, S,
                                                                     nETS, nNonSeasonal, nSeasonal, nArima, nXreg, nComponents, constant,
                                                                     vectorG, vecErrors(i - lagsModelMax));
            }

            // Fill in the head of the series
            for (int i = lagsModelMax - 1; i >= 0; i = i - 1)
            {
                profilesRecent(indexLookupTable.col(i)) = adamFvalue(profilesRecent(indexLookupTable.col(i)),
                                                                     matrixF, E, T, S, nETS, nNonSeasonal, nSeasonal, nArima, nComponents, constant);

                matrixVt.col(i) = profilesRecent(indexLookupTable.col(i));
            }

            // Change back the specific element in the state vector
            if (T == 'A')
            {
                profilesRecent(1) = -profilesRecent(1);
                // Write down correct initials
                // Fil's update on 04/03/2025
                matrixVt.col(0) = profilesRecent(indexLookupTable.col(0));
            }
            else if (T == 'M')
            {
                profilesRecent(1) = 1 / profilesRecent(1);
                // Write down correct initials
                // Fil's update on 04/03/2025
                matrixVt.col(0) = profilesRecent(indexLookupTable.col(0));
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
        py::arg("backcast"));
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
}
