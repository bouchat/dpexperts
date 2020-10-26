//
//  main.cpp
//  elicitedDPsim
//
//  Created by Alex Tahk on 7/25/16.
//  Copyright © 2016 Sarah Bouchat. All rights reserved.
//

#include <vector>
#include <iostream>
#include <cmath>
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>

extern "C" {
#define MATHLIB_STANDALONE
#include <Rmath.h>
}

namespace R
{
    inline double norm_rand()
    {
      return ::norm_rand();
    }
    inline double exp_rand()
  {
    return ::exp_rand();
  }
    inline double unif_rand()
  {
    return ::unif_rand();
  }
    inline double rgamma(const double shape, const double scale)
  {
    return ::rgamma(shape, scale);
  }
}

#include "eliciteddp.hpp"

int main(int argc, const char * argv[]) {
    uword n = 10;    // number of experts
    uword m = 6;    // number of questions per expert
    uword k = 4;     // number of covariates
    bool discrete = false;
    double concentration = 1; // concentration parameter
    uvec cluster = rdpcluster(n, concentration);
    mat beta(k,n);
    beta.imbue([&](){ return norm_rand(); } );
    mat x(m,k);
    x.imbue([&](){ return norm_rand(); } );
    mat y_star(m,n);
    y_star.imbue([&](){ return norm_rand(); } );
    for (uword j = 0; j < y_star.n_cols; ++j)
        y_star.col(j) += x * beta.col(cluster[j]);
    mat y(size(y_star));
    for (uword i = 0; i < y_star.n_rows; ++i)
        for (uword j = 0; j < y_star.n_cols; ++j)
            y(i,j) = y_star(i,j);

    mat prior_lambda(x.n_cols, x.n_cols, fill::eye);
    vec prior_mu(x.n_cols, fill::zeros);

    long max_iter = 10000;
    long thin = 10;
    long burn_in = 1000;
    double prior_alpha = 1;
    double prior_beta = 1;

    umat assignment_samples(y.n_cols, max_iter / thin);
    mat mu_samples(prior_mu.n_rows, max_iter / thin);
    cube lambda_samples(prior_lambda.n_rows, prior_lambda.n_cols, max_iter / thin);

    eliciteddp(y, x, discrete,
               max_iter, thin, burn_in,
               prior_lambda, prior_mu,
               prior_alpha, prior_beta,
               concentration, assignment_samples, mu_samples, lambda_samples);

    cout << assignment_samples << endl;
    cout << mu_samples << endl;
    cout << lambda_samples << endl;

    return 0;
}
