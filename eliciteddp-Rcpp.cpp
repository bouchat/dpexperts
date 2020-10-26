#define ARMA_DONT_USE_WRAPPER
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins("cpp11")]]

#include "eliciteddp.hpp"

using namespace Rcpp;

// [[Rcpp::export]]
uvec rdp_cluster(const long n, const double concentration)
{
    vector<size_t> counts(n,0);
    uvec assignments(n);
    size_t max = 0;
    for (long i = 0; i < n; ++i)
    {
        double total_prob = concentration + i;
        total_prob *= R::unif_rand();
        total_prob -= concentration;
        if ((max == 0) || (total_prob < 0.0))
        {
            counts[max]++;
            assignments[i] = max+1;
            max++;
        }
        else
        {
            for (long j = 0; j < max; ++j)
            {
                total_prob -= counts[j];
                if (total_prob < 0.0)
                {
                    counts[j]++;
                    assignments[i] = j+1;
                    break;
                }
            }
        }
    }
    return assignments;
}

// [[Rcpp::export]]
List elicited(mat y, mat x, bool discrete,
		      long max_iter, long thin, long burn_in,
		      mat prior_lambda, vec prior_mu,
		      double prior_alpha, double prior_beta,
		      double concentration) {

  umat assignment_samples(y.n_cols, max_iter / thin);
  mat mu_samples(prior_mu.n_rows, max_iter / thin);
  cube lambda_samples(prior_lambda.n_rows, prior_lambda.n_cols, max_iter / thin);

  Rcout << 1 << endl;
  
  eliciteddp(y, x, discrete,
	     max_iter, thin, burn_in,
	     prior_lambda, prior_mu,
	     prior_alpha, prior_beta,
	     concentration, assignment_samples, mu_samples, lambda_samples);

    Rcout << 2 << endl;

    return List::create(Named("assignment") = wrap(assignment_samples),
			      Named("mu") = wrap(mu_samples),
			      Named("lambda") = wrap(lambda_samples));
}


// [[Rcpp::export]]
arma::vec getEigenValues(arma::mat M) {
    return arma::eig_sym(M);
}


// [[Rcpp::export]]
NumericVector rtnorm_Rcpp(long n, double mean, double sd, double bound, bool above)
{
  NumericVector x(n);
  for (NumericVector::iterator i = x.begin(); i != x.end(); ++i)
    *i = rtnorm(mean, sd, bound, above);
  return x;
}
// end truncated normal 
