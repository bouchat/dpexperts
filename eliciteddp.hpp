#define KNOWN_VARIANCE

#define HALF_LOG_2_PI 0.91893853320467266954
#define LOG_2 0.69314718055994528623
#define LOG_PI_OVER_FOUR 0.28618247146235004097
// #define solve(X) (X).i()

using namespace std;
using namespace arma;
// using namespace MCMCpack;
// using namespace mvtnorm;
// using namespace MASS;

// truncated normal
double rtnorm(const double lower_bound)
{
    if (lower_bound < 0.45)
      {
        // rejection sample from normal
        // - sample from normal
        // - accept if above lower bound
        while (true)
          {
            double y = R::norm_rand();
            if (y >= lower_bound)
                return y;
          }
      }
    else
      {
        // tail rejection sample using exponential
        // - sample x from Exponential(lower_bound)
        // - accept y = x + lower_bound with prob = exp(- 0.5 * x * x)
        while (true)
          {
            double x = R::exp_rand() / lower_bound;
            if (R::exp_rand() > 0.5 * x * x)
                return x + lower_bound;
          }
      }
}


// mean:  mean of normal distribution (before truncation)
// sd:    standard deviation of normal distribution (before truncation)
// bound: truncate distribution at bound
// above: if true, sample above bound; if not, sample below bound
double rtnorm(const double mean,
              const double sd,
              const double bound,
              const bool above)
{
    if (above)
        return mean + sd * rtnorm( (bound - mean) / sd );
    else
        return mean - sd * rtnorm( (mean - bound) / sd );
}

mat rwishart(const double df, const mat& S)
{
    // Dimension of returned wishart
    auto m = S.n_rows;

    // Z composition:
    // sqrt chisqs on diagonal
    // random normals below diagonal
    // misc above diagonal
    mat Z(m,m);

    // Fill the diagonal
    for(auto i = 0; i < m; i++)
        Z(i,i) = sqrt(R::rgamma( 0.5*(df-i), 2.0 ));

    // Fill the lower matrix with random guesses
    for(auto j = 0; j < m; j++)
        for(auto i = j+1; i < m; i++)
            Z(i,j) = R::norm_rand();

    // Lower triangle * chol decomp
    mat C = trimatl(Z).t() * chol(S);

    // Return random wishart
    return C.t()*C;
}

vec mvrnorm(const vec& mean, const mat& sigma)
{
    vec X(size(mean));
    X.imbue( norm_rand );
    return mean + chol(sigma, "lower") * X;
}

void r_normal_wishart(vec& draw_mu, mat& Lambda,
                      const vec& nw_mu_0, const mat& nw_sigma_0, const double& nw_kappa_0, const double& nw_nu_0)
{
    Lambda = rwishart(nw_nu_0, nw_sigma_0);
    mat wish = Lambda.i() / nw_kappa_0;
//    cout << Lambda << endl << Lambda.i() << endl << nw_sigma_0 << endl << nw_sigma_0.i() << endl << wish << endl;
    draw_mu = mvrnorm(nw_mu_0, wish);
}

double lmgamma(const uword k, const double x)
{
    double result = static_cast<double>(k * (k - 1)) * LOG_PI_OVER_FOUR;
    for (uword j = 1; j <= k; ++j)
        result += lgamma(x + 0.5 * (1.0 - j));
    return result;
}

double ldwish(const mat& X, const double n, const mat& V)
{
    // det(X)^((n-p-1)/2) * e^(-trace(V.i() * X)/2) / (2 ^ ((n*p)/2) det(V)^(n/2) * mult_gamma(p, n/2) )
    double p = V.n_rows;
    double log_det_X, log_det_V, sign;
    log_det(log_det_X, sign, X);
    log_det(log_det_V, sign, V);
    return
    0.5 * (n - p - 1) * log_det_X
    -0.5 * trace(V.i() * X)
    -0.5 * n * p * LOG_2
    -0.5 * n * log_det_V
    -lmgamma(p, 0.5 * n);
}

double ldmvnorm(const vec& proposal_prior_mu, const vec& nw_mu_0, const mat& sigma)
{
    double val, sign;
    log_det(val, sign, sigma); // val = log(det(sigma))
//    double x = dot(proposal_prior_mu - nw_mu_0, solve(sigma, proposal_prior_mu - nw_mu_0));
    return
    -double(proposal_prior_mu.n_elem) * HALF_LOG_2_PI
    -0.5 * val
    -0.5 * dot(proposal_prior_mu - nw_mu_0, solve(sigma, proposal_prior_mu - nw_mu_0));
}

double log_d_normal_wishart(const vec& proposal_prior_mu, const mat& proposal_prior_lambda,
                            const vec& nw_mu_0, const mat& nw_sigma_0, const double nw_kappa_0, const double nw_nu_0)
{
    return
    ldmvnorm(proposal_prior_mu, nw_mu_0, proposal_prior_lambda.i()/nw_kappa_0)
    + ldwish(proposal_prior_lambda, nw_nu_0, nw_sigma_0);
}

class cluster
{
public:
    cluster(const mat& prior_lambda_in,
            const vec& prior_mu_in,
            const double prior_alpha_in,
            const double prior_beta_in)
    :prior_lambda(prior_lambda_in)
    ,prior_lambda_mu(prior_lambda_in * prior_mu_in)
    ,prior_alpha(prior_alpha_in)
    ,prior_beta(prior_beta_in)
    ,lambda(size(prior_lambda_in), fill::zeros)
    ,lambda_mu(size(prior_mu_in), fill::zeros)
    ,alpha(0.0)
    ,sum_yty(0.0)
    ,n(0)
    {
    }

    void add(const vec& y, const mat& x)
    {
      sum_yty += dot(y,y);
      lambda += x.t() * x;
      lambda_mu += x.t() * y;
      alpha += 0.5 * y.n_elem;
      n++;
    }

    void remove(const vec& y, const mat& x)
    {
      n--;
      alpha -= 0.5 * y.n_elem;
      lambda_mu -= x.t() * y;
      lambda -= x.t() * x;
      sum_yty -= dot(y,y);
    }

    void set_probability(const vec& y, const mat& x)
    {
      probability = n * marginal_likelihood(y, x);
    }

    void set_empty_probability(const vec& y, const mat& x, const double concentration)
    {
      probability = concentration * marginal_likelihood(y, x);
    }

    double get_probability() const
    {
      return probability;
    }

    bool is_empty() const
    {
      return n == 0;
    }

    uword get_id() const
    {
      return id;
    }

    void set_id(const uword new_id)
    {
      id = new_id;
    }

    double log_proposal_likelihood_ratio(const double alpha_p, const double beta_p,
                                         const mat& lambda_p, const vec& lambda_mu_p,
                                         const double alpha_c, const double beta_c,
                                         const mat& lambda_c, const vec& lambda_mu_c)
    {
      return log_marginal_likelihood_from_changes(alpha_p,
                                                  beta_p,
                                                  lambda_p,
                                                  lambda_mu_p,
                                                  n,
                                                  sum_yty,
                                                  lambda,
                                                  lambda_mu)
      - log_marginal_likelihood_from_changes(alpha_c,
                                             beta_c,
                                             lambda_c,
                                             lambda_mu_c,
                                             n,
                                             sum_yty,
                                             lambda,
                                             lambda_mu);
    }

    void change_prior(const double alpha_p, const double beta_p,
                      const mat& lambda_p, const vec& lambda_mu_p,
                      const double alpha_c, const double beta_c,
                      const mat& lambda_c, const vec& lambda_mu_c)
    {
      prior_alpha = alpha_p;
      prior_beta = beta_p;
      prior_lambda = lambda_p;
      prior_lambda_mu = lambda_mu_p;
    }

    void add_to_nw_guess(vec& sum_beta, mat& sum_beta_beta_t, unsigned long& n) const
    {
        double stddev;
        vec beta_hat = sample_coefficients(stddev);//solve(lambda, lambda_mu);
      sum_beta += beta_hat;
      sum_beta_beta_t += beta_hat * beta_hat.t();
      n += 1;
    }

    vec sample_coefficients(double& stddev) const
    {
      mat lambda_n = prior_lambda + lambda;
      vec lambda_mu_n = prior_lambda_mu + lambda_mu;
      vec mu_n = solve(lambda, lambda_mu_n);

#ifdef KNOWN_VARIANCE
      // For known variance:
        stddev = 1.0;
      return mvrnorm(mu_n, lambda_n.i());
#else
      // For unknown variance:
      double beta = prior_beta
      + 0.5 * (sum_yty
               + dot(prior_lambda_mu, solve(prior_lambda, prior_lambda_mu))
               - dot(lambda_mu_n, mu_n));
      double variance = 1.0 / R::rgamma(alpha, 1.0 / beta);
        stddev = sqrt(variance);
      return mvrnorm(mu_n, variance * lambda_n.i());
#endif
    }

private:
    inline double marginal_likelihood(const vec& y, const mat& x) const
    {
      mat lambda_0 = prior_lambda + lambda;
      vec lambda_mu_0 = prior_lambda_mu + lambda_mu;
      double alpha_0 = prior_alpha + 0.5 * alpha;
      double beta_0 = prior_beta
      + 0.5 * (sum_yty
               + dot(prior_lambda_mu, solve(prior_lambda, prior_lambda_mu))
               - dot(lambda_mu_0, solve(lambda_0, lambda_mu_0)));
      return exp(log_marginal_likelihood_from_changes(alpha_0,
                                                      beta_0,
                                                      lambda_0,
                                                      lambda_mu_0,
                                                      y.n_elem,
                                                      dot(y,y),
                                                      x.t() * x,
                                                      x.t() * y));
    }

    static double log_marginal_likelihood_from_changes(const double alpha_0,
                                                       const double beta_0,
                                                       const mat& lambda_0,
                                                       const vec& lambda_mu_0,
                                                       const double delta_n,
                                                       const double delta_yty,
                                                       const mat& delta_lambda,
                                                       const vec& delta_lambda_mu)
    {
      mat lambda_n = lambda_0 + delta_lambda;
      vec lambda_mu_n = lambda_mu_0 + delta_lambda_mu;
      double alpha_n = alpha_0 + 0.5 * delta_n;
      double beta_n = beta_0 + 0.5 *(delta_yty
                                     + dot(lambda_mu_0, solve(lambda_0, lambda_mu_0))
                                     - dot(lambda_mu_n, solve(lambda_n, lambda_mu_n)));
      return log_marginal_likelihood(alpha_0, beta_0, lambda_0, lambda_mu_0,
                                     alpha_n, beta_n, lambda_n, lambda_mu_n);
    }

#ifdef KNOWN_VARIANCE
    // For known variance
    static double log_marginal_likelihood(const double alpha_0,
                                          const double beta_0,
                                          const mat& lambda_0,
                                          const vec& lambda_mu_0,
                                          const double alpha_n,
                                          const double beta_n,
                                          const mat& lambda_n,
                                          const vec& lambda_mu_n)
    {
      vec mu = solve(lambda_0, lambda_mu_0);
      vec mu_n = solve(lambda_n, lambda_mu_n);
      // tlml is twice the log of the marginal likelihood
      // this calculation drops the term -log(2*pi)*y.n_elem
      // that will be the same for each cluster.
      double tlml = 0.0;
      double val, sign;
      log_det(val, sign, lambda_0);
      tlml += 0.5 * val;
      log_det(val, sign, lambda_n);
      tlml -= 0.5 * val;
      tlml += beta_0 - beta_n;
      return tlml;
    }
#else
    // For unknown variance
    static double log_marginal_likelihood(const double alpha_0,
                                          const double beta_0,
                                          const mat& lambda_0,
                                          const vec& lambda_mu_0,
                                          const double alpha_n,
                                          const double beta_n,
                                          const mat& lambda_n,
                                          const vec& lambda_mu_n)
    {
      vec mu = solve(lambda_0, lambda_mu_0);
      vec mu_n = solve(lambda_n, lambda_mu_n);
      // this calculation drops the term -0.5*log(2*pi)*y.n_elem
      // that will be the same for each cluster.
      double lml = 0.0;
      double val, sign;
      log_det(val, sign, lambda_0);
      lml += 0.5 * val;
      log_det(val, sign, lambda_n);
      lml -= 0.5 * val;
      lml += alpha_0 * log(beta_0) - alpha_n * log(beta_n);
      lml += lgamma(alpha_n) - lgamma(alpha_0);
      return lml;
    }
#endif

    double probability;
    mat prior_lambda;
    vec prior_lambda_mu;
    double prior_alpha;
    double prior_beta;
    mat lambda;
    vec lambda_mu;
    double alpha;
    double sum_yty;
    size_t n;
    uword id;
};

void eliciteddp(mat y_obs, mat x, bool discrete,
                long max_iter, long thin, long burn_in,
                mat prior_lambda, vec prior_mu,
                double prior_alpha, double prior_beta,
                double concentration,
                umat& assignment_samples, mat& mu_samples, cube& lambda_samples) {
    mat y = y_obs;
    if (discrete)
        y -= 0.5;
    vector<cluster*> assignment(y.n_cols, NULL);
    cluster empty_cluster(prior_lambda, prior_mu, prior_alpha, prior_beta);
    vector<cluster> allocated_clusters(y.n_cols, empty_cluster);
    vector<cluster*> clusters;
    clusters.reserve(y.n_cols);
    vector<cluster*> empty_clusters;
    empty_clusters.reserve(y.n_cols);

    for (size_t i = 0; i < allocated_clusters.size(); ++i)
      {
        allocated_clusters[i].set_id(allocated_clusters.size()-i);
        empty_clusters.push_back( &allocated_clusters[i] );
      }

    double nw_nu_0 = prior_lambda.n_cols;
    double nw_kappa_0 = nw_nu_0; //1.0;
    vec nw_mu_0(size(prior_mu), fill::zeros);
    mat nw_sigma_0(size(prior_lambda), fill::eye);
    nw_sigma_0 /= nw_nu_0;

//put something here with uvec to specify the vector/lists of rows without NAs, then modify all the lines with y(i) to be .elem and x to be x.rows to subset them by the prespecified vectors
    
    // first do 'burn_in' iterations with iter < 0
    // then do 'max_iter' iterations while recording every
    // 'thin' samples
    for (long iter = -burn_in; iter < max_iter; ++iter)
      {
          Rcpp::Rcout << iter << endl;
#ifdef Rcpp_hpp
          Rcpp::checkUserInterrupt();
          
#endif
          if (discrete)
              for (size_t i = 0; i < y.n_cols; ++i)
              {
                  if (assignment[i])
                  {
                      double stddev;
                      vec beta = assignment[i]->sample_coefficients(stddev);
                      assignment[i]->remove(y.col(i), x);
                      vec means = x * beta;
                      for (size_t j = 0; j < y.n_rows; ++j)
                      {
                          y(j,i) = rtnorm(means[j], 1, 0.0, y_obs(j,i)>0);
                      }
                      assignment[i]->add(y.col(i), x);
                  }
              }

        for (size_t i = 0; i < y.n_cols; ++i)
          {
            if (assignment[i])
              {
                assignment[i]->remove(y.col(i), x);
                if (assignment[i]->is_empty())
                  {
                    for (size_t k = 0; k < clusters.size()-1; ++k)
                        if (clusters[k] == assignment[i])
                          {
                            swap(clusters[k], clusters.back());
                            break;
                          }
                    empty_clusters.push_back(clusters.back());
                    clusters.pop_back();
                  }
              }
            double total_probability = 0.0;
            for (size_t j = 0; j < clusters.size(); ++j)
              {
                clusters[j]->set_probability(y.col(i), x);
                total_probability += clusters[j]->get_probability();
              }
            empty_cluster.set_empty_probability(y.col(i), x, concentration);
            total_probability += empty_cluster.get_probability();

            total_probability *= R::unif_rand();
            for (size_t j = 0; j < clusters.size(); ++j)
              {
                total_probability -= clusters[j]->get_probability();
                if (total_probability < 0)
                  {
                    assignment[i] = clusters[j];
                    break;
                  }
              }
            if (total_probability >= 0)
              {
                clusters.push_back(empty_clusters.back());
                empty_clusters.pop_back();
                assignment[i] = clusters.back();
              }
            assignment[i]->add(y.col(i), x);
            
            // sample proposals
              mat proposal_prior_lambda;
              vec proposal_prior_mu;
              double proposal_prior_alpha = prior_alpha;
              double proposal_prior_beta = prior_beta;
#define SMART_PROPOSAL
#ifdef SMART_PROPOSAL
            vec sum_beta(size(prior_mu), fill::zeros);
            mat sum_beta_beta_t(size(prior_lambda), fill::zeros);
            unsigned long n = 0;
            for (size_t j = 0; j < clusters.size(); ++j)
              {
                clusters[j]->add_to_nw_guess(sum_beta, sum_beta_beta_t, n);
              }
            vec beta_bar = sum_beta / n;
            mat C = sum_beta_beta_t - n * beta_bar * beta_bar.t();
            double nw_kappa = nw_kappa_0 + n;
            double nw_nu  = nw_nu_0 + n;
            vec nw_mu = (nw_kappa_0 * nw_mu_0 + sum_beta) / (nw_kappa);
            mat nw_sigma = (nw_sigma_0.i() + C + (beta_bar - nw_mu_0) * (beta_bar - nw_mu_0).t() * nw_kappa_0 * n / nw_kappa).i();
            nw_sigma /= nw_nu;
              vec& reverse_nw_mu = nw_mu;
              mat& reverse_nw_sigma = nw_sigma;
#else
              double nw_kappa = 10;
              double nw_nu  = nw_kappa;
              vec& nw_mu = prior_mu;
              mat nw_sigma = prior_lambda / nw_nu;
#endif
            r_normal_wishart(proposal_prior_mu, proposal_prior_lambda,
                             nw_mu, nw_sigma, nw_kappa, nw_nu);
#ifndef SMART_PROPOSAL
              vec& reverse_nw_mu = proposal_prior_mu;
              mat reverse_nw_sigma = proposal_prior_lambda / nw_nu;
#endif
            double log_acceptance_prob = 0.0;
            log_acceptance_prob += log_d_normal_wishart(prior_mu, prior_lambda,
                                                        reverse_nw_mu, reverse_nw_sigma, nw_kappa, nw_nu);
            log_acceptance_prob -= log_d_normal_wishart(proposal_prior_mu, proposal_prior_lambda,
                                                        nw_mu, nw_sigma, nw_kappa, nw_nu);
              double add_log_acceptance_prob = 0.0;
              add_log_acceptance_prob += log_d_normal_wishart(proposal_prior_mu, proposal_prior_lambda,
                                                          nw_mu_0, nw_sigma_0, nw_kappa_0, nw_nu_0);
              add_log_acceptance_prob -= log_d_normal_wishart(prior_mu, prior_lambda,
                                                          nw_mu_0, nw_sigma_0, nw_kappa_0, nw_nu_0);
            for (size_t j = 0; j < clusters.size(); ++j)
              {
                add_log_acceptance_prob += clusters[j]->log_proposal_likelihood_ratio(proposal_prior_alpha, proposal_prior_beta, proposal_prior_lambda, proposal_prior_lambda * proposal_prior_mu,
                                                                                  prior_alpha, prior_beta, prior_lambda, prior_lambda * prior_mu);
              }
              log_acceptance_prob += add_log_acceptance_prob;
            if (R::unif_rand() < exp(log_acceptance_prob))
              {
                for (size_t j = 0; j < clusters.size(); ++j)
                  {
                    clusters[j]->change_prior(proposal_prior_alpha, proposal_prior_beta, proposal_prior_lambda, proposal_prior_lambda * proposal_prior_mu,
                                              prior_alpha, prior_beta, prior_lambda, prior_lambda * prior_mu);
                  }
                prior_alpha = proposal_prior_alpha;
                prior_beta = proposal_prior_beta;
                prior_lambda = proposal_prior_lambda;
                prior_mu = proposal_prior_mu;
              }
          }
        if ((iter >= 0) && (((iter+1) % thin) == 0))
          {
            for (size_t i = 0; i < y.n_cols; ++i)
                assignment_samples(i, iter / thin) = assignment[i]->get_id();
            mu_samples.col(iter / thin) = prior_mu;
            lambda_samples.slice(iter / thin) = prior_lambda;
          }
      }
}

uvec rdpcluster(const long n, const double concentration)
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
