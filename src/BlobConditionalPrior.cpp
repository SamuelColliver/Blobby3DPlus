#include "BlobConditionalPrior.h"

#include <cmath>

#include "DNest4/code/DNest4.h"
#include "Data.h"

// TODO: DNest4 requires LogGaussian distribution
// TODO: Once above are performed just use a for-loop in from/to uniform

/*
  Public
*/
BlobConditionalPrior::BlobConditionalPrior(
  size_t nlines,
  double fluxlim_min, double fluxlim_max,
  double flux_std_min, double flux_std_max,
  double ratiofluxlim_min, double ratiofluxlim_max,
  double ratioflux_std_min, double ratioflux_std_max,
  double radiuslim_min, double radiuslim_max,
  double rc_max,
  double wd_min, double wd_max,
  double qlim_min
  ) :nlines(nlines)
    ,fluxlim_min(fluxlim_min)
    ,fluxlim_max(fluxlim_max)
    ,flux_std_min(flux_std_min)
    ,flux_std_max(flux_std_max)
    ,ratiofluxlim_min(ratiofluxlim_min)
    ,ratiofluxlim_max(ratiofluxlim_max)
    ,ratioflux_std_min(ratioflux_std_min)
    ,ratioflux_std_max(ratioflux_std_max)
    ,radiuslim_min(radiuslim_min)
    ,radiuslim_max(radiuslim_max)
    ,rc_max(rc_max)
    ,wd_min(wd_min)
    ,wd_max(wd_max)
    ,qlim_min(qlim_min) {
  // Hyperprior distributions
  hyperprior_fluxmu.push_back(DNest4::Uniform(fluxlim_min, fluxlim_max));
  hyperprior_fluxstd.push_back(DNest4::LogUniform(flux_std_min, flux_std_max));
  for (size_t i=1; i<nlines; i++) {
    hyperprior_fluxmu.push_back(
      DNest4::Uniform(ratiofluxlim_min, ratiofluxlim_max));
    hyperprior_fluxstd.push_back(
      DNest4::LogUniform(ratioflux_std_min, ratioflux_std_max));
  }

  hyperprior_radiusmax = DNest4::LogUniform(radiuslim_min, radiuslim_max);
  hyperprior_wd = DNest4::LogUniform(wd_min, wd_max);
  hyperprior_qmin = DNest4::Uniform(qlim_min, 1.0);
}

void BlobConditionalPrior::from_prior(DNest4::RNG& rng) {
  // Hyperparameters
  wd = hyperprior_wd.generate(rng);

  for (size_t i=0; i<nlines; i++) {
    flux_mu.push_back(hyperprior_fluxmu[i].generate(rng));
    flux_std.push_back(hyperprior_fluxstd[i].generate(rng));
  }

  radiusmax = hyperprior_radiusmax.generate(rng);
  q_min = hyperprior_qmin.generate(rng);

  // Prior Distributions
  prior_rc = DNest4::TruncatedExponential(wd, 0.0, rc_max);
  prior_theta = DNest4::Uniform(0.0, 2.0*M_PI);
  for (size_t i=0; i<nlines; i++)
    prior_logflux.push_back(DNest4::Gaussian(flux_mu[i], flux_std[i]));
  prior_width = DNest4::Uniform(radiuslim_min, radiusmax);
  prior_q = DNest4::Triangular(q_min, 1.0, 1.0);
  prior_phi = DNest4::Uniform(0.0, M_PI);
}

double BlobConditionalPrior::perturb_hyperparameters(DNest4::RNG& rng) {
  double logH = 0.0;
  int which = rng.rand_int(5);
  switch (which) {
    case 0:
      logH += hyperprior_wd.perturb(wd, rng);
      prior_rc = DNest4::TruncatedExponential(wd, 0.0, rc_max);
      break;
    case 1:
      which = rng.rand_int(nlines);
      logH += hyperprior_fluxmu[which].perturb(flux_mu[which], rng);
      prior_logflux[which] = DNest4::Gaussian(flux_mu[which], flux_std[which]);
      break;
    case 2:
      which = rng.rand_int(nlines);
      logH += hyperprior_fluxstd[which].perturb(flux_std[which], rng);
      prior_logflux[which] = DNest4::Gaussian(flux_mu[which], flux_std[which]);
      break;
    case 3:
      logH += hyperprior_radiusmax.perturb(radiusmax, rng);
      prior_width = DNest4::Uniform(radiuslim_min, radiusmax);
      break;
    case 4:
      logH += hyperprior_qmin.perturb(q_min, rng);
      prior_q = DNest4::Triangular(q_min, 1.0, 1.0);
      break;
    }

  return logH;
}

double BlobConditionalPrior::log_pdf(const std::vector<double>& vec) const {
  if (vec[0] < 0.0 ||
      vec[1] < 0.0 || vec[1] > 2.0*M_PI ||
      vec[2] < radiuslim_min || vec[2] > radiusmax ||
      vec[3] < q_min || vec[3] > 1.0 ||
      vec[4] < 0.0 || vec[4] > M_PI)
    return -1E300;

  double logp = 0.0;

  // Exponential for radius
  logp += prior_rc.log_pdf(vec[0]);

  // Uniform for width with changing boundaries
  logp += prior_width.log_pdf(vec[2]);

  // Triangular distribution for q
  logp += prior_q.log_pdf(vec[3]);

  // Lognormal for flux
  for (size_t i=0; i<nlines; i++)
    logp += prior_logflux[i].log_pdf(log(vec[5+i]));

  return logp;
}

void BlobConditionalPrior::from_uniform(std::vector<double>& vec) const {
  vec[0] = prior_rc.cdf_inverse(vec[0]);
  vec[1] = prior_theta.cdf_inverse(vec[1]);
  vec[2] = prior_width.cdf_inverse(vec[2]);
  vec[3] = prior_q.cdf_inverse(vec[3]);
  vec[4] = prior_phi.cdf_inverse(vec[4]);
  for (size_t i=0; i<nlines; i++)
    vec[5+i] = exp(prior_logflux[i].cdf_inverse(vec[5+i]));
}

void BlobConditionalPrior::to_uniform(std::vector<double>& vec) const {
  vec[0] = prior_rc.cdf(vec[0]);
  vec[1] = prior_theta.cdf(vec[1]);
  vec[2] = prior_width.cdf(vec[2]);
  vec[3] = prior_q.cdf(vec[3]);
  vec[4] = prior_phi.cdf(vec[4]);
  for (size_t i=0; i<nlines; i++)
    vec[5+i] = prior_logflux[i].cdf(log(vec[5+i]));
}

void BlobConditionalPrior::print(std::ostream& out) const {
  // Print hyperparameters
  out<<wd<<' '
     <<radiuslim_min<<' '<<radiusmax<<' '
     <<q_min<<' ';
  for (size_t i=0; i<nlines; i++)
    out<<' '<<exp(flux_mu[i])<<' '<<flux_std[i]<<' ';
}
