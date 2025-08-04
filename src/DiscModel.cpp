#include "DiscModel.h"

#include <cmath>

#include "DNest4/code/DNest4.h"
#include "Data.h"
#include "LookupExp.h"
#include "LookupErf.h"
#include "Conv.h"
#include "Constants.h"

// TODO: Remove references to sigma1 throughout code.
// Partial fix: not perturbing.

/*
  Public
*/
DiscModel::DiscModel()
    :blobs(
      5 + static_cast<int>(Data::get_instance().get_em_line().size()),
      Data::get_instance().get_nmax(),
      Data::get_instance().get_nfixed(),
      BlobConditionalPrior(
        Data::get_instance().get_em_line().size(),
        log(Data::get_instance().get_fluxmu_min()),
        log(Data::get_instance().get_fluxmu_max()),
        Data::get_instance().get_lnfluxsd_min(),
        Data::get_instance().get_lnfluxsd_max(),
        log(Data::get_instance().get_ratiofluxmu_min()),
        log(Data::get_instance().get_ratiofluxmu_max()),
        Data::get_instance().get_lnratiofluxsd_min(),
        Data::get_instance().get_lnratiofluxsd_max(),
        Data::get_instance().get_radiuslim_min(),
        Data::get_instance().get_radiuslim_max(),
        Data::get_instance().get_rc_max(),
        Data::get_instance().get_wd_min(),
        Data::get_instance().get_wd_max(),
        Data::get_instance().get_qlim_min()
        ),
      DNest4::PriorType::log_uniform
      ),
      model(Data::get_instance().get_model()) {
  const size_t nlines = Data::get_instance().get_em_line().size();
  const size_t ni = Data::get_instance().get_ni();
  const size_t nj = Data::get_instance().get_nj();
  const size_t nr = Data::get_instance().get_nr();
  const size_t x_pad = Data::get_instance().get_x_pad();
  const size_t y_pad = Data::get_instance().get_y_pad();
  const double x_min = Data::get_instance().get_x_min();
  const double x_max = Data::get_instance().get_x_max();
  const double y_min = Data::get_instance().get_y_min();
  const double y_max = Data::get_instance().get_y_max();
  const double x_pad_dx = Data::get_instance().get_x_pad_dx();
  const double y_pad_dy = Data::get_instance().get_y_pad_dy();

  /*
    initialise arrays
  */
  // model cube
  preconvolved.resize(ni);
  for (size_t i=0; i<ni; ++i) {
    preconvolved[i].resize(nj);
    for (size_t j=0; j<nj; ++j) {
      preconvolved[i][j].resize(nr);
    }
  }

  // Convolved cube
  convolved.resize(ni - 2*y_pad);
  for (size_t i=0; i<convolved.size(); i++) {
    convolved[i].resize(nj - 2*x_pad);
    for (size_t j=0; j<convolved[i].size(); j++) {
        convolved[i][j].resize(nr);
    }
  }

  // Shifted arrays
  x_shft.assign(ni, std::vector<double>(nj));
  y_shft.assign(ni, std::vector<double>(nj));

  // Radius and cos(angle) maps
  rad.assign(ni, std::vector<double>(nj));
  cos_angle.assign(ni, std::vector<double>(nj));

  // Moment maps
  flux.resize(nlines);
  for (size_t i=0; i<flux.size(); i++) {
    flux[i].resize(ni);
    for (size_t j=0; j<flux[i].size(); j++) {
      flux[i][j].resize(nj);
    }
  }

  rel_lambda.assign(ni, std::vector<double>(nj));
  vdisp.assign(ni, std::vector<double>(nj));

  /*
    Prior distributions
  */
  // Inclination
  inc = Data::get_instance().get_inc();
  prior_pa = DNest4::Uniform(0.0, 2.0*M_PI);
  prior_xc = DNest4::TruncatedCauchy(
    Data::get_instance().get_x_imcentre(),
    Data::get_instance().get_gamma_pos(),
    x_min + x_pad_dx, x_max - x_pad_dx
    );
  prior_yc = DNest4::TruncatedCauchy(
    Data::get_instance().get_y_imcentre(),
    Data::get_instance().get_gamma_pos(),
    y_min + y_pad_dy, y_max - y_pad_dy
    );

  prior_vsys = DNest4::TruncatedCauchy(
    0.0,
    Data::get_instance().get_vsys_gamma(),
    -Data::get_instance().get_vsys_max(),
    Data::get_instance().get_vsys_max()
    );
  prior_vmax = DNest4::LogUniform(
    Data::get_instance().get_vmax_min(),
    Data::get_instance().get_vmax_max());
  prior_vslope = DNest4::LogUniform(
    Data::get_instance().get_vslope_min(),
    Data::get_instance().get_vslope_max());
  prior_vgamma = DNest4::LogUniform(
    Data::get_instance().get_vgamma_min(),
    Data::get_instance().get_vgamma_max());
  prior_vbeta = DNest4::Uniform(
    Data::get_instance().get_vbeta_min(),
    Data::get_instance().get_vbeta_max());

  vdisp_order = Data::get_instance().get_vdisp_order();
  vdisp_param.assign(vdisp_order + 1, 0.0);
  prior_vdisp0 = DNest4::Uniform(
    Data::get_instance().get_vdisp0_min(),
    Data::get_instance().get_vdisp0_max());
  prior_vdisp = DNest4::Gaussian(
    0.0,
    Data::get_instance().get_vdispn_sigma());

  prior_sigma0 = DNest4::LogUniform(
    Data::get_instance().get_sigma_min(),
    Data::get_instance().get_sigma_max());
  prior_sigma1 = DNest4::LogUniform(
    Data::get_instance().get_sigma1_min(),
    Data::get_instance().get_sigma1_max());

  if (model != 0) {
    prior_Md = DNest4::LogUniform(
      Data::get_instance().get_Md_min(),
      Data::get_instance().get_Md_max());
    prior_wxd = DNest4::LogUniform(
      Data::get_instance().get_wxd_min(),
      Data::get_instance().get_wxd_max());
  }
}

void DiscModel::from_prior(DNest4::RNG& rng) {
  /*
    Initialise: global parameters
  */
  // Initialise: Position
  xcd = prior_xc.generate(rng);
  ycd = prior_yc.generate(rng);

  pa = prior_pa.generate(rng);

  /*
    Initialise: Velocity profile parameters
  */
  vsys = prior_vsys.generate(rng);
  vmax = prior_vmax.generate(rng);
  vslope = prior_vslope.generate(rng);
  vgamma = prior_vgamma.generate(rng);
  vbeta = prior_vbeta.generate(rng);

  /*
    Initialise: Blobs
  */
  if (model == 0) {
    do {
      blobs.from_prior(rng);
    } while (blobs.get_components().size() == 0);
  } else if (model == 2) {
    blobs.from_prior(rng);
  }
  /*
    Initialise: Disc
  */
  if ((model == 1) || (model == 2)) {
    Md = prior_Md.generate(rng);
    wxd = prior_wxd.generate(rng);
  } else {
    Md = 0.0;
    wxd = 0.0;
  }

  // Initialise: Dispersion
  vdisp_param[0] = prior_vdisp0.generate(rng);
  for (int v=1; v<vdisp_order+1; v++)
      vdisp_param[v] = prior_vdisp.generate(rng);

  // Initialise: Noise
  sigma0 = prior_sigma0.generate(rng);
  sigma1 = prior_sigma1.generate(rng);

  // Calculate cubes based on initial values
  array_perturb = true;
  vel_perturb = true;
  vdisp_perturb = true;
  blob_perturb = true;
  noise_perturb = true;

  if (model == 0)
    disc_flux_perturb = false;
  else
    disc_flux_perturb = true;

  calculate_cube();
}

double DiscModel::perturb(DNest4::RNG& rng) {
  double logH = 0.0;
  double rnd = rng.rand();

  array_perturb = false;
  vel_perturb = false;
  vdisp_perturb = false;
  disc_flux_perturb = false;

  if (rnd < 0.9) {
    // skip disc perturb for model 0
    if (model == 0)
      rnd = 0.8*rng.rand();
    else if (model == 1)
      rnd = 1.0 - 0.3*rng.rand();
    else
      rnd = rng.rand();

    if (rnd < 0.7) {
      // Perturb blob parameters
      logH += blobs.perturb(rng);
      if ((model == 0) & (blobs.get_components().size() == 0))
        return logH = -1E300;

    } else if (rnd < 0.8) {
      // Perturb disc parameters
      int which = rng.rand_int(10);

      switch (which) {
        case 0:
          logH += prior_xc.perturb(xcd, rng);
          array_perturb = true;
          break;
        case 1:
          logH += prior_yc.perturb(ycd, rng);
          array_perturb = true;
          break;
        case 2:
          logH += prior_vdisp0.perturb(vdisp_param[0], rng);
          vdisp_perturb = true;
          break;
        case 3:
          logH += prior_vsys.perturb(vsys, rng);
          vel_perturb = true;
          break;
        case 4:
          logH += prior_vmax.perturb(vmax, rng);
          vel_perturb = true;
          break;
        case 5:
          logH += prior_vslope.perturb(vslope, rng);
          vel_perturb = true;
          break;
        case 6:
          logH += prior_vgamma.perturb(vgamma, rng);
          vel_perturb = true;
          break;
        case 7:
          logH += prior_vbeta.perturb(vbeta, rng);
          vel_perturb = true;
          break;
        case 8:
          logH += prior_pa.perturb(pa, rng);
          array_perturb = true;
          break;
        case 9:
          which = rng.rand_int(vdisp_order);
          logH += prior_vdisp.perturb(vdisp_param[which+1], rng);
          vdisp_perturb = true;
          break;
      }
    } else {
      // Perturb disc flux parameters
      disc_flux_perturb = true;
      int which = rng.rand_int(2);
      switch (which) {
        case 0:
          logH += prior_Md.perturb(Md, rng);
          break;
        case 1:
          logH += prior_wxd.perturb(wxd, rng);
          break;
      }
    }

    // Pre-rejection trick
    if (log(rng.rand()) < logH) {
      logH = 0.0;
      calculate_cube();
    } else {
      logH = -1E300;
    }

  } else {
    int which = rng.rand_int(1);
    switch (which) {
      case 0:
        logH += prior_sigma0.perturb(sigma0, rng);
        break;
      case 1:
        // Currently redundant
        logH += prior_sigma1.perturb(sigma1, rng);
        break;
    }
  }

  return logH;
}

double DiscModel::log_likelihood() const {
  const std::vector< std::vector< std::vector<double> > >&
    data = Data::get_instance().get_data();
  const std::vector< std::vector< std::vector<double> > >&
    var_cube = Data::get_instance().get_var();
  const std::vector< std::vector<int> >&
    valid = Data::get_instance().get_valid();

  long double logL = 0.0;

  if ((model == 0) && (blobs.get_components().size() == 0)) {
    // If no blobs return prob = 0
    logL = -1E300;

  } else {
    double var;
    int i, j;

    double sigma0sq = sigma0*sigma0;

    for (size_t h=0; h<valid.size(); h++) {
      i = valid[h][0];
      j = valid[h][1];
      for (size_t r=0; r<data[i][j].size(); r++) {
        if (var_cube[i][j][r] != 0.0) {
          var = var_cube[i][j][r] + sigma0sq;
          logL += -0.5*log(2.0*M_PI*var);
          logL += -0.5*pow(data[i][j][r] - convolved[i][j][r], 2)/var;
        }
      }
    }
  }

  return logL;
}

void DiscModel::print(std::ostream& out) const {
  const int x_pad = Data::get_instance().get_x_pad();
  const int y_pad = Data::get_instance().get_y_pad();

  const bool save_maps = true;
  const bool save_preconvolved = true;
  const bool save_convolved = true;

  out<<std::setprecision(6);

  if (save_maps) {
    for (size_t l=0; l<flux.size(); l++)
      for (size_t i=0; i<flux[l].size(); i++)
        for (size_t j=0; j<flux[l][i].size(); j++)
          out << flux[l][i][j] << ' ';

    for (size_t i=0; i<rel_lambda.size(); i++)
      for (size_t j=0; j<rel_lambda[i].size(); j++)
        out << (rel_lambda[i][j] - 1.0)*constants::C << ' ';

    for (size_t i=0; i<vdisp.size(); i++)
      for (size_t j=0; j<vdisp[i].size(); j++)
        out << vdisp[i][j]*constants::C << ' ';
  }

  if (save_preconvolved) {
    for (size_t i=y_pad; i<preconvolved.size()-y_pad; i++)
        for (size_t j=x_pad; j<preconvolved[i].size()-x_pad; j++)
          for (size_t r=0; r<preconvolved[i][j].size(); r++)
              out << preconvolved[i][j][r] << ' ';
  }

  if (save_convolved) {
    for (size_t i=0; i<convolved.size(); i++)
      for (size_t j=0; j<convolved[i].size(); j++)
        for (size_t r=0; r<convolved[i][j].size(); r++)
          out << convolved[i][j][r] << ' ';
  }

  // Save components
  blobs.print(out); out<<' ';

  // Save global variables
  out<<xcd<<' ';
  out<<ycd<<' ';
  out<<Md<<' ';
  out<<wxd<<' ';
  out<<vsys<<' ';
  out<<vmax<<' ';
  out<<vslope<<' ';
  out<<vgamma<<' ';
  out<<vbeta<<' ';
  for (int i=0; i<vdisp_order+1; i++)
    out<<vdisp_param[i]<<' ';
  out<<inc<<' ';
  out<<pa<<' ';
  out<<sigma0<<' ';
  out<<sigma1<<' ';
}

std::string DiscModel::description() const {
  return std::string("blobs");
}

/*
  Private
*/
void DiscModel::calculate_cube() {
  /*
    Calculate cube as a function of model parameters.
  */
  bool update;  // Determine if adding blobs
  std::vector< std::vector<double> > components;

  // Calculate position arrays
  if (array_perturb)
    calculate_shifted_arrays();

  // Calculate relative lambda array
  if (vel_perturb || array_perturb)
    calculate_rel_lambda();

  // Calculate velocity dispersion array
  if (vdisp_perturb || array_perturb)
    calculate_vdisp();

  //  Calculate flux map
  switch (model) {
    case 0:
      // Blobs only model
      update = blobs.get_removed().size() == 0;
      if (array_perturb || !update) {
        clear_flux_map();
        components = blobs.get_components();

      } else {
        components = blobs.get_added();
      }
      add_blob_flux(components);
      break;
    case 1:
      // Disc only model
      update = false;
      clear_flux_map();
      if (disc_flux_perturb || array_perturb)
        add_disc_flux();
      break;
    case 2:
      // Disc + blobs model
      update = blobs.get_removed().size() == 0;
      if (disc_flux_perturb || array_perturb || !update) {
        clear_flux_map();
        components = blobs.get_components();
        add_disc_flux();

      } else {
        components = blobs.get_added();
      }
      add_blob_flux(components);
      break;
  }

  construct_cube();
  convolved = conv.apply(preconvolved);
}

void DiscModel::construct_cube() {
  /*
    Create cube from maps.
  */
  const std::vector< std::vector<double> >
    em_line = Data::get_instance().get_em_line();

  // clear cube where flux is 0
  for (size_t i=0; i<preconvolved.size(); i++)
    for (size_t j=0; j<preconvolved[i].size(); j++)
        std::fill(preconvolved[i][j].begin(), preconvolved[i][j].end(), 0.0);

  for (size_t l=0; l<em_line.size(); l++) {
    // Apply flux for main line
    construct_line_cube(em_line[l][0], 1.0, flux[l]);
    for (size_t ll=0; ll<(em_line[l].size()-1)/2; ll++)
      construct_line_cube(em_line[l][1+2*ll], em_line[l][2+2*ll], flux[l]);
  }
}

void DiscModel::construct_line_cube(
  double line, double factor, std::vector< std::vector<double> >& flux_map) {
  
  const double sigma_lsfsq = pow(Data::get_instance().get_lsf_sigma(), 2);
  const std::vector<Data::WavelengthWindow>& windows = Data::get_instance().get_wavelength_windows();
  const int nr = Data::get_instance().get_nr();  // add explicit nr definition
  
  // find which window contains this line
  int target_window = -1;
  for (size_t w = 0; w < windows.size(); w++) {
    if (line >= windows[w].r_min && line <= windows[w].r_max) {
      target_window = w;
      break;
    }
  }
  
  if (target_window == -1) {
    std::cerr << "# ERROR: Line " << line << " Å not found in any wavelength window!" << std::endl;
    std::cerr << "This should have been caught during validation." << std::endl;
    return;  // early return if line not found
  }
  
  const auto& window = windows[target_window];
  
  double lambda, sigma_lambda, invtwo_wlsq;
  double ha_cdf_min, ha_cdf_max;
  
  for (size_t i = 0; i < preconvolved.size(); i++) {
    for (size_t j = 0; j < preconvolved[i].size(); j++) {
      // calculate line properties at this spatial position
      lambda = line * rel_lambda[i][j];
      sigma_lambda = line * vdisp[i][j];
      invtwo_wlsq = 1.0 / sqrt(2.0 * (pow(sigma_lambda, 2) + sigma_lsfsq));
      
      // process all wavelength bins in the target window
      for (int local_r = 0; local_r < window.n_bins; local_r++) {
        int global_r = window.start_idx + local_r;  // global index in preconvolved array
        
        // add bounds checking
        if (global_r >= nr || global_r < 0) {
          std::cerr << "# ERROR: global_r index " << global_r 
                   << " out of bounds (nr=" << nr << ") for line " << line 
                   << " Å in window " << target_window << std::endl;
          continue;
        }
        
        double wave_center = window.r[local_r];
        
        ha_cdf_min = LookupErf::evaluate((wave_center - 0.5*window.dr - lambda) * invtwo_wlsq);
        ha_cdf_max = LookupErf::evaluate((wave_center + 0.5*window.dr - lambda) * invtwo_wlsq);
        
        preconvolved[i][j][global_r] += 0.5 * factor * flux_map[i][j] * (ha_cdf_max - ha_cdf_min);
      }
    }
  }
}

void DiscModel::calculate_shifted_arrays() {
  /*
    Calculate arrays shifted by disk parameters.
  */
  const std::vector< std::vector<double> >&
    x = Data::get_instance().get_x();
  const std::vector< std::vector<double> >&
    y = Data::get_instance().get_y();

  double sin_pa = sin(pa);
  double cos_pa = cos(pa);
  double invcos_inc = 1.0/cos(inc);

  double xx_rot, yy_rot;

  for (size_t i=0; i<preconvolved.size(); i++) {
    for (size_t j=0; j<preconvolved[i].size(); j++) {
      // Shift
      x_shft[i][j] = x[i][j] - xcd;
      y_shft[i][j] = y[i][j] - ycd;

      // rotate by pa around z (counter-clockwise, East pa = 0)
      xx_rot = x_shft[i][j]*cos_pa + y_shft[i][j]*sin_pa;
      yy_rot = -x_shft[i][j]*sin_pa + y_shft[i][j]*cos_pa;

      // rotate by inclination around yy_rot
      yy_rot *= invcos_inc;

      // calculate radius
      rad[i][j] = sqrt(xx_rot*xx_rot + yy_rot*yy_rot);

      // calculate angle to receding major axis
      if ((xx_rot != 0.0) || (yy_rot != 0.0))
        cos_angle[i][j] = cos(atan2(yy_rot, xx_rot));
      else
        cos_angle[i][j] = 1.0;
    }
  }
}

void DiscModel::add_disc_flux() {
  /*
    Add disc flux component to flux map. Assumes flux profile is same for all
    emission lines.
    TODO: Generalise Md, wxd to account for multiple emission lines.
  */
  const double dx = Data::get_instance().get_dx();
  const double dy = Data::get_instance().get_dy();

  double invwxd  = 1.0/wxd;
  double amp = dx*dy*Md*invwxd;

  for (size_t l=0; l<flux.size(); l++)
    for (size_t i=0; i<flux[l].size(); i++)
      for(size_t j=0; j<flux[l][i].size(); j++)
        flux[l][i][j] += amp*LookupExp::evaluate(rad[i][j]*invwxd);
}

void DiscModel::add_blob_flux(std::vector< std::vector<double> >& components) {
  /*
    Calculate flux map.
  */
  const double dx = Data::get_instance().get_dx();
  const double dy = Data::get_instance().get_dy();
  const double sigma_cutoffsq = pow(
    Data::get_instance().get_sigma_cutoff(), 2);
  const double pixel_width = Data::get_instance().get_pixel_width();
  const size_t nlines = Data::get_instance().get_em_line().size();

  double sin_pa = sin(pa);
  double cos_pa = cos(pa);
  double cos_inc = cos(inc);
  double invcos_inc = 1.0/cos_inc;

  // Blob parameters
  double rc, thetac, wx, q, phi;
  std::vector<double> f(nlines);
  double xc, yc;
  double wxsq;
  double qsq, invq, sqrtq;
  double invwxsq;
  double sin_phi, cos_phi;
  std::vector<double> amp(nlines);

  // Rotated disk coordinates
  double xd_shft, yd_shft;
  double xb_shft, yb_shft;
  double xxd_rot, yyd_rot;
  double xxb_rot, yyb_rot;
  double rsq;

  // oversampled parameters
  int si;
  double dxfs, dyfs;
  std::vector<double> amps(nlines);

  // Blob contribution
  for (size_t k=0; k<components.size(); ++k) {
    // Components
    rc = components[k][0];
    thetac = components[k][1];
    wx = components[k][2];
    q = components[k][3];
    phi = components[k][4];
    // for (size_t l=0; l<nlines; l++)
    //  f[l] = components[k][5+l];


    //before
    //f[0] = components[k][5];
    //for (size_t l=1; l<nlines; l++)
    //  f[l] = components[k][5]*components[k][5+1];  // BUG: All secondary lines get same ratio!

    //
    for (size_t l=0; l<nlines; l++)
      f[l] = components[k][5+l]; // remove flux constrained line
    //


    // xc, yc in disc plane
    xc = rc*cos(thetac);
    yc = rc*sin(thetac);

    // Component manipulations
    wxsq = wx*wx;
    qsq = q*q;
    sqrtq = sqrt(q);
    invwxsq = 1.0/(wx*wx);
    invq = 1.0/q;
    sin_phi = sin(phi);
    cos_phi = cos(phi);

    // Oversampling rate
    if (q*wx*cos_inc < 0.5*pixel_width)
      si = 2;
    else if (q*wx*cos_inc < 1.0*pixel_width)
      si = 1;
    else
      si = 0;

    dxfs = dx/(2.0*si + 1.0);
    dyfs = dy/(2.0*si + 1.0);

    // Flux normalised sum
    for (size_t l=0; l<nlines; l++)
      amp[l] = dxfs*dyfs*f[l]/(2.0*M_PI*wxsq*cos_inc);

    for (size_t i=0; i<flux[0].size(); i++) {
      for (size_t j=0; j<flux[0][i].size(); j++) {
        for (size_t l=0; l<flux.size(); l++)
          amps[l] = 0.0;
        for (int is=-si; is<=si; is++) {
          for (int js=-si; js<=si; js++) {
            /*
              Get rotated/inc disk coordinates
            */
            // Shift
            xd_shft = x_shft[i][j] + js*dxfs;
            yd_shft = y_shft[i][j] + is*dyfs;

            // rotate by pa around z (counter-clockwise, East pa = 0)
            xxd_rot = xd_shft*cos_pa + yd_shft*sin_pa;
            yyd_rot = -xd_shft*sin_pa + yd_shft*cos_pa;

            // rotate by inclination around yy_rot
            yyd_rot *= invcos_inc;

            /*
              Get distance wrt centre of blob in rotated/inc disk coordinates.
            */
            // Shift
            xb_shft = xxd_rot - xc;
            yb_shft = yyd_rot - yc;

            // Rotate
            xxb_rot = xb_shft*cos_phi + yb_shft*sin_phi;
            yyb_rot = -xb_shft*sin_phi + yb_shft*cos_phi;

            // Calculate normalised squared distance to centre of blob
            rsq = q*pow(xxb_rot, 2) + invq*pow(yyb_rot, 2);
            rsq *= invwxsq;

            if (rsq < sigma_cutoffsq) {
              for (size_t l=0; l<flux.size(); l++)
                amps[l] += amp[l]*LookupExp::evaluate(0.5*rsq);
            }
          }
        }
        for (size_t l=0; l<flux.size(); l++)
          flux[l][i][j] += amps[l];
      }
    }
  }
}

void DiscModel::calculate_rel_lambda() {
  /*
    Calculate relative lambda (ie. relative velocity) shift map.
  */
  double sin_inc = sin(inc);

  for (size_t i=0; i<rel_lambda.size(); i++) {
    for (size_t j=0; j<rel_lambda[i].size(); j++) {
      // Calc relative lambda
      if (rad[i][j] == 0.0) {
        rel_lambda[i][j] = 0.0;
      } else {
        rel_lambda[i][j] = vmax*pow(1.0 + vslope/rad[i][j], vbeta);
        rel_lambda[i][j] /= pow(
          1.0 + pow(vslope/rad[i][j], vgamma), 1.0/vgamma);
        rel_lambda[i][j] *= sin_inc*cos_angle[i][j];
      }
      rel_lambda[i][j] += vsys;
      rel_lambda[i][j] /= constants::C;
      rel_lambda[i][j] += 1.0;
    }
  }
}

void DiscModel::calculate_vdisp() {
  /*
    Calculate velocity dispersion map.
  */
  for (size_t i=0; i<vdisp.size(); i++) {
    for (size_t j=0; j<vdisp[i].size(); j++) {
      vdisp[i][j] = vdisp_param[0];
      for (int v=0; v<vdisp_order; v++)
        vdisp[i][j] += vdisp_param[v+1]*pow(rad[i][j], v+1);
      vdisp[i][j] = exp(vdisp[i][j])/constants::C;
    }
  }
}

void DiscModel::clear_flux_map() {
  for (size_t l=0; l<flux.size(); l++)
    for (size_t i=0; i<flux[l].size(); i++)
      for (size_t j=0; j<flux[l][i].size(); j++)
        flux[l][i][j] = 0.0;
}
