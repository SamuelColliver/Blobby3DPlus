#include "Data.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <algorithm>

#include "Constants.h"

Data Data::instance;

Data::Data() {}

void Data::load(const char* moptions_file) {
  // Loading data comment
  std::cout<<"\nLoading data:\n";

  // Load model options
  std::fstream fin(moptions_file, std::ios::in);
  if (!fin)
    std::cerr<<"# ERROR: couldn't open file "<<moptions_file<<"."<<std::endl;

  size_t n;
  std::string line;
  std::string name;
  std::string tmp_str;
  double tmp_double;
  std::vector<double> tmp_vector;
  bool line_flag = false;
  bool lsf_fwhm_flag = false;
  bool psf_amp_flag = false;
  bool psf_fwhm_flag = false;
  bool psf_beta_flag = false;
  bool inc_flag = false;
  bool gamma_pos_flag = false;
  bool radiuslim_min_flag = false;
  while (std::getline(fin, line)) {
    std::istringstream lin(line);
    lin >> name;
    std::transform(name.begin(), name.end(), name.begin(), ::toupper);

    if (line.empty()) {
      continue;
    } else if (name[0] == '#') {
      continue;
    } else if (name == "METADATA_FILE") {
        lin >> metadata_file;
    } else if (name == "DATA_FILE") {
      lin >> data_file;
    } else if (name == "VAR_FILE") {
      lin >> var_file;
    } else if (name == "CONVOLVE_METHOD") {
      lin >> convolve;
    } else if (name == "PSFWEIGHT") {
      while (lin >> tmp_double)
        psf_amp.push_back(tmp_double);
      psf_amp_flag = true;
    } else if (name == "PSFFWHM") {
      while (lin >> tmp_double)
        psf_fwhm.push_back(tmp_double);
      psf_fwhm_flag = true;
    } else if (name == "PSFBETA") {
      lin >> psf_beta;
      psf_beta_flag = true;
    } else if (name == "LSFFWHM") {
      lin >> lsf_fwhm;
      lsf_fwhm_flag = true;
    } else if (name == "NMAX") {
      lin >> nmax;
    } else if (name == "NFIXED") {
      lin >> tmp_str;
      std::transform(
        tmp_str.begin(), tmp_str.end(),
        tmp_str.begin(), ::toupper);
      if ((tmp_str == "FALSE") || (tmp_str == "0")) {
        nfixed = false;
      } else if ((tmp_str == "TRUE") || (tmp_str == "1")) {
        nfixed = true;
      } else {
        std::cerr<<"# ERROR: couldn't determine N_FIXED."<<std::endl;
        exit(0);
      }
    } else if (name == "VSYS_MAX") {
      lin >> vsys_max;
    } else if (name == "VSYS_GAMMA") {
      lin >> vsys_gamma;
    } else if (name == "VC_MIN") {
      lin >> vmax_min;
    } else if (name == "VC_MAX") {
      lin >> vmax_max;
    } else if (name == "MEANFLUX_MIN") {
      lin >> fluxmu_min;
    } else if (name == "MEANFLUX_MAX") {
      lin >> fluxmu_max;
    } else if (name == "LOGSIGMAFLUX_MIN") {
      lin >> lnfluxsd_min;
    } else if (name == "LOGSIGMAFLUX_MAX") {
      lin >> lnfluxsd_max;
    } else if (name == "MEANRATIOFLUX_MIN") {
      lin >> ratiofluxmu_min;
    } else if (name == "MEANRATIOFLUX_MAX") {
      lin >> ratiofluxmu_max;
    } else if (name == "LOGSIGMARATIOFLUX_MIN") {
      lin >> lnratiofluxsd_min;
    } else if (name == "LOGSIGMARATIOFLUX_MAX") {
      lin >> lnratiofluxsd_max;
    } else if (name == "SIGMAV0_MIN") {
      lin >> vdisp0_min;
    } else if (name == "SIGMAV0_MAX") {
      lin >> vdisp0_max;
    } else if (name == "QLIM_MIN") {
      lin >> qlim_min;
    } else if (name == "INC") {
      lin >> inc;
      inc_flag = true;
    } else if (name == "SIGMA0_MIN") {
      lin >> sigma_min;
    } else if (name == "SIGMA0_MAX") {
      lin >> sigma_max;
    } else if (name == "RADIUSLIM_MIN") {
      lin >> radiuslim_min; // Not passed to ConditionalPrior yet
      radiuslim_min_flag = true;
    } else if (name == "RADIUSLIM_MAX") {
      lin >> radiuslim_max;
    } else if (name == "WD_MIN") {
      lin >> wd_min;
    } else if (name == "WD_MAX") {
      lin >> wd_max;
    } else if (name == "VSLOPE_MIN") {
      lin >> vslope_min;
    } else if (name == "VSLOPE_MAX") {
      lin >> vslope_max;
    } else if (name == "VGAMMA_MIN") {
      lin >> vgamma_min;
    } else if (name == "VGAMMA_MAX") {
      lin >> vgamma_max;
    } else if (name == "VBETA_MIN") {
      lin >> vbeta_min;
    } else if (name == "VBETA_MAX") {
      lin >> vbeta_max;
    } else if (name == "VDISP_ORDER") {
      lin >> vdisp_order;
    } else if (name == "LOGVDISP0_MIN") {
      lin >> vdisp0_min;
    } else if (name == "LOGVDISP0_MAX") {
      lin >> vdisp0_max;
    } else if (name == "VDISPN_SIGMA") {
      lin >> vdispn_sigma;
    } else if (name == "SIGMA1_MIN") {
      lin >> sigma1_min;
    } else if (name == "SIGMA1_MAX") {
      lin >> sigma1_max;
    } else if (name == "MD_MIN") {
      lin >> Md_min;
    } else if (name == "MD_MAX") {
      lin >> Md_max;
    } else if (name == "WXD_MIN") {
      lin >> wxd_min;
    } else if (name == "WXD_MAX") {
      lin >> wxd_max;
    } else if (name == "CENTRE_GAMMA") {
      lin >> gamma_pos;
      gamma_pos_flag = true;
    } else if (name == "LINE") {
      tmp_vector.clear();
      while (lin >> tmp_double) {
        tmp_vector.push_back(tmp_double);
      }
      n = em_line.size();
      em_line.resize(n+1);
      em_line[n].resize(tmp_vector.size());
      for (size_t i=0; i<tmp_vector.size(); i++)
        em_line[n][i] = tmp_vector[i];
      line_flag = true;
    } else {
      std::cerr
        <<"Couldn't determine input parameter assignment for keyword: "
        <<name<<"."
        <<std::endl;
        exit(0);
    }
  }
  fin.close();

  // Check required parameters are provided
  if (!line_flag) {
    std::cerr
      <<"# ERROR: Required keyword (LINE) not provided."<<std::endl;
    exit(0);
  }

  if (!lsf_fwhm_flag) {
    std::cerr
      <<"# ERROR: Required keyword (LSFFWHM) not provided."<<std::endl;
    exit(0);
  }

  if (convolve == 0) {
    if (!psf_amp_flag) {
      std::cerr
        <<"# ERROR: Required keyword (PSFWEIGHT) not provided."<<std::endl;
      exit(0);
    }

    if (!psf_fwhm_flag) {
      std::cerr
        <<"# ERROR: Required keyword (PSFFWHM) not provided."<<std::endl;
      exit(0);
    }
  } else if (convolve == 1) {
    if (!psf_beta_flag) {
      std::cerr
        <<"# ERROR: Required keyword (PSFBETA) not provided."<<std::endl;
      exit(0);
    }
  }

  if (!inc_flag) {
    std::cerr<<"# ERROR: Required keyword (INC) not provided."<<std::endl;
    exit(0);
  }

  // sigma cutoff parameter for blobs
  sigma_cutoff = 5.0;

  // PSF convolution method message
  for(size_t i=0; i<psf_fwhm.size(); i++)
    psf_sigma.push_back(psf_fwhm[i]/sqrt(8.0*log(2.0)));

  // LSF in wavelength (needs to be redshift corrected)
  lsf_sigma = lsf_fwhm/sqrt(8.0*log(2.0));

  // Model choice (only for testing)
  model = 0;

  // Override blob parameters for disk model
  if (model == 1) {
    nmax = 0;
    nfixed = true;
  }

  // Spatial sampling of cube
  sample = 1;

  // NEW METADATA LOADING SECTION - REPLACES OLD METADATA READING
  std::cout << "Loading metadata from: " << metadata_file << std::endl;
  fin.open(metadata_file, std::ios::in);
  if (!fin) {
    std::cerr << "# ERROR: couldn't open file " << metadata_file << "." << std::endl;
    exit(1);
  }

  std::string metadata_line, keyword;
  double value1, value2;
  
  // initialise validation flags
  bool ni_set = false, nj_set = false;
  bool x_min_set = false, x_max_set = false;
  bool y_min_set = false, y_max_set = false;
  
  while (std::getline(fin, metadata_line)) {
    // skip empty lines and comments
    if (metadata_line.empty() || metadata_line[0] == '#') {
      continue;
    }
    
    // remove inline comments
    size_t comment_pos = metadata_line.find('#');
    if (comment_pos != std::string::npos) {
      metadata_line = metadata_line.substr(0, comment_pos);
    }
    
    // trim whitespace
    metadata_line.erase(0, metadata_line.find_first_not_of(" \t"));
    metadata_line.erase(metadata_line.find_last_not_of(" \t") + 1);
    
    if (metadata_line.empty()) continue;
    
    std::istringstream iss(metadata_line);
    iss >> keyword;
    
    // convert keyword to uppercase for case-insensitive comparison
    std::transform(keyword.begin(), keyword.end(), keyword.begin(), ::toupper);
    
    if (keyword == "NI") {
      iss >> ni;
      if (iss.fail()) {
        std::cerr << "# ERROR: Invalid ni value: " << metadata_line << std::endl;
        exit(1);
      }
      ni_set = true;
      std::cout << "  NI: " << ni << std::endl;
      
    } else if (keyword == "NJ") {
      iss >> nj;
      if (iss.fail()) {
        std::cerr << "# ERROR: Invalid nj value: " << metadata_line << std::endl;
        exit(1);
      }
      nj_set = true;
      std::cout << "  NJ: " << nj << std::endl;
      
    } else if (keyword == "X_MIN") {
      iss >> x_min;
      if (iss.fail()) {
        std::cerr << "# ERROR: Invalid x_min value: " << metadata_line << std::endl;
        exit(1);
      }
      x_min_set = true;
      std::cout << "  X_MIN: " << x_min << std::endl;
      
    } else if (keyword == "X_MAX") {
      iss >> x_max;
      if (iss.fail()) {
        std::cerr << "# ERROR: Invalid x_max value: " << metadata_line << std::endl;
        exit(1);
      }
      x_max_set = true;
      std::cout << "  X_MAX: " << x_max << std::endl;
      
    } else if (keyword == "Y_MIN") {
      iss >> y_min;
      if (iss.fail()) {
        std::cerr << "# ERROR: Invalid y_min value: " << metadata_line << std::endl;
        exit(1);
      }
      y_min_set = true;
      std::cout << "  Y_MIN: " << y_min << std::endl;
      
    } else if (keyword == "Y_MAX") {
      iss >> y_max;
      if (iss.fail()) {
        std::cerr << "# ERROR: Invalid y_max value: " << metadata_line << std::endl;
        exit(1);
      }
      y_max_set = true;
      std::cout << "  Y_MAX: " << y_max << std::endl;
      
    } else if (keyword == "WAVE_RANGE") {
      double r_min_win, r_max_win;
      int orig_start, orig_end, n_bins_win;
      
      iss >> r_min_win >> r_max_win >> orig_start >> orig_end >> n_bins_win;
      
      if (iss.fail()) {
        std::cerr << "# ERROR: Invalid wave_range format: " << metadata_line << std::endl;
        std::cerr << "Expected: wave_range <min> <max> <start_bin> <end_bin> <n_bins>" << std::endl;
        exit(1);
      }
      
      // validation checks
      if (r_min_win >= r_max_win) {
        std::cerr << "# ERROR: Invalid wavelength range - min must be < max: " 
                 << r_min_win << " >= " << r_max_win << std::endl;
        exit(1);
      }
      
      if (orig_start > orig_end) {
        std::cerr << "# ERROR: Invalid bin range - start_bin must be <= end_bin: " 
                 << orig_start << " > " << orig_end << std::endl;
        exit(1);
      }
      
      if (n_bins_win != (orig_end - orig_start + 1)) {
        std::cerr << "# ERROR: Inconsistent bin count: n_bins=" << n_bins_win 
                 << " but (end_bin - start_bin + 1)=" << (orig_end - orig_start + 1) << std::endl;
        exit(1);
      }
      
      WavelengthWindow window;
      window.r_min = r_min_win;
      window.r_max = r_max_win;
      window.orig_start_bin = orig_start;
      window.orig_end_bin = orig_end;
      window.n_bins = n_bins_win;
      window.dr = (r_max_win - r_min_win) / n_bins_win;
      
      wavelength_windows.push_back(window);
      
      std::cout << "  WAVE_RANGE: " << r_min_win << " - " << r_max_win << " Å "
               << "(bins " << orig_start << "-" << orig_end << ", n=" << n_bins_win << ")" << std::endl;
      
    } else if (!keyword.empty()) {
      std::cerr << "# WARNING: Unknown keyword in metadata: " << keyword << std::endl;
    }
  }
  fin.close();
  
  // validate required parameters
  if (!ni_set) {
    std::cerr << "# ERROR: 'ni' not specified in metadata file" << std::endl;
    exit(1);
  }
  if (!nj_set) {
    std::cerr << "# ERROR: 'nj' not specified in metadata file" << std::endl;
    exit(1);
  }
  if (!x_min_set || !x_max_set || !y_min_set || !y_max_set) {
    std::cerr << "# ERROR: Spatial coordinates not fully specified in metadata file" << std::endl;
    exit(1);
  }
  if (wavelength_windows.empty()) {
    std::cerr << "# ERROR: No wavelength ranges specified in metadata file" << std::endl;
    exit(1);
  }
  
  // sort wavelength windows by start bin index
  std::sort(wavelength_windows.begin(), wavelength_windows.end(),
            [](const WavelengthWindow& a, const WavelengthWindow& b) {
                return a.orig_start_bin < b.orig_start_bin;
            });
  
  // validate bin indices are sequential and non-overlapping
  validate_bin_indices();
  
  // process wavelength windows and calculate dimensions
  process_wavelength_windows();
  
  std::cout << "Metadata loaded successfully." << std::endl;
  std::cout << "Found " << wavelength_windows.size() << " wavelength windows, " 
            << nr << " total wavelength bins" << std::endl;

  // make sure maximum > minimum for spatial coordinates
  if (x_max <= x_min || y_max <= y_min)
    std::cerr<<"# ERROR: strange input in "<<metadata_file<<"."<<std::endl;

  data = read_cube(data_file);
  std::cout<<"Image Loaded...\n";

  var = read_cube(var_file);
  std::cout<<"Variance Loaded...\n";

  // check the desired emissions lines are located in the windows
  validate_emission_lines();

  /*
    Determine the valid data pixels
    Considered valid if sigma > 0.0 and there is at least 1 non-zero value.
   */
  valid.assign(1, std::vector<int>(2));
  double tmp_im, tmp_sig;
  std::vector<int> tmp_vec(2);
  nv = 0;
  for (size_t i=0; i<data.size(); i++) {
    for(size_t j=0; j<data[i].size(); j++) {
      tmp_im = 0.0;
      tmp_sig = 0.0;
      for (size_t r=0; r<data[i][j].size(); r++) {
        if (data[i][j][r] != 0.0) { tmp_im = 1.0; }
        tmp_sig += var[i][j][r];
	    }

      // Add valid pixels to array
      if ((tmp_im == 1.0) && (tmp_sig > 0.0)) {
        tmp_vec[0] = i;
        tmp_vec[1] = j;

        if (nv != 0)
          valid.push_back(tmp_vec);
        else
          valid[0] = tmp_vec;

        nv += 1;
      }
    }
  }
  std::cout<<"Valid pixels determined...\n\n";

  // Compute pixel widths
  dx = (x_max - x_min)/nj;
  dy = (y_max - y_min)/ni;
  // Note: dr is now calculated in process_wavelength_windows()
  
  for (size_t i=0; i<psf_sigma.size(); i++) {
    psf_sigma_overdx.push_back(psf_sigma[i]/dx);
    psf_sigma_overdy.push_back(psf_sigma[i]/dy);
  }

  // Calculate geometric widths
  pixel_width = sqrt(dx*dy);
  image_width = sqrt((x_max - x_min)*(y_max - y_min));

  // rc_max for TruncatedExponential distribution
  rc_max = sqrt(pow(abs(x_max - x_min), 2) + pow(abs(y_max - y_min), 2))/cos(inc);

  // Image centres
  x_imcentre = (x_min + x_max)/2.0;
  y_imcentre = (y_min + y_max)/2.0;

  // Array padding to help edge problems
  x_pad = (int)ceil(sigma_pad*psf_sigma[0]/dx);
  y_pad = (int)ceil(sigma_pad*psf_sigma[0]/dy);
  ni += 2*y_pad;
  nj += 2*x_pad;
  x_pad_dx = x_pad*dx;
  y_pad_dy = y_pad*dy;

  x_min -= x_pad_dx;
  x_max += x_pad_dx;
  y_min -= y_pad_dy;
  y_max += y_pad_dy;

  // Compute spatially oversampled parameters
  dxos = abs(dx)/sample;
  dyos = abs(dy)/sample;
  x_pados = (int)ceil(sigma_pad*psf_sigma[0]/dxos);
  y_pados = (int)ceil(sigma_pad*psf_sigma[0]/dyos);
  nios = sample*ni;
  njos = sample*nj;
  x_pad_dxos = x_pados*dxos;
  y_pad_dyos = y_pados*dyos;

  // Construct defaults that are dependent on data
  if (!radiuslim_min_flag) {
    radiuslim_min = pixel_width;
  }
  if (!gamma_pos_flag) {
    gamma_pos = 0.1*image_width;
  }

  // Compute x, y, r arrays
  compute_ray_grid();

  summarise_model();
}

void Data::process_wavelength_windows() {
    /*
        process wavelength windows using the provided bin indices
    */
    
    int total_wavelength_bins = 0;
    
    for (size_t w = 0; w < wavelength_windows.size(); w++) {
        auto& window = wavelength_windows[w];
        
        // set global indices for the combined array
        window.start_idx = total_wavelength_bins;
        window.end_idx = total_wavelength_bins + window.n_bins - 1;
        total_wavelength_bins += window.n_bins;
        
        // create wavelength array for this window
        window.r.resize(window.n_bins);
        
        for (int k = 0; k < window.n_bins; k++) {
            window.r[k] = window.r_min + (k + 0.5) * window.dr;
        }
        
        std::cout << "  Window " << w+1 << ": [" << window.r_min << ", " 
                 << window.r_max << "] Å (" << window.n_bins << " bins, " 
                 << "dr=" << window.dr << " Å/bin)" << std::endl;
    }
    
    nr = total_wavelength_bins;
    
    // create combined wavelength array
    r_full.clear();
    r_full.reserve(nr);
    for (const auto& window : wavelength_windows) {
        r_full.insert(r_full.end(), window.r.begin(), window.r.end());
    }
    
    // calculate overall dr for compatibility (weighted average)
    double total_range = wavelength_windows.back().r_max - wavelength_windows[0].r_min;
    dr = total_range / nr;
    
    std::cout << "Total wavelength coverage: " << wavelength_windows[0].r_min 
              << " - " << wavelength_windows.back().r_max << " Å" << std::endl;
    std::cout << "Average spectral resolution: " << dr << " Å/bin" << std::endl;
}

void Data::validate_bin_indices() {
    /*
        validate that bin indices are sequential and non-overlapping
    */
    
    int expected_start = 0;
    
    for (size_t w = 0; w < wavelength_windows.size(); w++) {
        const auto& window = wavelength_windows[w];
        
        if (window.orig_start_bin != expected_start) {
            std::cerr << "# ERROR: Non-sequential bin indices in window " << w+1 << std::endl;
            std::cerr << "Expected start_bin=" << expected_start 
                     << ", got start_bin=" << window.orig_start_bin << std::endl;
            std::cerr << "Windows must be sequential with no gaps or overlaps." << std::endl;
            exit(1);
        }
        
        expected_start = window.orig_end_bin + 1;
        
        std::cout << "  Validated window " << w+1 << ": bins " 
                 << window.orig_start_bin << "-" << window.orig_end_bin 
                 << " (" << window.n_bins << " bins)" << std::endl;
    }
}

void Data::validate_emission_lines() {
    /*
        validate that all emission lines fall within the defined wavelength windows
    */
    std::cout << "\nValidating emission lines against wavelength windows:" << std::endl;
    
    bool all_valid = true;
    
    for (size_t l = 0; l < em_line.size(); l++) {
        std::cout << "  Line group " << l+1 << ":" << std::endl;
        
        for (size_t i = 0; i < em_line[l].size(); i++) {
            double line_wave = em_line[l][i];
            bool found = false;
            
            for (size_t w = 0; w < wavelength_windows.size(); w++) {
                const auto& window = wavelength_windows[w];
                if (line_wave >= window.r_min && line_wave <= window.r_max) {
                    std::cout << "    Line " << line_wave << " Å: ✓ Found in window " 
                             << w+1 << " [" << window.r_min << ", " << window.r_max << "] Å" << std::endl;
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                std::cerr << "    Line " << line_wave << " Å: ✗ NOT FOUND in any wavelength window!" << std::endl;
                all_valid = false;
            }
        }
    }
    
    if (!all_valid) {
        std::cerr << "\n# ERROR: Some emission lines are outside the wavelength windows!" << std::endl;
        std::cerr << "Either:" << std::endl;
        std::cerr << "  1. Add more wave_range entries to metadata.txt (and subsequently make sure that is reflected in data.txt), or" << std::endl;
        std::cerr << "  2. Remove/modify LINE entries in MODEL_OPTIONS" << std::endl;
        exit(1);
    }
    
    std::cout << "✓ All emission lines validated successfully.\n" << std::endl;
}

std::vector< std::vector< std::vector<double> > > Data::arr_3d() {
  std::vector< std::vector< std::vector<double> > > arr;
  // Create 3D array with shape (ni, nj, nr)
  arr.resize(ni);
  for (int i=0; i<ni; i++) {
    arr[i].resize(nj);
    for(int j=0; j<nj; j++) {
      arr[i][j].resize(nr);
    }
  }

  return arr;
}

std::vector< std::vector< std::vector<double> > >
  Data::read_cube (std::string filepath) {
  // Read data file
  std::vector< std::vector< std::vector<double> > > cube = arr_3d();
  std::fstream fin(filepath, std::ios::in);

  if (!fin)
    std::cerr<<"# ERROR: couldn't open file "<<filepath<<"."<<std::endl;

  for (size_t i=0; i<cube.size(); i++)
    for (size_t j=0; j<cube[i].size(); j++)
      for (size_t r=0; r<cube[i][j].size(); r++)
        fin >> cube[i][j][r];
  fin.close();

  return cube;
}

void Data::compute_ray_grid() {
  // Make vectors of the correct size
  x.assign(ni, std::vector<double>(nj));
  y.assign(ni, std::vector<double>(nj));

  for (size_t i=0; i<x.size(); i++) {
    for (size_t j=0; j<x[i].size(); j++) {
      x[i][j] = x_min + (j + 0.5)*dx;
      y[i][j] = y_min + (i + 0.5)*dy; // Assuming origin=lower
    }
  }

  // UPDATED: use the combined wavelength array from all windows
  r.assign(nr, 0.0);
  for(size_t k=0; k<r.size(); k++)
    r[k] = r_full[k];  // Use the combined array from all windows
}

void Data::summarise_model() {
  // Print summarised model to terminal
  std::string dashline =
    "-----------------------------------------------------------";

  std::cout<<dashline<<std::endl;
  std::cout<<"Joint Prior Distribution"<<std::endl;

  std::cout<<dashline<<std::endl;
  std::cout<<"Constants"<<std::endl;
  std::cout<<dashline<<std::endl;

  std::cout<<"Emission Line(s):"<<std::endl;
  for (size_t l=0; l<em_line.size(); l++) {
    std::cout<<"  Line: "<<em_line[l][0]<<std::endl;
    for (size_t i=0; i<(em_line[l].size() - 1)/2; i++) {
      std::cout<<"   Constrained Line: "<<em_line[l][2*i+1];
      std::cout<<", Factor: "<<em_line[l][2*i+2];
      std::cout<<std::endl;
    }
  }

  // display wavelength window information
  std::cout<<"Wavelength Windows:"<<std::endl;
  for (size_t w = 0; w < wavelength_windows.size(); w++) {
    const auto& window = wavelength_windows[w];
    std::cout<<"  Window "<<w+1<<": "<<window.r_min<<" - "<<window.r_max<<" Å ("
             <<window.n_bins<<" bins)"<<std::endl;
  }
  std::cout<<"Total wavelength bins: "<<nr<<std::endl;

  if (nfixed)
    std::cout<<"N: "<<nmax<<std::endl;
  std::cout<<"PSF Profile: ";
  if (convolve == 0) {
    std::cout<<"Sum of concentric Gaussians."<<std::endl;
    std::cout<<"PSF_WEIGHTS: ";
    for (size_t i=0; i<psf_amp.size(); i++)
      std::cout<<psf_amp[i]<<" ";
    std::cout<<std::endl;
  } else if (convolve == 1) {
    std::cout<<"Moffat (WARNING: Not safe for multi-threading)."<<std::endl;
    std::cout<<"PSF_BETA: "<<psf_beta<<std::endl;
  }
  std::cout<<"PSF_FWHM: ";
  for (size_t i=0; i<psf_fwhm.size(); i++)
    std::cout<<psf_fwhm[i]<<" ";
  std::cout<<std::endl;
  std::cout<<"LSF_FWHM (Gauss Instr. Broadening): "<<lsf_fwhm<<std::endl;
  std::cout<<"i: "<<inc<<std::endl;

  std::cout<<dashline<<std::endl;
  std::cout<<"Global Parameters"<<std::endl;
  std::cout<<dashline<<std::endl;
  std::cout
    <<"x_c ~ Cauchy("<<x_imcentre<<", "<<gamma_pos<<")"
    <<"T("<<x_min + x_pad_dx<<", "<<x_max - x_pad_dx<<")"
    <<std::endl;
  std::cout
    <<"y_c ~ Cauchy("<<x_imcentre<<", "<<gamma_pos<<")"
    <<"T("<<y_min + y_pad_dy<<", "<<y_max - y_pad_dy<<")"
    <<std::endl;

  std::cout<<"Theta ~ Uniform(0, 2pi)"<<std::endl;

  if (!nfixed)
    std::cout<<"N ~ Loguniform(0, "<<nmax<<")"<<std::endl;

  std::cout<<dashline<<std::endl;
  std::cout<<"Blob hyperparameters"<<std::endl;
  std::cout<<dashline<<std::endl;
  std::cout<<"mu_r ~ Loguniform("<<wd_min<<", "<<wd_max<<")"<<std::endl;
  std::cout<<"mu_F ~ Loguniform("<<fluxmu_min<<", "<<fluxmu_max<<")"<<std::endl;
  std::cout<<"sigma_F ~ Loguniform("<<lnfluxsd_min<<", "<<lnfluxsd_max<<")"<<std::endl;
  std::cout<<"W_max ~ Loguniform("<<radiuslim_min<<", "<<radiuslim_max<<")"<<std::endl;
  std::cout<<"q_min ~ Uniform("<<qlim_min<<", "<<"1)"<<std::endl;

  std::cout<<dashline<<std::endl;
  std::cout<<"Blob parameters"<<std::endl;
  std::cout<<dashline<<std::endl;
  std::cout<<"F_j ~ Lognormal(mu_F, sigma_F^2)"<<std::endl;
  std::cout<<"r_j ~ Exponential(mu_r)"<<std::endl;
  std::cout<<"Theta_j ~ Uniform(0, 2pi)"<<std::endl;
  std::cout<<"w_j ~ Loguniform("<<radiuslim_min<<", W_max)"<<std::endl;
  std::cout<<"q_j ~ Triangular(q_min, 1)"<<std::endl;
  std::cout<<"phi_j ~ Uniform(0, pi)"<<std::endl;

  std::cout<<dashline<<std::endl;
  std::cout<<"Velocity profile parameters"<<std::endl;
  std::cout<<dashline<<std::endl;
  std::cout<<"v_sys ~ Cauchy(0, 30)T(-"<<vsys_max<<", "<<vsys_max<<")"<<std::endl;
  std::cout<<"v_c ~ Loguniform("<<vmax_min<<", "<<vmax_max<<")"<<std::endl;
  std::cout<<"r_t ~ Loguniform("<<vslope_min<<", "<<vslope_max<<")"<<std::endl;
  std::cout<<"gamma_v ~ Loguniform("<<vgamma_min<<", "<<vgamma_max<<")"<<std::endl;
  std::cout<<"beta_v ~ Uniform("<<vbeta_min<<", "<<vbeta_max<<")"<<std::endl;

  std::cout<<dashline<<std::endl;
  std::cout<<"Velocity dispersion profile parameters"<<std::endl;
  std::cout<<dashline<<std::endl;
  std::cout<<"sigma_v0 ~ Loguniform("<<vdisp0_min<<", "<<vdisp0_max<<")"<<std::endl;
  std::cout<<"sigma_vn ~ Normal(0, "<<vdispn_sigma<<")"<<std::endl;

  std::cout<<dashline<<std::endl;
  std::cout<<"Systematic noise parameters"<<std::endl;
  std::cout<<dashline<<std::endl;
  std::cout<<"sigma_0 ~ Loguniform("<<sigma_min<<", "<<sigma_max<<")"<<std::endl;

  std::cout<<dashline<<std::endl;
  std::cout<<std::endl;
}