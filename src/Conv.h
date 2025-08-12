#ifndef BLOBBY3D_CONV_H_
#define BLOBBY3D_CONV_H_

#include <vector>
#include <fftw3.h>

/*
  Class for convolving by the PSF
*/
class Conv {
 private:
  // convolve method
  int convolve;

  // PSF parameters
  std::vector<double> psf_amp;
  std::vector<double> psf_fwhm;
  double psf_beta;
  std::vector<double> psf_sigma;
  std::vector<double> psf_sigma_overdx, psf_sigma_overdy;
  int ni, nj, nr;
  double dx, dy;
  int x_pad, y_pad;
  double sigma_cutoff;

  // vectors for separable kernel
  std::vector< std::vector<double> > kernel_x;
  std::vector< std::vector<double> > kernel_y;
  std::vector< std::vector<double> > convolved_tmp_2d;

  // vector for moffat kernel
  std::vector< std::vector<double> > kernel;

  // fftw pointers/plans
  fftw_complex *out, *kernelout, *conv;
  fftw_plan p, q, k;
  double* in;
  double* in2;
  int Ni, Nj;
  int nik, njk;
  int midik, midjk;

  // Convolved matrix
  std::vector< std::vector< std::vector<double> > > convolved;

  /*
    Convolution Methods
  */
  // brute force gaussian blur
  std::vector< std::vector< std::vector<double> > > brute_gaussian_blur(
      std::vector< std::vector< std::vector<double> > >& preconvolved);

  // fftw moffat blur
  std::vector< std::vector< std::vector<double> > > fftw_moffat_blur(
      std::vector< std::vector< std::vector<double> > >& preconvolved);

  // Constructor
  // static Conv instance;

 public:
  Conv(
    int convolve,
    std::vector<double> psf_amp,
    std::vector<double> psf_fwhm,
    double psf_beta,
    std::vector<double> psf_sigma,
    std::vector<double> psf_sigma_overdx,
    std::vector<double> psf_sigma_overdy,
    int ni,
    int nj,
    int nr,
    double dx,
    double dy,
    double x_pad,
    double y_pad
    );

  // Apply convolution by implied method passed to class constructor.
  std::vector< std::vector< std::vector<double> > > apply(
      std::vector< std::vector< std::vector<double> > >& preconvolved);
};

#endif  // BLOBBY3D_CONV_H_
