# Blobby3D

Blobby3D is designed to perform Bayesian inference for gas kinematics on emission line observations of galaxies using Integral Field Spectroscopy. The code was designed with two main goals in mind:

 - To robustly infer gas kinematics for regularly rotating galaxies even if the gas profiles have significant substructure.
 - Infer gas kinematic properties free from the effects of beam smearing. Where beam smearing is the effect of the observational seeing spatially blurring the gas profiles. This has significant effects on the observed gas kinematic properties - particularly the observed velocity dispersion.

You can find the motivation and first implementation of Blobby3D in this [paper](https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.4024V/abstract). Please cite this paper if you use the code in your research.

## Getting Started

These instructions will get you a copy of the project up and running on your machine for testing purposes. If you have any trouble, feel free to contact Mathew Varidel at mathew.varidel@sydney.edu.au.

### Prerequisites

Blobby3D is dependent on the following:

C++11\
[DNest4](https://github.com/eggplantbren/DNest4)  \
[FFTW3](http://www.fftw.org)  \
[Python 3](https://www.python.org)

[FFTW3](http://www.fftw.org) will typically be available on scientific machines/clusters, so you will often have it or can get it relatively easily by contacting your system administrator.

[DNest4](https://github.com/eggplantbren/DNest4) is a bespoke nested sampling algorithm written by Brendon Brewer. The code will need to be cloned from github in the usual manner. It is only available for Unix like machines.

### Compilling

To install Blobby3D you need to clone this repository. You then need to compile the C++ code and install the python module.

To compile the C++ code type 'make' in the root directory. Note that this assumes that you have setup a DNEST4_PATH environment variable as suggested in the [DNEST4 paper](https://arxiv.org/abs/1606.03757). If you have not setup a DNEST4_PATH environment variable, then simply make using the following:

make DNEST4_PATH=/path/to/DNest4

Where /path/to/DNest4 is the root directory for DNest4 on your machine.

To install the python module you can just do the following:

pip install pyblobby3d

The python module is then available using:

import pyblobby3d

### Running Blobby3D on Examples

To get a feel for Blobby3D you will want to start with an example. There is an example using GAMA 485885 in the examples folder. The code can be run as an executable from within the subdirectory as follows:

../../Blobby3D -t 1 -f MODEL_OPTIONS

You may want to setup a terminal alias such that you can run the executable from anywhere on your machine. The -t 1 parameter sets the number of threads. Blobby3D is computationally expensive, so you should run it using multiple threads. The -f MODEL_OPTIONS tells the model to use the input hyperparameters that define the priors. Note the DNest4 parameters are defined in the OPTIONS file (see [DNest4](https://github.com/eggplantbren/DNest4) documentation for that). Option files are provided in the examples.

There will be 3 output files created as Blobby3D runs:

sample_info.txt\
sample.txt\
sampler_state.txt

At any time during or after the run, you can perform postprocessing of the current Blobby3D output. An example postprocessing script is available in the examples folders labeled post.py.

### Running Your Own Data

Blobby3D requires several files to run. Blobby3D accepts three data files typically named data.txt, var.txt, and metadata.txt. data.txt and var.txt correspond to the data and variance cubes. The file format is in whitespace separated values of (x, y, wavelength) represented in row-major format. The metadata format describes the data width given by whitespace separated (x, y, wavelength) bins, followed by minimum, maximum values of (x, y, wavelength). The minimum and maximum values are the left-most and right-most edge of each array. The data is assumed to be de-redshifted and centred about (0, 0) spatial coordinates.

There is also a MODEL_OPTIONS file that describes model parameterisation options. Note that the default parameterisation can be found in the [paper](https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.4024V/abstract). It has the following required parameters:

LSFFWHM : float\
&nbsp;&nbsp;Line Spread Function Full-Width Half-Maximum measured in Angstroms.\
PSFWEIGHT : floats\
&nbsp;&nbsp;Point Spread Function weights, which are the integral of each Gaussian and must sum to 1.\
PSFFWHM : floats\
&nbsp;&nbsp;Point Spread Function Full-Width Half-Maximum. The PSF is modelled a sum of concentric Gaussians described with pairs of (PSFWEIGHT, PSFFWHM).\
INC : float\
&nbsp;&nbsp;Inclination in radians. INC = 0 for a face-on galaxy.\
LINE : float, or list of floats\
&nbsp;&nbsp;Vacuum wavelength in Angstroms for a line to be modelled. This can also describe coupled lines, for example for the [NII] lines around H-alpha it can be described as:\
&nbsp;&nbsp;&nbsp;&nbsp;LINE	6583.1	6548.1	0.3333\
&nbsp;&nbsp;This couples the [NII] line at 6583.1 A to the line at 6548.1 A assuming that the second line has a known ratio to the first of 0.3333. Triplets (or more) can also be described by adding an extra pair similar to the second [NII] line above.

The following allows you to change the hyperparameters with defaults as set in the paper:

NMAX : int, default is 300\
&nbsp;&nbsp;Maximum number of blobs allowed\
NFIXED : int, default is False\
&nbsp;&nbsp;Whether the number of blobs is allowed to change. For example, if you want to run a single component model you can easily do NMAX as 1 and NFIXED as TRUE.\
WD_MIN : float, default is 0.03\
&nbsp;&nbsp;Minimum width for typical distance (mu_r in the paper). Value is given in the same units as (x, y) coordinates.\
WD_MAX: float, default is 30 in (x, y) units. Value is given in the same units as (x, y) coordinates.\
&nbsp;&nbsp;Minimum width for typical distance (mu_r in the paper).\
MEANFLUX_MIN : float, default is 1E-3\
&nbsp;&nbsp;Minimum for typical flux of blobs (mu_F in the paper).\
MEANFLUX_MAX : float, default is 1E3\
&nbsp;&nbsp;Maximum for typical flux of blobs (mu_F in the paper).
RADIUSLIM_MAX : float, default is 30 in (x, y) units\
&nbsp;&nbsp;Maximum value in the prior for the maximum allowed width for blobs.\
VSYS_MAX : float, default is 150\
&nbsp;&nbsp;Symmetrical maximum allowed systemic velocity in km/s.
VMAX_MIN : float, default is 40\
&nbsp;&nbsp;Minimum allowed asymptotic velocity.\
VMAX_MAX : float, default is 400\
&nbsp;&nbsp;Maximum allowed asymptotic velocity.\
VSLOPE_MIN : float, default is 0.03\
&nbsp;&nbsp;Minimum allowed turnover radius. Value is given in the same units as (x, y) coordinates.\
VSLOPE_MAX : float, default is 30\
&nbsp;&nbsp;Maximum allowed turnover radius. Value is given in the same units as (x, y) coordinates.\
VGAMMA_MIN : float, default is 1\
&nbsp;&nbsp;Minimum allowed gamma_v shape parameter.\
VGAMMA_MAX : float, default is 100\
&nbsp;&nbsp;Maximum allowed gamma_v shape parameter.\
VBETA_MIN : float, default is -0.75\
&nbsp;&nbsp;Minimum allowed beta_v shape parameter.\
VBETA_MAX : float, default is 0.75\
&nbsp;&nbsp;Maximum allowed beta_v shape parameter.\
VDISP_ORDER : int, default is 1\
&nbsp;&nbsp;Velocity dispersion profile polynomial order.\
LOGVDISP0_MIN : float, default is 0.\
&nbsp;&nbsp;Log minimum velocity dispersion for zeroth order moment of the polynomial.\
LOGVDISP0_MAX : float, default is log(200)\
&nbsp;&nbsp;Log maximum velocity dispersion for zeroth order moment of the polynomial.\
VDISPN_SIGMA : float, default is 0.2\
&nbsp;&nbsp;Width for normal prior for the log velocity dispersion gradient.\
