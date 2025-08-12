#include <iostream>
#include <ctime>

#include "DNest4/code/DNest4.h"

#include "Data.h"
#include "DiscModel.h"

int main(int argc, char** argv) {
  // clock_t begin = clock();

  // Use wider tails randh
  DNest4::RNG::randh_is_randh2 = true;

  // Get command line options
  DNest4::CommandLineOptions options(argc, argv);

  // Get model specific options file
  const char* moptions_file;
  if (options.get_config_file() == "") {
    std::cerr
      <<"# ERROR: No model options file provided. "
	    <<"Specify model options file as a command line option "
	    <<"using -f <MODEL_OPTIONS_FILE>."
      <<std::endl;
    exit(0);
  } else {
    moptions_file = options.get_config_file().c_str();
  }

  // Load data
  Data::get_instance().load(moptions_file);

  // Setup and run sampler
  DNest4::Sampler<DiscModel> sampler = DNest4::setup<DiscModel>(options);
  sampler.run();

  // clock_t end = clock();
  // double elapsed_secs = double(end - begin)/CLOCKS_PER_SEC;
  // std::cout<<"TIME: "<<elapsed_secs<<std::endl;

  return 0;
}
