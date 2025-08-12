#ifndef BLOBBY3D_LOOKUPERF_H_
#define BLOBBY3D_LOOKUPERF_H_

#include <vector>

/*
* Lookup tables for speeding things up
* Singleton pattern
*/

class LookupErf {
  private:
    int num;
    double xMin, xMax, dx, one_over_dx;
    std::vector<double> _erf; // exp(-x) for x >= 0

    LookupErf();
    LookupErf(const LookupErf& other);

    static LookupErf instance;

  public:
    static double evaluate(double x);
};

#endif  // BLOBBY3D_LOOKUPERF_H_

