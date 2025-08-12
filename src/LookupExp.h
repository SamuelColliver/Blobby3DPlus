#ifndef BLOBBY3D_LOOKUPEXP_H_
#define BLOBBY3D_LOOKUPEXP_H_

#include <vector>

/*
* Lookup tables for speeding things up
* Singleton pattern
*/
class LookupExp {
  private:
    int num;
    double xMin, xMax, dx, one_over_dx;
    std::vector<double> _exp; // exp(-x) for x >= 0

    LookupExp();
    LookupExp(const LookupExp& other);

    static LookupExp instance;

    public:
      static double evaluate(double x);
};

#endif  // BLOBBY3D_LOOKUPEXP_H_

