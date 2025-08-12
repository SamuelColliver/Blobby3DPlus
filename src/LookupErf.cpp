#include "LookupErf.h"

#include <cmath>
#include <cassert>
#include <iostream>

LookupErf LookupErf::instance;

LookupErf::LookupErf()
    :num(4001)
    ,xMin(-4.0)
    ,xMax(4.0)
    ,dx((xMax - xMin)/(num-1))
    ,one_over_dx(1.0/dx)
    ,_erf(num) {
  for(int i=0; i<num; i++)
    _erf[i] = erf(xMin + i*dx);
}

double LookupErf::evaluate(double x) {
  int i = static_cast<int>((x - instance.xMin)/instance.dx);
  double frac = (x - instance.xMin - i*instance.dx)*instance.one_over_dx;

  if (i < 0)
    return -1.0;
  else if (i >= (static_cast<int>(instance._erf.size()) - 1))
    return 1.0;
  else
    return frac*instance._erf[i+1] + (1.0 - frac)*instance._erf[i];
}
