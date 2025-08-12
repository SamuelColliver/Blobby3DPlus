#include "LookupExp.h"

#include <cmath>
#include <cassert>

LookupExp LookupExp::instance;

LookupExp::LookupExp()
    :num(1251)
    ,xMin(0.0)
    ,xMax(12.5)
    ,dx((xMax - xMin)/(num-1))
    ,one_over_dx(1.0/dx)
    ,_exp(num) {

  for(int i=0; i<num; i++)
    _exp[i] = exp(-i*dx);
}

double LookupExp::evaluate(double x) {
  int i = static_cast<int>(x/instance.dx);
  double frac = (x - i*instance.dx)*instance.one_over_dx;

  if (i < 0 || i >= (static_cast<int>(instance._exp.size()) - 1))
    return 0.0;
  else
    return frac*instance._exp[i+1] + (1.0 - frac)*instance._exp[i];
}
