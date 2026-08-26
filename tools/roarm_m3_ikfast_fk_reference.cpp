// Offline reference wrapper around Waveshare's generated RoArm-M3 IKFast solver.
// Build example:
//   g++ -O2 -std=c++11 -DIKFAST_NO_MAIN \
//     -I<official-plugin>/include \
//     -I<official-plugin>/src \
//     tools/roarm_m3_ikfast_reference.cpp \
//     <official-plugin>/src/roarm_m3_hand_ikfast_solver.cpp -o /tmp/roarm_m3_ikfast_reference
// Input: one canonical "base shoulder elbow wrist roll" row per line.
// Output: official FK translation (metres) followed by the tool +Z direction.
// The generated solver is TranslationDirection5D, so ComputeFk returns three
// direction values rather than a 3x3 rotation matrix.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "ikfast.h"

typedef double IkReal;

extern int GetNumJoints();
extern void ComputeFk(const IkReal* joints, IkReal* translation, IkReal* rotation);
extern bool ComputeIk(
    const IkReal* translation,
    const IkReal* rotation,
    const IkReal* free_parameters,
    ikfast::IkSolutionListBase<IkReal>& solutions);

static double wrapped_error(double value, double reference) {
  return std::remainder(value - reference, 2.0 * M_PI);
}

int main() {
  if (GetNumJoints() != 5) {
    std::cerr << "expected 5 IKFast joints\n";
    return 2;
  }
  std::cout << std::setprecision(17);
  IkReal q[5];
  while (std::cin >> q[0] >> q[1] >> q[2] >> q[3] >> q[4]) {
    IkReal translation[3];
    IkReal direction[3];
    ComputeFk(q, translation, direction);
    std::cout << translation[0] << ' ' << translation[1] << ' ' << translation[2];
    for (int index = 0; index < 3; ++index) {
      std::cout << ' ' << direction[index];
    }
    std::cout << '\n';
  }
  return 0;
}
