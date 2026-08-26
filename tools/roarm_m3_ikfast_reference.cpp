// Offline reference wrapper around Waveshare's generated RoArm-M3 IKFast solver.
// Build example:
//   g++ -O2 -std=c++11 -DIKFAST_NO_MAIN \
//     -I<official-plugin>/include \
//     -I<official-plugin>/src \
//     tools/roarm_m3_ikfast_reference.cpp \
//     <official-plugin>/src/roarm_m3_hand_ikfast_solver.cpp -o /tmp/roarm_m3_ikfast_reference
// Input: one canonical "base shoulder elbow wrist roll" row per line.
// Output: solution_count and the nearest wrapped joint-space error.

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
    IkReal rotation[9];
    ComputeFk(q, translation, rotation);
    ikfast::IkSolutionList<IkReal> solutions;
    const bool ok = ComputeIk(translation, rotation, NULL, solutions);
    double nearest = std::numeric_limits<double>::infinity();
    if (ok) {
      std::vector<IkReal> values(5);
      for (std::size_t i = 0; i < solutions.GetNumSolutions(); ++i) {
        const ikfast::IkSolutionBase<IkReal>& solution = solutions.GetSolution(i);
        std::vector<IkReal> free_values(solution.GetFree().size());
        solution.GetSolution(values.data(), free_values.empty() ? NULL : free_values.data());
        double squared = 0.0;
        for (int joint = 0; joint < 5; ++joint) {
          const double error = wrapped_error(values[joint], q[joint]);
          squared += error * error;
        }
        nearest = std::min(nearest, std::sqrt(squared));
      }
    }
    std::cout << (ok ? 1 : 0) << ' ' << solutions.GetNumSolutions() << ' ' << nearest << ' '
              << translation[0] << ' ' << translation[1] << ' ' << translation[2] << '\n';
  }
  return 0;
}
