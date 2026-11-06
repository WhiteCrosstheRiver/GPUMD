// SPDX-License-Identifier: GPL-3.0-or-later

/*----------------------------------------------------------------------------80
Moment-space Levenberg-Marquardt training interface.
------------------------------------------------------------------------------*/

#pragma once
#include "dataset.cuh"
#include <memory>
#include <vector>

class GNEP;
class Parameters;

class NEP_LM
{
public:
  NEP_LM(Parameters& para, GNEP& potential);
  ~NEP_LM();
  Batch_MSE update(
    Parameters& para,
    GNEP& potential,
    std::vector<Dataset>& data,
    std::vector<Dataset>& guard,
    float* parameters,
    int step);

private:
  struct State;
  std::unique_ptr<State> state;
};
