// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <vector>

class GNEP;
class Parameters;
class Dataset;

// Experimental single-device projection of the current TRAINING batch only.
void fcvp_project(
  Parameters& para, GNEP& potential, std::vector<Dataset>& dataset, float* parameters, int step);

void fcvp_verify_gradient(
  Parameters& para, GNEP& potential, std::vector<Dataset>& dataset, float* parameters);
