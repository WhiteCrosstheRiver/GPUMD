// SPDX-License-Identifier: GPL-3.0-or-later

/*----------------------------------------------------------------------------80
Output-layer projection of the current training batch across device shards.
------------------------------------------------------------------------------*/

#pragma once

#include <vector>

class GNEP;
class Parameters;
class Dataset;

// Experimental projection of the current training batch across device shards.
void fcvp_project(
  Parameters& para, GNEP& potential, std::vector<Dataset>& dataset, float* parameters, int step);

void fcvp_verify_gradient(
  Parameters& para, GNEP& potential, std::vector<Dataset>& dataset, float* parameters);
