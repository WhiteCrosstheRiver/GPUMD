// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

class GNEP;
class Parameters;
struct GNEPTrainingBatch;

// Experimental projection of the output layer over the device shards of one batch.
void fcvp_project(
  Parameters& para, GNEP& potential, GNEPTrainingBatch& batch, float* parameters, int step);

// Finite-difference gate for the loss_rmse gradients on one batch.
void fcvp_verify_gradient(
  Parameters& para, GNEP& potential, GNEPTrainingBatch& batch, float* parameters);
