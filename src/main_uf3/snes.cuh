/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Separable Natural Evolution Strategy (SNES) — adapted from main_nep.
Maintains a Gaussian search distribution N(mu, sigma) and updates it using
natural gradient of the fitness ranking.
Ref: Schaul et al., GECCO 2011, https://doi.org/10.1145/2001576.2001692
------------------------------------------------------------------------------*/

#pragma once
#include "fitness.cuh"

void run_snes(UF3_Parameters& para, Uf3Fitness& fitness);
