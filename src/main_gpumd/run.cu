/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Run simulation according to the inputs in the run.in file.
------------------------------------------------------------------------------*/

#include "cohesive.cuh"
#include "force/force.cuh"
#include "integrate/ensemble.cuh"
#include "integrate/integrate.cuh"
#include "measure/measure.cuh"
#include "minimize/minimize.cuh"
#include "model/box.cuh"
#include "model/read_xyz.cuh"
#include "phonon/hessian.cuh"
#include "replicate.cuh"
#include "run.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include "atom_mutation.cuh"
#include "control.cuh"
#include "delete.cuh"
#include "variable.cuh"
#include "utilities/nep_io/compact_nep.cuh"
#include "utilities/run_input.cuh"
#include "velocity.cuh"
#include <chrono>

static __global__ void gpu_find_largest_v2(
  int N, int number_of_rounds, double* g_vx, double* g_vy, double* g_vz, double* g_v2_max)
{
  int tid = threadIdx.x;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;
  for (int round = 0; round < number_of_rounds; ++round) {
    int n = round * 1024 + tid;
    if (n < N) {
      double vx = g_vx[n];
      double vy = g_vy[n];
      double vz = g_vz[n];
      double v2 = vx * vx + vy * vy + vz * vz;
      if (s_data[tid] < v2) {
        s_data[tid] = v2;
      }
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_data[tid] < s_data[tid + offset]) {
        s_data[tid] = s_data[tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_v2_max[0] = s_data[0];
  }
}

__device__ double device_v2_max[1];

static void calculate_time_step(
  double max_distance_per_step,
  GPU_Vector<double>& velocity_per_atom,
  double initial_time_step,
  double& time_step)
{
  if (max_distance_per_step <= 0.0) {
    return;
  }
  const int N = velocity_per_atom.size() / 3;
  double* gpu_v2_max;
  CHECK(gpuGetSymbolAddress((void**)&gpu_v2_max, device_v2_max));
  gpu_find_largest_v2<<<1, 1024>>>(
    N,
    (N - 1) / 1024 + 1,
    velocity_per_atom.data(),
    velocity_per_atom.data() + N,
    velocity_per_atom.data() + N * 2,
    gpu_v2_max);
  GPU_CHECK_KERNEL
  double cpu_v2_max[1] = {0.0};
  CHECK(gpuMemcpy(cpu_v2_max, gpu_v2_max, sizeof(double), gpuMemcpyDeviceToHost));
  double cpu_v_max = sqrt(cpu_v2_max[0]);
  double time_step_min = max_distance_per_step / cpu_v_max;

  if (time_step_min < initial_time_step) {
    time_step = time_step_min;
  } else {
    time_step = initial_time_step;
  }
}

static bool parse_initial_replicate(
  const RunInput& run_input, int replicate_size[3])
{
  for (const auto& line : run_input.lines()) {
    if (line.tokens.empty()) {
      continue;
    }
    if (line.tokens[0] != "replicate") {
      return false;
    }
    parse_replicate(line.tokens, replicate_size);
    return true;
  }
  return false;
}

Run::Run(const RunInput& run_input)
{
  print_line_1();
  printf("Started initializing positions and related parameters.\n");
  fflush(stdout);
  print_line_2();

  has_replicate_ = parse_initial_replicate(run_input, replicate_size_);
  register_named_deposit_species(run_input);

  initialize_position(
    run_input,
    has_velocity_in_xyz,
    number_of_types,
    box,
    group,
    atom);
  first_potential_filename_ = get_first_potential_filename(run_input);

  if (has_replicate_) {
    Replicate(replicate_size_, box, atom, group);
  }

  if (atom.number_of_atoms < 2) {
    PRINT_INPUT_ERROR("Number of atoms should >= 2.");
  }

  velocity.initialize_cpu(
    has_velocity_in_xyz, 300, atom, false, 123);

  if (has_velocity_in_xyz) {
    printf("Initialized velocities with data in model.xyz.\n");
  } else {
    printf("Initialized velocities with default T = 300 K.\n");
  }

  allocate_memory_gpu(group, atom, thermo);

  print_line_1();
  printf("Finished initializing positions and related parameters.\n");
  fflush(stdout);
  print_line_2();

  execute_run_in(run_input);
}

// Ensembles and measurements are built for the current atom count when they are parsed.
void Run::check_atom_count_change() const
{
  const EnsembleType type = integrate.get_type();
  if (integrate.has_ensemble() && type != EnsembleType::NVE && !is_standard_nvt(type)) {
    PRINT_INPUT_ERROR("Put deposit and delete before this ensemble in a run block.");
  }
  for (const std::string& name : measure.action_names()) {
    if (name != "dump_xyz" && name != "dump_thermo" && name != "dump_restart") {
      PRINT_INPUT_ERROR("Put deposit and delete before computes and dumps in a run block.");
    }
  }
}

void Run::delete_isolated_batch(const std::vector<std::vector<std::string>>& commands)
{
  check_atom_count_change();
  AtomMutation::sync_cpu_from_gpu(atom);
  DeleteIsolatedSequence(commands, box, atom, group, thermo, force);
  integrate.update_fix_union(atom);
}

static __global__ void gpu_deposit_entry(
  int N, int first, int count, double nx, double ny, double nz, double height,
  const double* position, int* entered, int* status)
{
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= count) {
    return;
  }
  const int i = first + j;
  const double x = position[i], y = position[i + N], z = position[i + 2 * N];
  if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
    atomicExch(status + 1, 1);
    return;
  }
  if (x * nx + y * ny + z * nz <= height + 1.0e-10) {
    entered[j] = 1;
  }
  if (!entered[j]) {
    atomicAdd(status, 1);
  }
}

bool Run::update_deposit_entry()
{
  int status[2] = {0, 0};
  deposit_entry_status.copy_from_host(status);
  gpu_deposit_entry<<<(deposit_event.count + 127) / 128, 128>>>(
    atom.number_of_atoms,
    deposit_event.first_atom,
    deposit_event.count,
    deposit_event.normal[0],
    deposit_event.normal[1],
    deposit_event.normal[2],
    deposit_event.entry_height,
    atom.position_per_atom.data(),
    deposit_entered.data(),
    deposit_entry_status.data());
  GPU_CHECK_KERNEL
  deposit_entry_status.copy_to_host(status);
  if (status[1]) {
    PRINT_INPUT_ERROR("Non-finite coordinate during deposit pulse entry check.");
  }
  return status[0] == 0;
}

void Run::validate_deposit_dynamics(const bool pulse) const
{
  const EnsembleType type = integrate.get_type();
  if (!integrate.has_ensemble() || (type != EnsembleType::NVE && !is_standard_nvt(type))) {
    PRINT_INPUT_ERROR("deposit flight skip/pulse requires an NVE or constant-temperature NVT ensemble.");
  }
  if (type != EnsembleType::NVE && integrate.get_temperature1() != integrate.get_temperature2()) {
    PRINT_INPUT_ERROR("deposit flight skip/pulse does not support a temperature ramp.");
  }
  if (velocity.do_velocity_correction || integrate.get_move_group() >= 0) {
    PRINT_INPUT_ERROR("deposit flight skip/pulse does not support velocity correction or moving groups.");
  }
  if (!std::isfinite(time_step) || time_step <= 0.0) {
    PRINT_INPUT_ERROR("deposit flight skip/pulse requires a positive finite MD time step.");
  }
  for (const std::string& name : measure.action_names()) {
    if (
      name == "add_efield" || name == "add_force" || name == "add_random_force" ||
      name == "add_spring" || name == "electron_stop" || name == "mc" || name == "deform" ||
      name == "compute_hnemd" || name == "compute_hnemdec" || name == "enhanced_sampling" ||
      name == "plumed") {
      PRINT_INPUT_ERROR("deposit flight skip/pulse does not support driven dynamics.");
    }
    if (pulse && name != "dump_xyz" && name != "dump_thermo" && name != "dump_restart") {
      PRINT_INPUT_ERROR(
        "deposit pulse supports only dump_xyz, dump_thermo and dump_restart. Use emit plus run.");
    }
  }
}

void Run::parse_deposit(const std::vector<std::string>& tokens)
{
  check_atom_count_change();
  const bool execution = tokens.size() >= 3 && (tokens[2] == "emit" || tokens[2] == "pulse");
  const bool pulse = execution && tokens[2] == "pulse";
  const std::string name = !execution ? std::string{}
                           : variables ? variables->expand_text(tokens[1], box)
                                       : tokens[1];
  const bool skip = execution && deposit_registry.skips_flight(name);
  // An emit may come before the ensemble of its run, which then checks at the run.
  if (pulse || (skip && integrate.has_ensemble())) {
    validate_deposit_dynamics(pulse);
  }
  if (skip) {
    printf(
      "Flight skipping is ballistic time compression: substrate evolution and thermostat action "
      "during omitted time are not integrated.\n");
  }
  DepositEvent event =
    deposit_registry.execute(tokens, box, atom, group, thermo, force, variables);
  integrate.update_fix_union(atom);
  if (!event.active) {
    return;
  }
  pending_skipped_flight_ = pending_skipped_flight_ || skip;
  if (!event.pulse) {
    return;
  }
  if (
    event.count <= 0 || event.max_wait <= 0 || event.relax_steps < 0 ||
    event.max_wait > std::numeric_limits<int>::max() - event.relax_steps) {
    PRINT_INPUT_ERROR("Invalid deposit pulse step budget or particle count.");
  }
  deposit_event = event;
  deposit_entered.resize(event.count);
  std::vector<int> not_entered(event.count, 0);
  deposit_entered.copy_from_host(not_entered.data());
  deposit_entry_status.resize(2);
  const int number_of_steps = event.max_wait + event.relax_steps;
  force.set_temperature_range(
    integrate.get_temperature1(), integrate.get_temperature2(), number_of_steps);
  perform_a_run(number_of_steps);
  deposit_event = DepositEvent{};
}

void Run::execute_command(std::vector<std::string> tokens)
{
  if (tokens.size() >= 2 && tokens[0] == "potential") {
    tokens[1] = get_compact_nep_filename(tokens[1]);
  }
  if (has_replicate_ && first_effective_command_ && tokens[0] == "replicate") {
    first_effective_command_ = false;
    return;
  }
  first_effective_command_ = false;
  parse_one_keyword(tokens, *run_input_);
}

void Run::execute_run_in(const RunInput& run_input)
{
  print_line_1();
  printf("Started executing the commands in run.in.\n");
  fflush(stdout);
  print_line_2();

  run_input_ = &run_input;
  first_effective_command_ = true;
  execute_control_script(*this, run_input);

  if (integrate.has_ensemble()) {
    PRINT_INPUT_ERROR("The last ensemble is not followed by a run.");
  }

  print_line_1();
  printf("Finished executing the commands in run.in.\n");
  fflush(stdout);
  print_line_2();

}

void Run::compute_force()
{
  if (is_pimd(integrate.get_type())) {
    for (int k = 0; k < integrate.get_number_of_beads(); ++k) {
      force.compute(
        box,
        atom.position_beads[k],
        atom.type,
        group,
        atom.potential_beads[k],
        atom.force_beads[k],
        atom.virial_beads[k],
        atom.velocity_beads[k],
        atom.mass);
    }
  } else {
    force.compute(box, atom, group);
  }
}

void Run::perform_a_run(const int number_of_steps)
{
  if (pending_skipped_flight_) {
    validate_deposit_dynamics(deposit_event.pulse);
  }
  integrate.initialize(time_step, atom, box, group);
  // Other ensembles read the full virial or per-atom energy, or use replicas.
  const EnsembleType ensemble_type = integrate.get_type();
  force.set_fixed_group(integrate.get_fixed_grouping_method(), integrate.get_fixed_group());
  force.set_need_full_observables(
    full_observables_next_run_ ||
    !(ensemble_type == EnsembleType::NVE || is_standard_nvt(ensemble_type)));
  full_observables_next_run_ = false;
  measure.pre_run(number_of_steps, time_step, integrate, group, atom, box, force);

  // setup force for the first integrate step
  compute_force();
  atom.update_unwrapped_position(box);
  measure.setup_force(time_step, integrate, group, atom, box, force);

  double initial_time_step = time_step;

  const auto time_begin = std::chrono::high_resolution_clock::now();

  int completed_steps = 0;
  int entry_step = deposit_event.pulse && update_deposit_entry() ? 0 : -1;
  bool pulse_timeout = false;
  const double run_start_time = global_time;

  for (int step = 0; step < number_of_steps; ++step) {
    if (
      deposit_event.pulse && entry_step >= 0 &&
      completed_steps >= entry_step + deposit_event.relax_steps) {
      break;
    }

    velocity.correct_velocity(step, group, atom);

    calculate_time_step(
      max_distance_per_step, atom.velocity_per_atom, initial_time_step, time_step);
    global_time += time_step;

    integrate.compute1(time_step, step, number_of_steps, group, box, atom, thermo);

    measure.post_integrate1(step, time_step, integrate, group, atom, box, force);

    force.advance_temperature();

    measure.pre_force(step, time_step, integrate, group, atom, box, force);
    compute_force();

    atom.update_unwrapped_position(box);
    measure.post_force(step, time_step, integrate, group, atom, box, force);

    integrate.compute2(time_step, step, number_of_steps, group, box, atom, thermo, force);
    atom.update_unwrapped_position(box);

    completed_steps = step + 1;
    if (deposit_event.pulse && entry_step < 0) {
      if (update_deposit_entry()) {
        entry_step = completed_steps;
      } else if (completed_steps >= deposit_event.max_wait) {
        pulse_timeout = true;
      }
    }

    measure.end_of_step(
      number_of_steps,
      step,
      integrate.get_fixed_group(),
      integrate.get_move_group(),
      global_time,
      integrate.get_temperature2(),
      integrate,
      box,
      group,
      thermo,
      atom,
      force);

    int base = (10 <= number_of_steps) ? (number_of_steps / 10) : 1;
    if (0 == (step + 1) % base) {
      printf("    %d steps completed.\n", step + 1);
      fflush(stdout);
    }
    if (pulse_timeout) {
      break;
    }
  }

  print_line_1();
  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_used = time_finish - time_begin;

  printf("Time used for this run = %g second.\n", time_used.count());
  double run_speed = atom.number_of_atoms * (completed_steps * 1.0 / time_used.count());
  printf("Speed of this run = %g atom*step/second.\n", run_speed);
  print_line_2();

  measure.post_run(
    atom, box, integrate, completed_steps, time_step, integrate.get_temperature2());
  if (deposit_event.pulse && !pulse_timeout) {
    printf(
      "Deposit pulse completed: wait_steps=%d relax_steps=%d md_steps=%d md_time_fs=%.17g\n",
      entry_step,
      deposit_event.relax_steps,
      completed_steps,
      (global_time - run_start_time) * TIME_UNIT_CONVERSION);
  }

  integrate.finalize(atom, box);
  integrate.finish_fix_union(group);
  pending_skipped_flight_ = false;
  if (pulse_timeout) {
    PRINT_INPUT_ERROR("deposit pulse reached max_wait before all particles entered.");
  }
  velocity.finalize();
  force.finalize();
  max_distance_per_step = 0.0;
}

void Run::parse_one_keyword(
  const std::vector<std::string>& tokens, const RunInput& run_input)
{
  if (tokens[0] == "replicate") {
    PRINT_INPUT_ERROR("replicate must be the first effective command.");
  }

  const int num_param = tokens.size();
  const int max_num_param = 32;
  if (num_param > max_num_param)
    PRINT_INPUT_ERROR("The number of parameters should be less than 32.\n");

#ifdef GPUMD_WPE_ENABLED
  force.wpe_process_command(tokens, atom.number_of_atoms);
#endif

  if (tokens[0] == "potential") {
    force.parse_potential(tokens, box, atom, run_input);
  } else if (tokens[0] == "minimize") {
    Minimize minimize;
    minimize.parse_minimize(
      tokens,
      integrate.get_fixed_group(),
      integrate.get_fixed_grouping_method(),
      force,
      box,
      atom,
      group);
  } else if (tokens[0] == "compute_phonon") {
    Hessian hessian;
    hessian.parse(tokens);
    if (!has_replicate_) {
      PRINT_INPUT_ERROR("replicate keyword not found in run.in file.");
    }
    hessian.compute(force, box, atom, group, replicate_size_);
  } else if (tokens[0] == "compute_cohesive") {
    Cohesive cohesive;
    cohesive.parse(tokens, 0);
    cohesive.compute(box, atom, group, force);
  } else if (tokens[0] == "compute_elastic") {
    Cohesive cohesive;
    cohesive.parse(tokens, 1);
    cohesive.compute(box, atom, group, force);
  } else if (tokens[0] == "change_box") {
    parse_change_box(tokens);
  } else if (tokens[0] == "velocity") {
    parse_velocity(tokens);
  } else if (tokens[0] == "ensemble") {
    integrate.parse_ensemble(tokens, atom, box, group, force);
  } else if (tokens[0] == "time_step") {
    parse_time_step(tokens);
  } else if (tokens[0] == "correct_velocity") {
    parse_correct_velocity(tokens, group);
  } else if (tokens[0] == "fix_skip") {
    parse_fix_skip(tokens);
  } else if (tokens[0] == "full_observables") {
    if (tokens.size() != 1) {
      PRINT_INPUT_ERROR("full_observables has no parameters.");
    }
    full_observables_next_run_ = true;
  } else if (tokens[0] == "delete") {
    check_atom_count_change();
    AtomMutation::sync_cpu_from_gpu(atom);
    std::vector<const char*> param;
    for (const std::string& token : tokens) {
      param.push_back(token.c_str());
    }
    Delete(param.data(), num_param, box, atom, group, thermo, force);
    integrate.update_fix_union(atom);
  } else if (tokens[0] == "deposit") {
    parse_deposit(tokens);
  } else if (tokens[0] == "fix") {
    if (tokens.size() > 1 && tokens[1] == "shell") {
      atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
    }
    integrate.parse_fix(tokens, group, atom, box);
  } else if (tokens[0] == "move") {
    integrate.parse_move(tokens, group);
  } else if (tokens[0] == "kspace") {
    if (has_seen_kspace_command) {
      PRINT_INPUT_ERROR("kspace can only appear once.");
    }
    has_seen_kspace_command = true;
  } else if (tokens[0] == "dftd3") {
    if (has_seen_dftd3_command) {
      PRINT_INPUT_ERROR("dftd3 can only appear once.");
    }
    has_seen_dftd3_command = true;
  } else if (tokens[0] == "run") {
    parse_run(tokens);
  } else if (!measure.parse_action(
               tokens,
               number_of_types,
               integrate,
               group,
               atom,
               box,
               force,
               first_potential_filename_)) {
    PRINT_KEYWORD_ERROR(tokens[0].c_str());
  }
}

void Run::parse_velocity(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  double initial_temperature;
  int seed = 0;
  bool use_seed = false;
  if (!(num_param == 2 || num_param == 4)) {
    PRINT_INPUT_ERROR("velocity should have 1 or 3 parameters.\n");
  }

  if (!is_valid_real(tokens[1], &initial_temperature)) {
    PRINT_INPUT_ERROR("initial temperature should be a real number.\n");
  }
  if (initial_temperature <= 0.0) {
    PRINT_INPUT_ERROR("initial temperature should be a positive number.\n");
  }

  if (num_param == 4) {
    if (tokens[2] != "seed") {
      PRINT_INPUT_ERROR("The second parameter for velocity should be 'seed'.\n");
    }
    use_seed = true;
    if (!is_valid_int(tokens[3], &seed) || seed <= 0) {
      PRINT_INPUT_ERROR("seed should be a positive integer.\n");
    }
  }

  velocity.initialize_cpu(
    has_velocity_in_xyz,
    initial_temperature,
    atom,
    use_seed,
    seed);
  atom.velocity_per_atom.copy_from_host(atom.cpu_velocity_per_atom.data());
  if (!has_velocity_in_xyz) {
    printf("Initialized velocities with input T = %g K.\n", initial_temperature);
  }
}

void Run::parse_correct_velocity(
  const std::vector<std::string>& tokens, const std::vector<Group>& group)
{
  const int num_param = tokens.size();
  printf("Correct linear and angular momenta.\n");

  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("correct_velocity should have 1 or 2 parameters.\n");
  }
  if (!is_valid_int(tokens[1], &velocity.velocity_correction_interval)) {
    PRINT_INPUT_ERROR("velocity correction interval should be an integer.\n");
  }
  if (velocity.velocity_correction_interval < 10) {
    PRINT_INPUT_ERROR("velocity correction interval should >= 10.\n");
  }

  printf("    every %d steps.\n", velocity.velocity_correction_interval);

  if (num_param == 3) {
    if (!is_valid_int(tokens[2], &velocity.velocity_correction_group_method)) {
      PRINT_INPUT_ERROR("velocity correction group method should be an integer.\n");
    }
    if (velocity.velocity_correction_group_method < 0) {
      PRINT_INPUT_ERROR("grouping method should >= 0.\n");
    }
    if (velocity.velocity_correction_group_method >= group.size()) {
      PRINT_INPUT_ERROR("grouping method should < maximum number of grouping methods.\n");
    }
  }

  if (velocity.velocity_correction_group_method < 0) {
    printf("    for the whole system.\n");
  } else {
    printf(
      "    for individual groups in group method %d.\n", velocity.velocity_correction_group_method);
  }

  velocity.do_velocity_correction = true;
}

// fix_skip [off] [min_frozen <f>]. Propagating: stays until the next fix_skip line.
void Run::parse_fix_skip(const std::vector<std::string>& tokens)
{
  SkipOptions options = force.skip_options();
  options.enabled = true;
  size_t i = 1;
  while (i < tokens.size()) {
    if (tokens[i] == "off") {
      options.enabled = false;
      ++i;
    } else if (tokens[i] == "min_frozen" && i + 1 < tokens.size()) {
      if (
        !is_valid_real(tokens[i + 1], &options.min_frozen_fraction) ||
        options.min_frozen_fraction < 0.0 || options.min_frozen_fraction > 1.0) {
        PRINT_INPUT_ERROR("fix_skip min_frozen should be in [0, 1].");
      }
      i += 2;
    } else if (
      (tokens[i] == "rebuild" || tokens[i] == "skin" || tokens[i] == "cache_alloc") &&
      i + 1 < tokens.size()) {
      // The candidate neighbor list now decides when the influence set is rebuilt.
      printf("fix_skip ignores %s: it is no longer needed.\n", tokens[i].c_str());
      i += 2;
    } else {
      PRINT_INPUT_ERROR("Usage: fix_skip [off] [min_frozen <f>]");
    }
  }
  force.set_skip_options(options);
  printf(
    "fix_skip: %s, min_frozen %g.\n",
    options.enabled ? "on" : "off",
    options.min_frozen_fraction);
}

void Run::parse_time_step(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("time_step should have 1 or 2 parameters.\n");
  }
  if (!is_valid_real(tokens[1], &time_step)) {
    PRINT_INPUT_ERROR("time_step should be a real number.\n");
  }
  printf("Time step for this run is %g fs.\n", time_step);
  time_step /= TIME_UNIT_CONVERSION;
  if (num_param == 3) {
    if (!is_valid_real(tokens[2], &max_distance_per_step)) {
      PRINT_INPUT_ERROR("max distance per step should be a real number.\n");
    }
    if (max_distance_per_step <= 0.0) {
      PRINT_INPUT_ERROR("max distance per step should > 0.\n");
    }
    printf("    max distance per step = %g A.\n", max_distance_per_step);
  }
}

void Run::parse_run(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  int number_of_steps;
  if (num_param != 2) {
    PRINT_INPUT_ERROR("run should have 1 parameter.\n");
  }
  if (!is_valid_int(tokens[1], &number_of_steps)) {
    PRINT_INPUT_ERROR("number of steps should be an integer.\n");
  }
  if (number_of_steps <= 0) {
    PRINT_INPUT_ERROR("number of steps should be positive.\n");
  }
  if (!integrate.has_ensemble()) {
    PRINT_INPUT_ERROR("An ensemble must be specified before each run.");
  }
  printf("Run %d steps.\n", number_of_steps);

  // set target temperature for temperature-dependent NEP
  force.set_temperature_range(
    integrate.get_temperature1(), integrate.get_temperature2(), number_of_steps);

  perform_a_run(number_of_steps);
}

static __global__ void gpu_deform_atom(
  int N,
  double mu0,
  double mu1,
  double mu2,
  double mu3,
  double mu4,
  double mu5,
  double mu6,
  double mu7,
  double mu8,
  double* g_x,
  double* g_y,
  double* g_z)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double x_old = g_x[i];
    double y_old = g_y[i];
    double z_old = g_z[i];
    g_x[i] = mu0 * x_old + mu1 * y_old + mu2 * z_old;
    g_y[i] = mu3 * x_old + mu4 * y_old + mu5 * z_old;
    g_z[i] = mu6 * x_old + mu7 * y_old + mu8 * z_old;
  }
}

void Run::parse_change_box(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  if (num_param != 2 && num_param != 4 && num_param != 7) {
    PRINT_INPUT_ERROR("change_box can only have 1 or 3 or 6 parameters\n.");
  }

  double deformation_matrix[3][3] = {0.0};

  if (!is_valid_real(tokens[1], &deformation_matrix[0][0])) {
    PRINT_INPUT_ERROR("box change parameter in xx should be a number.");
  }
  deformation_matrix[1][1] = deformation_matrix[2][2] = deformation_matrix[0][0];

  if (num_param >= 4) {
    if (!is_valid_real(tokens[2], &deformation_matrix[1][1])) {
      PRINT_INPUT_ERROR("box change parameter in yy should be a number.");
    }
    if (!is_valid_real(tokens[3], &deformation_matrix[2][2])) {
      PRINT_INPUT_ERROR("box change parameter in zz should be a number.");
    }
  }

  if (num_param == 7) {
    if (!is_valid_real(tokens[4], &deformation_matrix[1][2])) {
      PRINT_INPUT_ERROR("box change parameter in yz should be a number.");
    }
    if (!is_valid_real(tokens[5], &deformation_matrix[0][2])) {
      PRINT_INPUT_ERROR("box change parameter in xz should be a number.");
    }
    if (!is_valid_real(tokens[6], &deformation_matrix[0][1])) {
      PRINT_INPUT_ERROR("box change parameter in xy should be a number.");
    }
    deformation_matrix[1][0] = deformation_matrix[0][1];
    deformation_matrix[2][0] = deformation_matrix[0][2];
    deformation_matrix[2][1] = deformation_matrix[1][2];
  }

  printf("Change box:\n");
  printf("    in xx by %g A.\n", deformation_matrix[0][0]);
  printf("    in yy by %g A.\n", deformation_matrix[1][1]);
  printf("    in zz by %g A.\n", deformation_matrix[2][2]);
  printf("    in yz and zy by strain %g.\n", deformation_matrix[1][2]);
  printf("    in xz and zx by strain %g.\n", deformation_matrix[0][2]);
  printf("    in xy and yz by strain %g.\n", deformation_matrix[0][1]);

  for (int d = 0; d < 3; ++d) {
    deformation_matrix[d][d] =
      (box.cpu_h[d * 3 + d] + deformation_matrix[d][d]) / box.cpu_h[d * 3 + d];
  }

  printf("    Deformation matrix =\n");
  for (int d1 = 0; d1 < 3; ++d1) {
    printf("        ");
    for (int d2 = 0; d2 < 3; ++d2) {
      printf("%g ", deformation_matrix[d1][d2]);
    }
    printf("\n");
  }

  printf("    Original box h = [a, b, c] is\n");
  for (int d1 = 0; d1 < 3; ++d1) {
    printf("        ");
    for (int d2 = 0; d2 < 3; ++d2) {
      printf("%g ", box.cpu_h[d1 * 3 + d2]);
    }
    printf("\n");
  }

  double h_old[9];
  for (int i = 0; i < 9; ++i) {
    h_old[i] = box.cpu_h[i];
  }

  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      double tmp = 0.0;
      for (int k = 0; k < 3; ++k) {
        tmp += deformation_matrix[r][k] * h_old[k * 3 + c];
      }
      box.cpu_h[r * 3 + c] = tmp;
    }
  }
  box.get_inverse();

  const int number_of_atoms = atom.position_per_atom.size() / 3;
  gpu_deform_atom<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    deformation_matrix[0][0],
    deformation_matrix[0][1],
    deformation_matrix[0][2],
    deformation_matrix[1][0],
    deformation_matrix[1][1],
    deformation_matrix[1][2],
    deformation_matrix[2][0],
    deformation_matrix[2][1],
    deformation_matrix[2][2],
    atom.position_per_atom.data(),
    atom.position_per_atom.data() + number_of_atoms,
    atom.position_per_atom.data() + number_of_atoms * 2);
  GPU_CHECK_KERNEL

  printf("    Changed box h = [a, b, c] is\n");
  for (int d1 = 0; d1 < 3; ++d1) {
    printf("        ");
    for (int d2 = 0; d2 < 3; ++d2) {
      printf("%g ", box.cpu_h[d1 * 3 + d2]);
    }
    printf("\n");
  }
}
