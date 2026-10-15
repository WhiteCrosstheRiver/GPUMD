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

#include "parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

Parameters::Parameters()
{
  print_line_1();
  printf("Started reading gnep.in.\n");
  print_line_2();

  set_default_parameters();
  read_gmlp_in();
  if (!prediction && !is_type_set) {
    PRINT_INPUT_ERROR("type in gnep.in has not been set.");
  }
  if (!prediction && is_zbl_set) {
    read_zbl_in();
  }
  if (fcvp_interval && weight_decay != 0.0f) {
    PRINT_INPUT_ERROR("FCVP requires weight_decay 0; use lambda_2 for norm regularization.");
  }
  if ((lambda_1 > 0 || lambda_2 > 0) && !loss_rmse) {
    PRINT_INPUT_ERROR("lambda_1/lambda_2 require loss_rmse or fcvp.");
  }
  if (loss_rmse && use_full_batch) {
    PRINT_INPUT_ERROR("loss_rmse/FCVP does not support batch accumulation; use batch <size> 0.");
  }
  if (
    loss_rmse && (fcvp_epsilon < std::numeric_limits<float>::min() ||
                  fcvp_epsilon > std::numeric_limits<float>::max())) {
    PRINT_INPUT_ERROR("RMSE epsilon must be in the positive normal float range.");
  }
  if (prediction) {
    read_prediction_model();
  } else {
    calculate_parameters();
  }
  report_inputs();

  print_line_1();
  printf("Finished reading gnep.in.\n");
  print_line_2();
}

void Parameters::set_default_parameters()
{
  is_prediction_set = false;
  is_type_set = false;
  is_cutoff_set = false;
  is_n_max_set = false;
  is_basis_size_set = false;
  is_l_max_set = false;
  is_neuron_set = false;
  is_weight_decay_set = false;
  is_start_lr_set = false;
  is_stop_lr_set = false;
  is_lambda_e_set = is_lambda_f_set = is_lambda_v_set = false;
  is_lambda_shear_set = false;
  is_batch_set = false;
  is_epoch_set = false;
  is_type_weight_set = false;
  is_zbl_set = false;
  is_force_delta_set = false;
  is_use_typewise_cutoff_set = false;
  is_use_typewise_cutoff_zbl_set = false;
  is_energy_shift_set = false;
  is_lr_cos_restart_set = false;

  prediction = 0;              // not prediction mode
  rc_radial = 8.0f;            // large enough for vdw/coulomb
  rc_angular = 4.0f;           // large enough in most cases
  basis_size_radial = 8;       // large enough in most cases
  basis_size_angular = 8;      // large enough in most cases
  n_max_radial = 4;            // a relatively small value to achieve high speed
  n_max_angular = 4;           // a relatively small value to achieve high speed
  L_max = 4;                   // preserve the legacy three-body default
  num_neurons1 = 30;           // a relatively small value to achieve high speed
  weight_decay = 0.0f;         // no weight decay by default (Adam). In general, 1e-6 ~ 1e-4 for AdamW
  lr = 1e-3f;                 
  start_lr = 1e-3f;   
  stop_lr = 1e-7f;             
  lambda_e = 1.0f;           // energy important
  lambda_f = 2.0f;         // force is more important
  lambda_v = 0.1f;             // virial is less important, virial is inaccuracy in most cases
  lambda_shear = 1.0f;         // do not weight shear virial more by default
  force_delta = 0.0f;          // no modification of force loss
  batch_size = 2;           // mini-batch for adam optimizer
  use_full_batch = 0;          // default is not to enable effective full-batch
  epoch = 50;               
  use_typewise_cutoff = false;
  use_typewise_cutoff_zbl = false;
  typewise_cutoff_radial_factor = -1.0f;
  typewise_cutoff_angular_factor = -1.0f;
  typewise_cutoff_zbl_factor = -1.0f;
  energy_shift = 0;
  output_descriptor = false;

  // default for lr cosine restart scheduler
  lr_restart_enable = 0;
  lr_warmup_epochs = 1;
  lr_restart_initial_period_epochs = 10; 
  lr_restart_period_factor = 2.0f;
  lr_restart_decay_factor = 0.8f;
  

  type_weight_cpu.resize(NUM_ELEMENTS);
  for (int n = 0; n < NUM_ELEMENTS; ++n) {
    type_weight_cpu[n] = 1.0f; // uniform weight by default
  }
  enable_zbl = false;   // default is not to include ZBL
  flexible_zbl = false; // default Universal ZBL
}

void Parameters::read_gmlp_in()
{
  std::ifstream input("gnep.in");
  if (!input.is_open()) {
    std::cout << "Failed to open gnep.in." << std::endl;
    exit(1);
  }

  while (input.peek() != EOF) {
    std::vector<std::string> tokens = get_tokens(input);
    std::vector<std::string> tokens_without_comments;
    for (const auto& t : tokens) {
      if (t[0] != '#') {
        tokens_without_comments.emplace_back(t);
      } else {
        break;
      }
    }
    if (tokens_without_comments.size() > 0) {
      parse_one_keyword(tokens_without_comments);
    }
  }

  input.close();
}

void Parameters::read_zbl_in()
{
  FILE* fid_zbl = fopen("zbl.in", "r");
  if (fid_zbl == NULL) {
    flexible_zbl = false;
  } else {
    flexible_zbl = true;
    zbl_para.resize(10 * num_types * (num_types + 1) / 2);
    for (int n = 0; n < (num_types * (num_types + 1) / 2) * 10; ++n) {
      int count = fscanf(fid_zbl, "%f", &zbl_para[n]);
      PRINT_SCANF_ERROR(count, 1, "Reading error for zbl.in.");
    }
    fclose(fid_zbl);
    validate_zbl_parameters();
  }
}

void Parameters::calculate_parameters()
{
  if (!is_type_set) {
    PRINT_INPUT_ERROR("type in gnep.in has not been set.");
  }
  dim_radial = n_max_radial + 1;             // 2-body descriptors q^i_n
  num_L = L_max + (L_max_4body == 2) + (L_max_5body == 1);
  dim_angular = (n_max_angular + 1) * num_L;
  dim = dim_radial + dim_angular;
  q_scaler_cpu.resize(dim, 1.0e10f);
#ifdef USE_FIXED_SCALER
  for (int n = 0; n < q_scaler_cpu.size(); ++n) {
    q_scaler_cpu[n] = 0.01f;
  }
#endif

  number_of_variables_ann = ((dim + 2) * num_neurons1 + 1) * num_types;

  number_of_variables_descriptor =
    num_types * num_types *
    (dim_radial * (basis_size_radial + 1) + (n_max_angular + 1) * (basis_size_angular + 1));

  number_of_variables = number_of_variables_ann + number_of_variables_descriptor;

  int deviceCount;
  CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 1 || deviceCount > 16) {
    PRINT_INPUT_ERROR("GNEP requires between 1 and 16 visible GPUs.");
  }
  for (int device_id = 0; device_id < deviceCount; device_id++) {
    CHECK(cudaSetDevice(device_id));
    q_scaler_gpu[device_id].resize(dim);
    s_max[device_id].resize(dim, -1000000.0f);
    s_min[device_id].resize(dim, +1000000.0f);
    q_scaler_gpu[device_id].copy_from_host(q_scaler_cpu.data());
    energy_shift_gpu.resize(num_types, 0.0f);
  }
}

void Parameters::validate_zbl_parameters()
{
  for (const float value : zbl_para) {
    if (!std::isfinite(value)) {
      PRINT_INPUT_ERROR("ZBL parameters must be finite.");
    }
  }
  for (size_t pair = 0; pair < zbl_para.size(); pair += 10) {
    if (!(zbl_para[pair] >= 0 && zbl_para[pair + 1] > zbl_para[pair]) ||
        zbl_para[pair + 1] > rc_radial) {
      PRINT_INPUT_ERROR("ZBL requires 0 <= inner < outer <= radial cutoff for each pair.");
    }
  }
}

void Parameters::read_prediction_model()
{
  std::ifstream input("nep.txt");
  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Failed to open nep.txt.");
  }
  auto tokens = get_tokens(input);
  if (tokens.size() < 3 || (tokens[0] != "nep5" && tokens[0] != "nep5_zbl")) {
    PRINT_INPUT_ERROR("GNEP prediction requires a NEP5 or NEP5_ZBL energy model.");
  }
  enable_zbl = is_zbl_set = tokens[0] == "nep5_zbl";
  elements.clear();
  atomic_numbers.clear();
  tokens[0] = "type";
  parse_one_keyword(tokens);

  auto header = [&](const char* name, size_t minimum, size_t maximum) {
    auto line = get_tokens(input);
    if (line.size() < minimum || line.size() > maximum || line[0] != name) {
      PRINT_INPUT_ERROR("Invalid or truncated nep.txt model header.");
    }
    return line;
  };
  auto real = [&](const std::string& token) {
    double value = 0;
    if (!is_valid_real(token.c_str(), &value) || !std::isfinite(value) ||
        std::abs(value) > std::numeric_limits<float>::max()) {
      PRINT_INPUT_ERROR("nep.txt requires finite FP32 model parameters.");
    }
    return float(value);
  };
  flexible_zbl = false;
  if (enable_zbl) {
    tokens = header("zbl", 3, 3);
    zbl_rc_inner = real(tokens[1]);
    zbl_rc_outer = real(tokens[2]);
    flexible_zbl = zbl_rc_inner == 0 && zbl_rc_outer == 0;
    if (!flexible_zbl && !(zbl_rc_inner >= 0 && zbl_rc_outer > zbl_rc_inner)) {
      PRINT_INPUT_ERROR("Invalid universal ZBL cutoffs in nep.txt.");
    }
  }
  tokens = header("cutoff", 5, 8);
  if (tokens.size() != 5 && tokens.size() != 8) {
    PRINT_INPUT_ERROR("Invalid cutoff header in nep.txt.");
  }
  rc_radial = real(tokens[1]);
  rc_angular = real(tokens[2]);
  for (int field = 3; field <= 4; ++field) {
    int maximum_neighbors = 0;
    if (!is_valid_int(tokens[field].c_str(), &maximum_neighbors) || maximum_neighbors < 0) {
      PRINT_INPUT_ERROR("Invalid neighbor capacity in nep.txt cutoff header.");
    }
  }
  if (!(rc_radial > 0 && rc_angular > 0 && rc_angular <= rc_radial)) {
    PRINT_INPUT_ERROR("nep.txt requires 0 < angular cutoff <= radial cutoff.");
  }
  use_typewise_cutoff = use_typewise_cutoff_zbl = false;
  typewise_cutoff_radial_factor = typewise_cutoff_angular_factor =
    typewise_cutoff_zbl_factor = -1.0f;
  if (tokens.size() == 8) {
    typewise_cutoff_radial_factor = real(tokens[5]);
    typewise_cutoff_angular_factor = real(tokens[6]);
    typewise_cutoff_zbl_factor = real(tokens[7]);
    if ((typewise_cutoff_radial_factor > 0) != (typewise_cutoff_angular_factor > 0)) {
      PRINT_INPUT_ERROR("nep.txt typewise radial/angular factors must be enabled together.");
    }
    use_typewise_cutoff = typewise_cutoff_radial_factor > 0;
    use_typewise_cutoff_zbl = typewise_cutoff_zbl_factor > 0;
  }
  tokens = header("n_max", 3, 3);
  parse_one_keyword(tokens);
  tokens = header("basis_size", 3, 3);
  parse_one_keyword(tokens);
  tokens = header("l_max", 4, 4);
  parse_one_keyword(tokens);
  tokens = header("ANN", 3, 3);
  if (tokens[2] != "0") {
    PRINT_INPUT_ERROR("GNEP supports one hidden layer in nep.txt.");
  }
  tokens[0] = "neuron";
  tokens.resize(2);
  parse_one_keyword(tokens);
  calculate_parameters();
  auto next_value = [&]() {
    std::string token;
    if (!(input >> token)) {
      PRINT_INPUT_ERROR("Truncated nep.txt parameter/scaler/ZBL data.");
    }
    return real(token);
  };
  prediction_parameters.resize(number_of_variables);
  for (int i = 0; i < number_of_variables_ann; ++i) {
    prediction_parameters[i] = next_value();
  }
  // NEP5 has per-type biases plus a shared bias in its serialized format.
  const float common_bias = next_value();
  const int block = (dim + 2) * num_neurons1 + 1;
  for (int t = 0; t < num_types; ++t) {
    prediction_parameters[(t + 1) * block - 1] += common_bias;
    if (!std::isfinite(prediction_parameters[(t + 1) * block - 1])) {
      PRINT_INPUT_ERROR("nep.txt shared/per-type bias sum exceeds the FP32 range.");
    }
  }
  for (int i = number_of_variables_ann; i < number_of_variables; ++i) {
    prediction_parameters[i] = next_value();
  }
  for (float& scale : q_scaler_cpu) {
    scale = next_value();
    if (!(scale > 0)) {
      PRINT_INPUT_ERROR("nep.txt descriptor scales must be positive.");
    }
  }
  zbl_para.clear();
  if (flexible_zbl) {
    zbl_para.resize(10 * num_types * (num_types + 1) / 2);
    for (float& value : zbl_para) {
      value = next_value();
    }
    validate_zbl_parameters();
  } else if (enable_zbl && zbl_rc_outer > rc_radial) {
    PRINT_INPUT_ERROR("ZBL cutoff exceeds the radial neighbor cutoff.");
  }
  std::string extra;
  if (input >> extra) {
    PRINT_INPUT_ERROR("Unexpected trailing parameters in nep.txt.");
  }
  int device_count = 0;
  CHECK(cudaGetDeviceCount(&device_count));
  for (int device = 0; device < device_count; ++device) {
    CHECK(cudaSetDevice(device));
    q_scaler_gpu[device].copy_from_host(q_scaler_cpu.data());
  }
  CHECK(cudaSetDevice(0));
}

void Parameters::report_inputs()
{
  if (!is_type_set) {
    PRINT_INPUT_ERROR("type in gnep.in has not been set.");
  }

  printf("Input or default parameters:\n");
  printf("    seed = %d; loss_rmse = %d; fcvp interval = %d\n", seed, loss_rmse, fcvp_interval);
  if (loss_rmse)
    printf(
      "    RMSE epsilon = %g; lambda_1 = %g; lambda_2 = %g\n", fcvp_epsilon, lambda_1, lambda_2);

  std::string train_mode_name = "potential";
  printf("model_type = %s.\n", train_mode_name.c_str());

  std::string calculation_mode_name = "train";
  if (prediction == 1) {
    calculation_mode_name = "predict";
  }
  if (is_prediction_set) {
    printf("    (input)   calculation mode = %s.\n", calculation_mode_name.c_str());
  } else {
    printf("    (default) calculation mode = %s.\n", calculation_mode_name.c_str());
  }

  printf("    (input)   number of atom types = %d.\n", num_types);
  if (is_type_weight_set) {
    for (int n = 0; n < num_types; ++n) {
      printf(
        "        (input)   type %d (%s with Z = %d) has force weight of %g.\n",
        n,
        elements[n].c_str(),
        atomic_numbers[n],
        type_weight_cpu[n]);
    }
  } else {
    for (int n = 0; n < num_types; ++n) {
      printf(
        "        (default) type %d (%s with Z = %d) has force weight of %g.\n",
        n,
        elements[n].c_str(),
        atomic_numbers[n],
        type_weight_cpu[n]);
    }
  }

  if (is_zbl_set) {
    if (flexible_zbl) {
      printf("    (input)   will add the flexible ZBL potential\n");
    } else {
      printf(
        "    (input)   will add the universal ZBL potential with outer cutoff %g A and inner "
        "cutoff %g A.\n",
        zbl_rc_outer,
        zbl_rc_inner);
    }
  } else {
    printf("    (default) will not add the ZBL potential.\n");
  }

  if (is_cutoff_set) {
    printf("    (input)   radial cutoff = %g A.\n", rc_radial);
    printf("    (input)   angular cutoff = %g A.\n", rc_angular);
  } else {
    printf("    (default) radial cutoff = %g A.\n", rc_radial);
    printf("    (default) angular cutoff = %g A.\n", rc_angular);
  }

  if (is_use_typewise_cutoff_set) {
    printf("    (input)   use %s cutoff for GNEP.\n", use_typewise_cutoff ? "typewise" : "global");
    printf("              radial factor = %g.\n", typewise_cutoff_radial_factor);
    printf("              angular factor = %g.\n", typewise_cutoff_angular_factor);
  } else {
    printf("    (default) use %s cutoff for GNEP.\n", use_typewise_cutoff ? "typewise" : "global");
  }

  if (is_use_typewise_cutoff_zbl_set) {
    printf(
      "    (input)   use %s cutoff for ZBL.\n", use_typewise_cutoff_zbl ? "typewise" : "global");
    printf("              factor = %g.\n", typewise_cutoff_zbl_factor);
  } else {
    printf(
      "    (default) use %s cutoff for ZBL.\n", use_typewise_cutoff_zbl ? "typewise" : "global");
  }

  if (is_n_max_set) {
    printf("    (input)   n_max_radial = %d.\n", n_max_radial);
    printf("    (input)   n_max_angular = %d.\n", n_max_angular);
  } else {
    printf("    (default) n_max_radial = %d.\n", n_max_radial);
    printf("    (default) n_max_angular = %d.\n", n_max_angular);
  }

  if (is_basis_size_set) {
    printf("    (input)   basis_size_radial = %d.\n", basis_size_radial);
    printf("    (input)   basis_size_angular = %d.\n", basis_size_angular);
  } else {
    printf("    (default) basis_size_radial = %d.\n", basis_size_radial);
    printf("    (default) basis_size_angular = %d.\n", basis_size_angular);
  }

  if (is_l_max_set) {
    printf("    (input)   l_max_3body = %d.\n", L_max);
  } else {
    printf("    (default) l_max_3body = %d.\n", L_max);
  }
  printf("              l_max_4body = %d; l_max_5body = %d.\n", L_max_4body, L_max_5body);

  if (is_neuron_set) {
    printf("    (input)   number of neurons = %d.\n", num_neurons1);
  } else {
    printf("    (default) number of neurons = %d.\n", num_neurons1);
  }

  if (is_weight_decay_set) {
    printf("    (input)   weight_decay = %g.\n", weight_decay);
  } else {
    printf("    (default) weight_decay = %g.\n", weight_decay);
  }

  if (is_start_lr_set) {
    printf("    (input)   start learning rate = %g.\n", start_lr);
  } else {
    printf("    (default) start learning rate = %g.\n", start_lr);
  }

  if (is_stop_lr_set) {
    printf("    (input)   stop learing rate = %g.\n", stop_lr);
  } else {
    printf("    (default) stop learing rate = %g.\n", stop_lr);
  }

  if (is_lambda_e_set) {
    printf("    (input)   lambda_e = %g.\n", lambda_e);
  } else {
    printf("    (default) lambda_e = %g.\n", lambda_e);
  }

  if (is_lambda_f_set) {
    printf("    (input)   lambda_f = %g.\n", lambda_f);
  } else {
    printf("    (default) lambda_f = %g.\n", lambda_f);
  }

  if (is_lambda_v_set) {
    printf("    (input)   lambda_v = %g.\n", lambda_v);
  } else {
    printf("    (default) lambda_v = %g.\n", lambda_v);
  }

  if (is_lambda_shear_set) {
    printf("    (input)   lambda_shear = %g.\n", lambda_shear);
  } else {
    printf("    (default) lambda_shear = %g.\n", lambda_shear);
  }

  if (is_force_delta_set) {
    printf("    (input)   force_delta = %g.\n", force_delta);
  } else {
    printf("    (default) force_delta = %g.\n", force_delta);
  }

  if (is_batch_set) {
    printf("    (input)   batch size = %d.\n", batch_size);
  } else {
    printf("    (default) batch size = %d.\n", batch_size);
  }

  if (is_epoch_set) {
    printf("    (input)   maximum number of epochs = %d.\n", epoch);
  } else {
    printf("    (default) maximum number of epochs = %d.\n", epoch);
  }

  // report lr cosine restart settings
  if (lr_restart_enable) {
    printf("    (input)   lr_cos_restart enabled.\n");
    printf("              warmup_epochs = %d.\n", lr_warmup_epochs);
    printf("              initial_period_epochs = %d.\n", lr_restart_initial_period_epochs);
    printf("              period_factor = %g.\n", lr_restart_period_factor);
    printf("              decay_factor = %g.\n", lr_restart_decay_factor);
  }

  // some calcuated parameters:
  printf("Some calculated parameters:\n");
  printf("    number of radial descriptor components = %d.\n", dim_radial);
  printf("    number of angular descriptor components = %d.\n", dim_angular);
  printf("    total number of descriptor components = %d.\n", dim);
  printf("    NN architecture = %d-%d-1.\n", dim, num_neurons1);
  printf(
    "    number of NN parameters to be optimized = %d.\n",
    number_of_variables_ann);
  printf(
    "    number of descriptor parameters to be optimized = %d.\n", number_of_variables_descriptor);
  printf("    total number of parameters to be optimized = %d.\n", number_of_variables);
  // GNEP has larger work arrays than the separate MD inference implementation.
  if (basis_size_radial > 16 || basis_size_angular > 16 ||
      (L_max > 0 && n_max_angular > 16) || dim_angular > 90 || dim > 103 ||
      (flexible_zbl && num_types > 10)) {
    printf("WARNING: This model exceeds the current GPUMD MD inference capacity.\n"
           "         Native GNEP training/prediction is supported; GPUMD will reject this model.\n");
  }
  if (enable_zbl) {
    printf("    MD compatibility: every effective ZBL cutoff must fit the MD angular neighbor list.\n"
           "    GNEP uses a NEP/ZBL neighbor union; MD rejects unsupported cutoff coverage.\n");
  }
}

void Parameters::parse_one_keyword(std::vector<std::string>& tokens)
{
  int num_param = tokens.size();
  const char* param[105]; // never use more than 104 parameters
  for (int n = 0; n < num_param; ++n) {
    param[n] = tokens[n].c_str();
  }
  if (strcmp(param[0], "fcvp") == 0) {
    parse_fcvp(param, num_param);
  } else if (strcmp(param[0], "fcvp_memory") == 0) {
    parse_fcvp_memory(param, num_param);
  } else if (strcmp(param[0], "fcvp_check") == 0) {
    parse_fcvp_check(param, num_param);
  } else if (strcmp(param[0], "loss_rmse") == 0) {
    parse_loss_rmse(param, num_param);
  } else if (strcmp(param[0], "lambda_1") == 0 || strcmp(param[0], "lambda_2") == 0) {
    parse_regularization(param, num_param);
  } else if (strcmp(param[0], "seed") == 0) {
    parse_seed(param, num_param);
  } else if (strcmp(param[0], "prediction") == 0) {
    parse_prediction(param, num_param);
  } else if (strcmp(param[0], "type") == 0) {
    parse_type(param, num_param);
  } else if (strcmp(param[0], "cutoff") == 0) {
    parse_cutoff(param, num_param);
  } else if (strcmp(param[0], "n_max") == 0) {
    parse_n_max(param, num_param);
  } else if (strcmp(param[0], "basis_size") == 0) {
    parse_basis_size(param, num_param);
  } else if (strcmp(param[0], "l_max") == 0) {
    parse_l_max(param, num_param);
  } else if (strcmp(param[0], "neuron") == 0) {
    parse_neuron(param, num_param);
  } else if (strcmp(param[0], "batch") == 0) {
    parse_batch(param, num_param);
  } else if (strcmp(param[0], "epoch") == 0) {
    parse_epoch(param, num_param);
  } else if (strcmp(param[0], "weight_decay") == 0) {
    parse_weight_decay(param, num_param);
  } else if (strcmp(param[0], "start_lr") == 0) {
    parse_start_lr(param, num_param);
  } else if (strcmp(param[0], "stop_lr") == 0) {
    parse_stop_lr(param, num_param);
  } else if (strcmp(param[0], "lambda_e") == 0) {
    parse_lambda_e(param, num_param);
  } else if (strcmp(param[0], "lambda_f") == 0) {
    parse_lambda_f(param, num_param);
  } else if (strcmp(param[0], "lambda_v") == 0) {
    parse_lambda_v(param, num_param);
  } else if (strcmp(param[0], "lambda_shear") == 0) {
    parse_lambda_shear(param, num_param);
  } else if (strcmp(param[0], "type_weight") == 0) {
    parse_type_weight(param, num_param);
  } else if (strcmp(param[0], "force_delta") == 0) {
    parse_force_delta(param, num_param);
  } else if (strcmp(param[0], "zbl") == 0) {
    parse_zbl(param, num_param);
  } else if (strcmp(param[0], "use_typewise_cutoff") == 0) {
    parse_use_typewise_cutoff(param, num_param);
  } else if (strcmp(param[0], "use_typewise_cutoff_zbl") == 0) {
    parse_use_typewise_cutoff_zbl(param, num_param);
  } else if (strcmp(param[0], "energy_shift") == 0) {
    parse_energy_shift(param, num_param);
  } else if (strcmp(param[0], "output_descriptor") == 0) {
    parse_output_descriptor(param, num_param);
  } else if (strcmp(param[0], "lr_cos_restart") == 0) {
    parse_lr_cos_restart(param, num_param);
  } else {
    PRINT_KEYWORD_ERROR(param[0]);
  }
}

void Parameters::parse_prediction(const char** param, int num_param)
{
  is_prediction_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("prediction should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &prediction)) {
    PRINT_INPUT_ERROR("prediction should be an integer.\n");
  }
  if (prediction != 0 && prediction != 1) {
    PRINT_INPUT_ERROR("prediction should = 0 or 1.");
  }
}

void Parameters::parse_type(const char** param, int num_param)
{
  is_type_set = true;

  if (num_param < 3) {
    PRINT_INPUT_ERROR("type should have at least 2 parameters.\n");
  }
  if (!is_valid_int(param[1], &num_types)) {
    PRINT_INPUT_ERROR("number of types should be integer.\n");
  }

  if (num_types < 1 || num_types > NUM_ELEMENTS) {
    PRINT_INPUT_ERROR("number of types should >=1 and <= NUM_ELEMENTS.");
  }
  if (num_param != 2 + num_types) {
    PRINT_INPUT_ERROR("number of types and the number of listed elements do not match.\n");
  }
  for (int n = 0; n < num_types; ++n) {
    elements.emplace_back(param[2 + n]);
    bool is_valid_element = false;
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (elements.back() == ELEMENTS[m]) {
        is_valid_element = true;
        atomic_number = m + 1;
        break;
      }
    }
    atomic_numbers.emplace_back(atomic_number);
    if (!is_valid_element) {
      PRINT_INPUT_ERROR("Some element in gnep.in is not in the periodic table.");
    }
  }
}

void Parameters::parse_type_weight(const char** param, int num_param)
{
  is_type_weight_set = true;

  if (!is_type_set) {
    PRINT_INPUT_ERROR("Please set type before setting type weight.\n");
  }

  if (num_param != 1 + num_types) {
    PRINT_INPUT_ERROR("type_weight should have num_types parameters.\n");
  }

  for (int n = 0; n < num_types; ++n) {
    double weight_tmp = 0.0;
    if (!is_valid_real(param[1 + n], &weight_tmp)) {
      PRINT_INPUT_ERROR("type weight should be a number.\n");
    }
    type_weight_cpu[n] = weight_tmp;
  }
}

void Parameters::parse_zbl(const char** param, int num_param)
{
  is_zbl_set = true;
  enable_zbl = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("zbl should have 1 parameter.\n");
  }

  double zbl_rc_outer_tmp = 0.0;
  if (!is_valid_real(param[1], &zbl_rc_outer_tmp)) {
    PRINT_INPUT_ERROR("outer cutoff for ZBL should be a number.\n");
  }
  zbl_rc_outer = zbl_rc_outer_tmp;
  zbl_rc_inner = zbl_rc_outer * 0.5f;

  if (zbl_rc_outer < 1.0f) {
    PRINT_INPUT_ERROR("outer cutoff for ZBL should >= 1.0 A.");
  } else if (zbl_rc_outer > 2.5f) {
    PRINT_INPUT_ERROR("outer cutoff for ZBL should <= 2.5 A.");
  }
}

void Parameters::parse_force_delta(const char** param, int num_param)
{
  is_force_delta_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("force_delta should have 1 parameter.\n");
  }

  double force_delta_tmp = 0.0;
  if (!is_valid_real(param[1], &force_delta_tmp)) {
    PRINT_INPUT_ERROR("force_delta should be a number.\n");
  }
  force_delta = force_delta_tmp;
}

void Parameters::parse_cutoff(const char** param, int num_param)
{
  is_cutoff_set = true;

  if (num_param != 3) {
    PRINT_INPUT_ERROR("cutoff should have 2 parameters.\n");
  }

  double rc_radial_tmp = 0.0;
  if (!is_valid_real(param[1], &rc_radial_tmp)) {
    PRINT_INPUT_ERROR("radial cutoff should be a number.\n");
  }
  rc_radial = rc_radial_tmp;

  double rc_angular_tmp = 0.0;
  if (!is_valid_real(param[2], &rc_angular_tmp)) {
    PRINT_INPUT_ERROR("angular cutoff should be a number.\n");
  }
  rc_angular = rc_angular_tmp;

  if (rc_angular > rc_radial) {
    PRINT_INPUT_ERROR("angular cutoff should <= radial cutoff.");
  }
  if (rc_angular < 2.5f) {
    PRINT_INPUT_ERROR("angular cutoff should >= 2.5 A.");
  }
  if (rc_radial > 10.0f) {
    PRINT_INPUT_ERROR("radial cutoff should <= 10 A.");
  }
}

void Parameters::parse_n_max(const char** param, int num_param)
{
  is_n_max_set = true;

  if (num_param != 3) {
    PRINT_INPUT_ERROR("n_max should have 2 parameters.\n");
  }
  if (!is_valid_int(param[1], &n_max_radial)) {
    PRINT_INPUT_ERROR("n_max_radial should be an integer.\n");
  }
  if (!is_valid_int(param[2], &n_max_angular)) {
    PRINT_INPUT_ERROR("n_max_angular should be an integer.\n");
  }
  if (n_max_radial < 0) {
    PRINT_INPUT_ERROR("n_max_radial should >= 0.");
  } else if (n_max_radial > 19) {
    PRINT_INPUT_ERROR("n_max_radial should <= 19.");
  }
  if (n_max_angular < 0) {
    PRINT_INPUT_ERROR("n_max_angular should >= 0.");
  } else if (n_max_angular > 19) {
    PRINT_INPUT_ERROR("n_max_angular should <= 19.");
  }
}

void Parameters::parse_basis_size(const char** param, int num_param)
{
  is_basis_size_set = true;

  if (num_param != 3) {
    PRINT_INPUT_ERROR("basis_size should have 2 parameters.\n");
  }
  if (!is_valid_int(param[1], &basis_size_radial)) {
    PRINT_INPUT_ERROR("basis_size_radial should be an integer.\n");
  }
  if (!is_valid_int(param[2], &basis_size_angular)) {
    PRINT_INPUT_ERROR("basis_size_angular should be an integer.\n");
  }
  if (basis_size_radial < 0) {
    PRINT_INPUT_ERROR("basis_size_radial should >= 0.");
  } else if (basis_size_radial > 19) {
    PRINT_INPUT_ERROR("basis_size_radial should <= 19.");
  }
  if (basis_size_angular < 0) {
    PRINT_INPUT_ERROR("basis_size_angular should >= 0.");
  } else if (basis_size_angular > 19) {
    PRINT_INPUT_ERROR("basis_size_angular should <= 19.");
  }
}

void Parameters::parse_l_max(const char** param, int num_param)
{
  is_l_max_set = true;

  if (num_param < 2 || num_param > 4) {
    PRINT_INPUT_ERROR("l_max requires L3 [L4 [L5]].\n");
  }
  if (!is_valid_int(param[1], &L_max)) {
    PRINT_INPUT_ERROR("l_max for 3-body descriptors should be an integer.\n");
  }
  if (L_max < 0) {
    PRINT_INPUT_ERROR("l_max for 3-body descriptors should >= 0.");
  }
  if (L_max > 8) {
    PRINT_INPUT_ERROR("l_max for 3-body descriptors should <= 8.");
  }
  L_max_4body = L_max_5body = 0;
  if (num_param >= 3 && !is_valid_int(param[2], &L_max_4body)) {
    PRINT_INPUT_ERROR("l_max for 4-body descriptors should be an integer.");
  }
  if (num_param == 4 && !is_valid_int(param[3], &L_max_5body)) {
    PRINT_INPUT_ERROR("l_max for 5-body descriptors should be an integer.");
  }
  if ((L_max_4body != 0 && L_max_4body != 2) || L_max_4body > L_max) {
    PRINT_INPUT_ERROR("l_max_4body must be 0 or 2 and not exceed l_max_3body.");
  }
  if ((L_max_5body != 0 && L_max_5body != 1) || (L_max_5body && !L_max_4body)) {
    PRINT_INPUT_ERROR("l_max_5body must be 0 or 1; 1 requires l_max_4body = 2.");
  }
}

void Parameters::parse_fcvp_memory(const char** param, int num_param)
{
  if (num_param != 2 || !is_valid_int(param[1], &fcvp_memory_mib) || fcvp_memory_mib < 1) {
    PRINT_INPUT_ERROR("fcvp_memory requires a positive integer number of MiB per GPU.");
  }
}

std::vector<int> Parameters::output_parameter_indices() const
{
  std::vector<int> result;
  const int block = (dim + 2) * num_neurons1 + 1;
  for (int t = 0; t < num_types; ++t) {
    for (int j = 0; j <= num_neurons1; ++j) {
      result.push_back(t * block + (dim + 1) * num_neurons1 + j);
    }
  }
  return result;
}

void Parameters::parse_neuron(const char** param, int num_param)
{
  is_neuron_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("neuron should have 1 parameter.\n");
  }

  if (!is_valid_int(param[1], &num_neurons1)) {
    PRINT_INPUT_ERROR("number of neurons should be an integer.\n");
  }
  if (num_neurons1 < 1) {
    PRINT_INPUT_ERROR("number of neurons should >= 1.");
  } else if (num_neurons1 > 200) {
    PRINT_INPUT_ERROR("number of neurons should <= 200.");
  }
}

void Parameters::parse_fcvp(const char** param, int num_param)
{
  if (
    num_param != 5 || !is_valid_int(param[1], &fcvp_interval) ||
    !is_valid_int(param[2], &fcvp_iterations) || !is_valid_real(param[3], &fcvp_epsilon) ||
    !is_valid_real(param[4], &fcvp_tolerance) || fcvp_interval < 1 || fcvp_iterations < 1 ||
    !(fcvp_epsilon > 0) || !(fcvp_tolerance > 0) || !std::isfinite(fcvp_epsilon) ||
    !std::isfinite(fcvp_tolerance)) {
    PRINT_INPUT_ERROR("fcvp requires: interval>=1 iterations>=1 epsilon>0 tolerance>0.");
  }
  loss_rmse = true;
}

void Parameters::parse_fcvp_check(const char** param, int num_param)
{
  if (num_param != 2 || !is_valid_int(param[1], &fcvp_check) || fcvp_check < 1 || fcvp_check > 2) {
    PRINT_INPUT_ERROR("fcvp_check requires 1 (sampled) or 2 (all parameters).");
  }
  loss_rmse = true;
}

void Parameters::parse_loss_rmse(const char** param, int num_param)
{
  if (
    num_param != 2 || !is_valid_real(param[1], &fcvp_epsilon) || !(fcvp_epsilon > 0) ||
    !std::isfinite(fcvp_epsilon)) {
    PRINT_INPUT_ERROR("loss_rmse requires a positive finite smoothing epsilon.");
  }
  loss_rmse = true;
}

void Parameters::parse_regularization(const char** param, int num_param)
{
  double value;
  if (num_param != 2 || !is_valid_real(param[1], &value) || value < 0 || !std::isfinite(value)) {
    PRINT_INPUT_ERROR("Regularization weight must be finite and nonnegative.");
  }
  if (strcmp(param[0], "lambda_1") == 0) {
    lambda_1 = value;
  } else {
    lambda_2 = value;
  }
}

void Parameters::parse_seed(const char** param, int num_param)
{
  if (num_param != 2 || !is_valid_int(param[1], &seed) || seed < 0) {
    PRINT_INPUT_ERROR("seed must be a nonnegative integer.");
  }
  is_seed_set = true;
}

void Parameters::parse_weight_decay(const char** param, int num_param)
{
  is_weight_decay_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("weight_decay should have 1 parameter.\n");
  }

  double weight_decay_tmp = 0.0;
  if (!is_valid_real(param[1], &weight_decay_tmp)) {
    PRINT_INPUT_ERROR("Adam with decoupled weight decay should be a number.\n");
  }
  weight_decay = weight_decay_tmp;

  if (weight_decay < 0.0f) {
    PRINT_INPUT_ERROR("weight decay should >= 0.");
  }
}

void Parameters::parse_start_lr(const char** param, int num_param)
{
  is_start_lr_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("start_lr should have 1 parameter.\n");
  }

  double lr_tmp = 0.0;
  if (!is_valid_real(param[1], &lr_tmp)) {
    PRINT_INPUT_ERROR("start learning rate should be a number.\n");
  }
  lr = lr_tmp;
  start_lr = lr;

  if (lr < 0.0f) {
    PRINT_INPUT_ERROR("learning rate should > 0.");
  }
}

void Parameters::parse_stop_lr(const char** param, int num_param)
{
  is_stop_lr_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("stop_lr rate should have 1 parameter.\n");
  }

  double stop_lr_tmp = 0.0;
  if (!is_valid_real(param[1], &stop_lr_tmp)) {
    PRINT_INPUT_ERROR("stop learning rate should be a number.\n");
  }
  stop_lr = stop_lr_tmp;

  if (stop_lr < 0.0f) {
    PRINT_INPUT_ERROR("stop learning rate should > 0.");
  }
}

void Parameters::parse_lambda_e(const char** param, int num_param)
{
  is_lambda_e_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_e should have 1 parameter.\n");
  }

  double lambda_e_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_e_tmp)) {
    PRINT_INPUT_ERROR("Energy loss weight should be a number.\n");
  }
  lambda_e = lambda_e_tmp;

  if (lambda_e < 0.0f) {
    PRINT_INPUT_ERROR("Energy loss weight should >= 0.");
  }
}

void Parameters::parse_lambda_f(const char** param, int num_param)
{
  is_lambda_f_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_f should have 1 parameter.\n");
  }

  double lambda_f_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_f_tmp)) {
    PRINT_INPUT_ERROR("Force loss weight should be a number.\n");
  }
  lambda_f = lambda_f_tmp;

  if (lambda_f < 0.0f) {
    PRINT_INPUT_ERROR("Force loss weight should >= 0.");
  }
}

void Parameters::parse_lambda_v(const char** param, int num_param)
{
  is_lambda_v_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_v should have 1 parameter.\n");
  }

  double lambda_v_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_v_tmp)) {
    PRINT_INPUT_ERROR("Virial loss weight should be a number.\n");
  }
  lambda_v = lambda_v_tmp;

  if (lambda_v < 0.0f) {
    PRINT_INPUT_ERROR("Virial loss weight should >= 0.");
  }
}

void Parameters::parse_lambda_shear(const char** param, int num_param)
{
  is_lambda_shear_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_shear should have 1 parameter.\n");
  }

  double lambda_shear_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_shear_tmp)) {
    PRINT_INPUT_ERROR("Shear virial weight should be a number.\n");
  }
  lambda_shear = lambda_shear_tmp;

  if (lambda_shear < 0.0) {
    PRINT_INPUT_ERROR("Shear virial weight should >= 0.");
  }
}

void Parameters::parse_batch(const char** param, int num_param)
{
  is_batch_set = true;

  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("batch should have 1 or 2 parameters.\n");
  }
  if (!is_valid_int(param[1], &batch_size)) {
    PRINT_INPUT_ERROR("batch size should be an integer.\n");
  }
  if (batch_size < 1) {
    PRINT_INPUT_ERROR("batch size should >= 1.");
  }

  if (num_param == 3) {
    if (!is_valid_int(param[2], &use_full_batch)) {
      PRINT_INPUT_ERROR("use_full_batch should be an integer.\n");
    }
    if (use_full_batch != 0 && use_full_batch != 1) {
      PRINT_INPUT_ERROR("use_full_batch should = 0 or 1.");
    }
  }
}

void Parameters::parse_epoch(const char** param, int num_param)
{
  is_epoch_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("epoch should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &epoch)) {
    PRINT_INPUT_ERROR("maximum number of epochs should be an integer.\n");
  }
  if (epoch < 0) {
    PRINT_INPUT_ERROR("maximum number of epochs should >= 0.");
  } else if (epoch > 10000) {
    PRINT_INPUT_ERROR("maximum number of epochs should <= 10000.");
  }
}

void Parameters::parse_use_typewise_cutoff(const char** param, int num_param)
{
  if (num_param != 1 && num_param != 3) {
    PRINT_INPUT_ERROR("use_typewise_cutoff should have 0 or 2 parameters.\n");
  }
  use_typewise_cutoff = true;
  is_use_typewise_cutoff_set = true;
  typewise_cutoff_radial_factor = 2.5f;
  typewise_cutoff_angular_factor = 2.0f;

  if (num_param == 3) {
    double typewise_cutoff_radial_factor_temp = 0.0;
    if (!is_valid_real(param[1], &typewise_cutoff_radial_factor_temp)) {
      PRINT_INPUT_ERROR("typewise_cutoff_radial_factor should be a number.\n");
    }
    typewise_cutoff_radial_factor = typewise_cutoff_radial_factor_temp;

    double typewise_cutoff_angular_factor_temp = 0.0;
    if (!is_valid_real(param[2], &typewise_cutoff_angular_factor_temp)) {
      PRINT_INPUT_ERROR("typewise_cutoff_angular_factor should be a number.\n");
    }
    typewise_cutoff_angular_factor = typewise_cutoff_angular_factor_temp;
  }

  if (typewise_cutoff_angular_factor < 1.5f) {
    PRINT_INPUT_ERROR("typewise_cutoff_angular_factor must >= 1.5.\n");
  }

  if (typewise_cutoff_radial_factor < typewise_cutoff_angular_factor) {
    PRINT_INPUT_ERROR("typewise_cutoff_radial_factor must >= typewise_cutoff_angular_factor.\n");
  }
}

void Parameters::parse_use_typewise_cutoff_zbl(const char** param, int num_param)
{
  if (num_param != 1 && num_param != 2) {
    PRINT_INPUT_ERROR("use_typewise_cutoff_zbl should have 0 or 1 parameter.\n");
  }
  use_typewise_cutoff_zbl = true;
  is_use_typewise_cutoff_zbl_set = true;
  typewise_cutoff_zbl_factor = 0.65f;

  if (num_param == 2) {
    double typewise_cutoff_zbl_factor_temp = 0.0;
    if (!is_valid_real(param[1], &typewise_cutoff_zbl_factor_temp)) {
      PRINT_INPUT_ERROR("typewise_cutoff_zbl_factor should be a number.\n");
    }
    typewise_cutoff_zbl_factor = typewise_cutoff_zbl_factor_temp;
  }

  if (typewise_cutoff_zbl_factor < 0.5f) {
    PRINT_INPUT_ERROR("typewise_cutoff_zbl_factor must >= 0.5.\n");
  }
}

void Parameters::parse_energy_shift(const char** param, int num_param)
{
  is_energy_shift_set = true;
  if (num_param != 2) {
    PRINT_INPUT_ERROR("energy_shift should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &energy_shift)) {
    PRINT_INPUT_ERROR("energy_shift should be an integer.\n");
  }
  if (energy_shift != 0 && energy_shift != 1) {
    PRINT_INPUT_ERROR("energy_shift should = 0 or 1.");
  }
}

void Parameters::parse_output_descriptor(const char** param, int num_param)
{
  output_descriptor = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("output_descriptor should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &output_descriptor)) {
    PRINT_INPUT_ERROR("output_descriptor should be an integer.\n");
  }
  if (output_descriptor < 0 || output_descriptor > 2) {
    PRINT_INPUT_ERROR("output_descriptor should >= 0 and <= 2.");
  }
}

void Parameters::parse_lr_cos_restart(const char** param, int num_param)
{
  // formats supported:
  // lr_cos_restart enable warmup_epochs initial_period_epochs period_factor decay_factor
  // minimal: lr_cos_restart 1
  if (num_param != 2 && num_param != 6) {
    PRINT_INPUT_ERROR("lr_cos_restart should have 1 or 5 parameters.\n");
  }
  int enable_tmp = 0;
  if (!is_valid_int(param[1], &enable_tmp)) {
    PRINT_INPUT_ERROR("lr_cos_restart enable should be an integer.\n");
  }
  if (enable_tmp != 0 && enable_tmp != 1) {
    PRINT_INPUT_ERROR("lr_cos_restart enable should = 0 or 1.\n");
  }
  lr_restart_enable = enable_tmp;
  is_lr_cos_restart_set = true;
  if (num_param == 6) {
    int warmup_tmp = lr_warmup_epochs;
    int init_period_tmp = lr_restart_initial_period_epochs;
    double period_factor_tmp = lr_restart_period_factor;
    double decay_factor_tmp = lr_restart_decay_factor;

    if (!is_valid_int(param[2], &warmup_tmp)) {
      PRINT_INPUT_ERROR("lr_cos_restart warmup_epochs should be an integer.\n");
    }
    if (!is_valid_int(param[3], &init_period_tmp)) {
      PRINT_INPUT_ERROR("lr_cos_restart initial_period_epochs should be an integer.\n");
    }
    if (!is_valid_real(param[4], &period_factor_tmp)) {
      PRINT_INPUT_ERROR("lr_cos_restart period_factor should be a number.\n");
    }
    if (!is_valid_real(param[5], &decay_factor_tmp)) {
      PRINT_INPUT_ERROR("lr_cos_restart decay_factor should be a number.\n");
    }

    if (warmup_tmp < 0) {
      PRINT_INPUT_ERROR("lr_cos_restart warmup_epochs should >= 0.\n");
    }
    if (init_period_tmp < 1) {
      PRINT_INPUT_ERROR("lr_cos_restart initial_period_epochs should >= 1.\n");
    }
    if (period_factor_tmp <= 0.0) {
      PRINT_INPUT_ERROR("lr_cos_restart period_factor should > 0.\n");
    }
    if (decay_factor_tmp <= 0.0) {
      PRINT_INPUT_ERROR("lr_cos_restart decay_factor should > 0.\n");
    }

    lr_warmup_epochs = warmup_tmp;
    lr_restart_initial_period_epochs = init_period_tmp;
    lr_restart_period_factor = (float)period_factor_tmp;
    lr_restart_decay_factor = (float)decay_factor_tmp;
  }
}
