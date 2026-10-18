# Experimental SOAP optimizer

This directory keeps the SOAP training experiment separate from the default GNEP build. The four files in `overlay/` are a complete, tested replacement for their namesakes in `src/main_gnep/`. Copy them into an isolated checkout before building `gnep`; the default source tree is unchanged on this branch.

The optimizer is selected through `GPUMD_EXPERIMENT_OPTIMIZER=soap`. `adam_soap_control` selects the matched Adam control. With the variable unset, the ordinary Adam path runs. The experimental modes require `loss_rmse` and cannot be combined with FCVP. Eigen 3.4.1 is needed to compile the host SOAP matrix operations; add its include directory to the `gnep` compiler flags. The accepted test build used C++17, `-O3`, and `-arch=sm_80` on A100.

The experiment uses the same NEP descriptor, model format, and gradient pipeline as GNEP. `experimental_soap.cuh` contains the host optimizer; `soap_parameter_groups.cuh` maps NEP parameters into tensor groups; `adam.cu` and `adam.cuh` contain the opt-in integration. `LICENSE.SOAP` records the referenced implementation's MIT license and provenance. The independent implementation and its mathematical differences are described in the header.

Tests in `tests/` cover the optimizer recurrence, parameter-group layout, independent NumPy comparison, and a bounded GPU acceptance run. The GPU test also checks the default Adam path against the baseline program and includes E/F/V and ZBL checks. See `verify_soap_gpu.py --help` and `verify_soap_reference.py --help` for arguments.

The style-aligned version passed 12/12 GPU checks and three byte-identical CPU traces against the pre-format experiment. The corresponding A100 validation is archived outside this repository at `/home/fzm001/projects/GPUMD-style-check-20261018.rcvYuO/style_validation/soap_gpu/` and `soap_cpu/`. No large convergence benchmark was repeated for this formatting change.
