// SPDX-License-Identifier: GPL-3.0-or-later
// Compile against the external, pinned NEP_CPU tree; no upstream code is copied.
#include "nep.h"
#include "main_gnep/fcvp_solver.cuh"
#include <algorithm>
#include <exception>
#include <string>

static thread_local std::string last_error;
extern "C" {
const char* fcvp_error() { return last_error.c_str(); }
void* fcvp_load(const char* path) {
  try { auto* n = new NEP; n->init_from_file(path, false); return n; }
  catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
void fcvp_free(void* handle) { delete static_cast<NEP*>(handle); }
int fcvp_predict(void* handle, int n, const int* types, const double* box,
                 const double* positions, int p, const int* indices,
                 const double* beta, double* out) {
  auto& nep = *static_cast<NEP*>(handle);
  try {
    for (int j=0; j<p; ++j) nep.parameters.at(indices[j]) = beta[j];
    nep.update_potential(nep.parameters.data(), nep.annmb);
    std::vector<int> t(types, types+n);
    std::vector<double> b(box, box+9), r(positions, positions+3*n);
    std::vector<double> e(n), f(3*n), v(9*n);
    nep.compute(t,b,r,e,f,v);
    out[0]=0;
    for (double x:e) out[0]+=x/n;
    std::copy(f.begin(),f.end(),out+1);
    const int components[6]={0,4,8,1,5,6};
    for (int c=0; c<6; ++c) {
      out[1+3*n+c]=0;
      for (int i=0; i<n; ++i) out[1+3*n+c]+=v[components[c]*n+i]/n;
    }
    return 0;
  } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
int fcvp_solve(int p, int groups, const double* h, const double* g, const double* yy,
               const double* weights, double* beta, int iterations, double epsilon,
               double tolerance, int total, double lambda1, double lambda2,
               double fixed1, double fixed2, double* history, int* count) {
  try {
    std::vector<fcvp::Group> data(groups);
    for (int k=0;k<groups;++k) {
      data[k].h.assign(h+k*p*p,h+(k+1)*p*p);
      data[k].g.assign(g+k*p,g+(k+1)*p);
      data[k].yy=yy[k]; data[k].lambda=weights[k];
    }
    fcvp::Options options;
    options.iterations=iterations; options.epsilon=epsilon; options.tolerance=tolerance;
    options.total_parameters=total; options.lambda1=lambda1; options.lambda2=lambda2;
    options.fixed_l1=fixed1; options.fixed_l2_squared=fixed2;
    auto result=fcvp::solve(data,std::vector<double>(beta,beta+p),options);
    std::copy(result.beta.begin(),result.beta.end(),beta);
    std::copy(result.objectives.begin(),result.objectives.end(),history);
    *count=result.objectives.size();
    return 0;
  } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
}
