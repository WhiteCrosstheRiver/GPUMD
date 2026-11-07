// fix shell, fix type and fix group: a frozen set built from several sources.
// Grouping method 0 is rewritten for the run (0 unfixed, 1 frozen) and restored afterwards,
// so "fix <group>" and "fix <method> <group>" keep upstream's meaning.

#include "../../extensions/common/surface/substrate_shell.cuh"
#include "../../extensions/common/surface/substrate_shell_v2.cuh"
#include "integrate.cuh"
#include "model/atom.cuh"
#include "utilities/error.cuh"
#include "utilities/common.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <unordered_set>

static bool is_fix_shell_keyword(const std::string& token)
{
  return token == "substrate" || token == "cutoff" || token == "offset" || token == "type" ||
         token == "region" || token == "surface" || token == "cell" || token == "mode" ||
         token == "connect" || token == "every" || token == "reset";
}

static void apply_group0_labels(Group& group, const std::vector<int>& label, const int number)
{
  const int N = static_cast<int>(label.size());
  group.number = number;
  group.cpu_label = label;
  group.find_size(N, 0);
  group.find_contents(N);
  group.label.resize(N);
  group.size.resize(number);
  group.size_sum.resize(number);
  group.contents.resize(N);
  group.label.copy_from_host(group.cpu_label.data());
  group.size.copy_from_host(group.cpu_size.data());
  group.size_sum.copy_from_host(group.cpu_size_sum.data());
  group.contents.copy_from_host(group.cpu_contents.data());
}

static void parse_fix_shell_v2(
  FixUnionState& state,
  Atom& atom,
  Box& box,
  const int axis,
  const double cutoff,
  const double offset,
  const double surface_radius,
  const double cell_size,
  const bool grow,
  const shell2::Connect connect,
  const bool has_every,
  const int every,
  const bool reset,
  const std::unordered_set<std::string>& types,
  const std::vector<char>* eligible,
  const double* region_cubic)
{
  if (!grow && (connect == shell2::Connect::Frozen || has_every || reset)) {
    PRINT_INPUT_ERROR("fix shell: connect frozen, every and reset need mode grow.");
  }
  shell2::Params p;
  p.axis = axis;
  p.cutoff = cutoff;
  p.offset = offset;
  p.surface_radius = surface_radius;
  p.cell_size = cell_size;
  p.mode = grow ? shell2::Mode::Grow : shell2::Mode::Recede;
  p.connect = connect;
  p.every = every;
  p.reset = reset;
  p.has_region = region_cubic != nullptr;
  if (p.has_region) {
    std::copy(region_cubic, region_cubic + 6, p.region);
  }
  p.types.assign(types.begin(), types.end());
  std::sort(p.types.begin(), p.types.end());

  const int N = atom.number_of_atoms;
  shell2::Report rep;
  if (grow) {
    if (!state.grow_state) {
      state.grow_state = std::make_shared<shell2::GrowState>();
    }
    shell2::update_grow(atom, box, p, eligible, *state.grow_state, rep);
    if (!rep.ok) {
      PRINT_INPUT_ERROR("fix shell found no substrate connected to the minimum along that axis.");
    }
    for (int n = 0; n < N; ++n) {
      if (atom.cpu_shell_frozen[n]) {
        state.frozen[n] = 1;
      }
    }
    state.grow_committed = true;
  } else {
    static shell2::Workspace ws;
    std::vector<char> frozen;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = shell2::classify_full(atom, box, p, eligible, ws, frozen, rep);
    rep.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) {
      PRINT_INPUT_ERROR("fix shell found no substrate connected to the minimum along that axis.");
    }
    for (int n = 0; n < N; ++n) {
      if (frozen[n]) {
        state.frozen[n] = 1;
      }
    }
  }
  shell2::print_report(p, rep);
}

void Integrate::commit_fix_union(std::vector<Group>& group, Atom& atom)
{
  const int N = atom.number_of_atoms;
  std::vector<int> label(N);
  for (int n = 0; n < N; ++n) {
    label[n] = fix_union.frozen[n] ? 1 : 0;
  }
  if (group.empty()) {
    group.resize(1);
    fix_union.created_group = true;
  }
  apply_group0_labels(group[0], label, 2);
  atom.cpu_fixed = fix_union.frozen;
  fixed_grouping_method = 0;
  fixed_group = 1;
  printf(
    "    unfixed %d atoms, frozen %d atoms (union, group 1).\n",
    group[0].cpu_size[0],
    group[0].cpu_size[1]);
}

void Integrate::update_fix_union(const Atom& atom)
{
  if (fix_union.active) {
    fix_union.frozen = atom.cpu_fixed;
  }
}

void Integrate::finish_fix_union(std::vector<Group>& group)
{
  if (fix_union.grow_state && !fix_union.grow_committed) {
    // The ratchet assumes its atoms never move, and this run may have moved them.
    fix_union.grow_state->valid = false;
  }
  fix_union.grow_committed = false;
  if (fix_union.active) {
    if (fix_union.created_group) {
      group.clear();
    } else {
      const std::vector<int> user_label = group[0].cpu_label_user;
      group[0].cpu_label_user.clear();
      int number = fix_union.user_number;
      for (const int label : user_label) {
        number = std::max(number, label + 1);
      }
      apply_group0_labels(group[0], user_label, number);
    }
  }
  fix_union.active = false;
  fix_union.created_group = false;
  fix_union.frozen.clear();
}

void Integrate::parse_fix_union(
  const std::vector<std::string>& tokens, std::vector<Group>& group, Atom& atom, Box& box)
{
  const int num_param = static_cast<int>(tokens.size());
  const int N = atom.number_of_atoms;
  if (N < 1) {
    PRINT_INPUT_ERROR("fix needs existing atoms.");
  }
  atom.cpu_fixed.resize(N, 0);
  atom.cpu_shell_frozen.resize(N, 0);

  if (!fix_union.active) {
    fix_union.frozen.assign(N, 0);
    if (fixed_group >= 0) {
      // An earlier plain fix in this block joins the union.
      const std::vector<int>& plain = group[fixed_grouping_method].cpu_label;
      for (int n = 0; n < N; ++n) {
        fix_union.frozen[n] = plain[n] == fixed_group ? 1 : 0;
      }
    }
    if (!group.empty()) {
      group[0].cpu_label_user = group[0].cpu_label;
      fix_union.user_number = group[0].number;
    }
    fix_union.active = true;
  }

  if (tokens[1] == "shell") {
    int axis = -1;
    double cutoff = 0.0;
    double offset = 0.0;
    bool has_substrate = false;
    bool has_cutoff = false;
    bool has_offset = false;
    bool has_region = false;
    std::unordered_set<std::string> types;
    double region_cubic[6];
    // v2 (substrate_shell_v2.cuh); any of these keywords selects the v2 path
    bool use_v2 = false;
    bool has_surface_radius = false;
    double surface_radius = 0.0;
    double cell_size = 1.0;
    bool grow = false;
    bool has_connect = false;
    shell2::Connect connect = shell2::Connect::Full;
    bool has_every = false;
    int every = 1;
    bool reset = false;

    int i = 2;
    while (i < num_param) {
      if (tokens[i] == "surface") {
        if (i + 2 >= num_param || tokens[i + 1] != "radius") {
          PRINT_INPUT_ERROR("Usage: fix shell ... surface radius <R>");
        }
        if (!is_valid_real(tokens[i + 2], &surface_radius) || surface_radius < 0.0) {
          PRINT_INPUT_ERROR("fix shell surface radius should be a number >= 0.");
        }
        has_surface_radius = true;
        use_v2 = true;
        i += 3;
      } else if (tokens[i] == "cell") {
        if (i + 1 >= num_param || !is_valid_real(tokens[i + 1], &cell_size) || cell_size <= 0.0) {
          PRINT_INPUT_ERROR("Usage: fix shell ... cell <c> with c > 0.");
        }
        use_v2 = true;
        i += 2;
      } else if (tokens[i] == "mode") {
        if (i + 1 >= num_param) {
          PRINT_INPUT_ERROR("Usage: fix shell ... mode recede|grow");
        }
        if (tokens[i + 1] == "grow") {
          grow = true;
        } else if (tokens[i + 1] == "recede") {
          grow = false;
        } else {
          PRINT_INPUT_ERROR("fix shell mode should be recede or grow.");
        }
        use_v2 = true;
        i += 2;
      } else if (tokens[i] == "connect") {
        if (i + 1 >= num_param) {
          PRINT_INPUT_ERROR("Usage: fix shell ... connect full|frozen|none");
        }
        if (tokens[i + 1] == "full") {
          connect = shell2::Connect::Full;
        } else if (tokens[i + 1] == "frozen") {
          connect = shell2::Connect::Frozen;
        } else if (tokens[i + 1] == "none") {
          connect = shell2::Connect::None;
        } else {
          PRINT_INPUT_ERROR("fix shell connect should be full, frozen, or none.");
        }
        has_connect = true;
        use_v2 = true;
        i += 2;
      } else if (tokens[i] == "every") {
        if (i + 1 >= num_param || !is_valid_int(tokens[i + 1], &every) || every < 1) {
          PRINT_INPUT_ERROR("Usage: fix shell ... every <K> with K >= 1.");
        }
        has_every = true;
        use_v2 = true;
        i += 2;
      } else if (tokens[i] == "reset") {
        reset = true;
        use_v2 = true;
        i += 1;
      } else if (tokens[i] == "substrate") {
        if (i + 1 >= num_param) {
          PRINT_INPUT_ERROR("Usage: fix shell substrate <x|y|z> ...");
        }
        if (tokens[i + 1] == "x" || tokens[i + 1] == "X") {
          axis = 0;
        } else if (tokens[i + 1] == "y" || tokens[i + 1] == "Y") {
          axis = 1;
        } else if (tokens[i + 1] == "z" || tokens[i + 1] == "Z") {
          axis = 2;
        } else {
          PRINT_INPUT_ERROR("fix shell substrate should be x, y, or z.");
        }
        has_substrate = true;
        i += 2;
      } else if (tokens[i] == "cutoff") {
        if (i + 1 >= num_param) {
          PRINT_INPUT_ERROR("Usage: fix shell ... cutoff <r>");
        }
        if (!is_valid_real(tokens[i + 1], &cutoff)) {
          PRINT_INPUT_ERROR("fix shell cutoff should be a number.");
        }
        if (cutoff <= 0.0) {
          PRINT_INPUT_ERROR("fix shell cutoff should be > 0.");
        }
        has_cutoff = true;
        i += 2;
      } else if (tokens[i] == "offset") {
        if (i + 1 >= num_param) {
          PRINT_INPUT_ERROR("Usage: fix shell ... offset <thickness>");
        }
        if (!is_valid_real(tokens[i + 1], &offset)) {
          PRINT_INPUT_ERROR("fix shell offset should be a number.");
        }
        if (offset <= 0.0) {
          PRINT_INPUT_ERROR("fix shell offset should be > 0.");
        }
        has_offset = true;
        i += 2;
      } else if (tokens[i] == "type") {
        i += 1;
        int n_type = 0;
        while (i < num_param && !is_fix_shell_keyword(tokens[i])) {
          types.insert(tokens[i]);
          ++n_type;
          ++i;
        }
        if (n_type == 0) {
          PRINT_INPUT_ERROR("Usage: fix shell ... type <symbol> [symbol ...]");
        }
      } else if (tokens[i] == "region") {
        if (i + 7 >= num_param || tokens[i + 1] != "cubic") {
          PRINT_INPUT_ERROR(
            "Usage: fix shell ... region cubic <xmin> <xmax> <ymin> <ymax> <zmin> <zmax>");
        }
        for (int k = 0; k < 6; ++k) {
          if (!is_valid_real(tokens[i + 2 + k], region_cubic + k)) {
            PRINT_INPUT_ERROR("fix shell region cubic bounds should be numbers.");
          }
        }
        if (region_cubic[0] > region_cubic[1] || region_cubic[2] > region_cubic[3] ||
            region_cubic[4] > region_cubic[5]) {
          PRINT_INPUT_ERROR("fix shell region cubic min should be <= max for each component.");
        }
        has_region = true;
        i += 8;
      } else {
        PRINT_INPUT_ERROR(
          "Unknown keyword in fix shell. Expected substrate, cutoff, offset, type, region, "
          "surface, cell, mode, connect, every, or reset.");
      }
    }

    if (!has_substrate || !has_cutoff || !has_offset) {
      PRINT_INPUT_ERROR(
        "Usage: fix shell substrate <x|y|z> cutoff <r> offset <d> [type <sym> ...] "
        "[region cubic <xmin> <xmax> <ymin> <ymax> <zmin> <zmax>]");
    }

    std::vector<char> eligible;
    const std::vector<char>* eligible_ptr = nullptr;
    if (!types.empty()) {
      eligible.assign(N, 0);
      int n_ok = 0;
      for (int n = 0; n < N; ++n) {
        if (types.count(atom.cpu_atom_symbol[n])) {
          eligible[n] = 1;
          ++n_ok;
        }
      }
      if (n_ok == 0) {
        PRINT_INPUT_ERROR("fix shell type matched no atoms.");
      }
      eligible_ptr = &eligible;
    }

    if (use_v2) {
      parse_fix_shell_v2(
        fix_union,
        atom,
        box,
        axis,
        cutoff,
        offset,
        has_surface_radius ? surface_radius : (grow ? 3.0 : 0.0),
        cell_size,
        grow,
        has_connect ? connect : (grow ? shell2::Connect::Frozen : shell2::Connect::Full),
        has_every,
        every,
        reset,
        types,
        eligible_ptr,
        has_region ? region_cubic : nullptr);
      commit_fix_union(group, atom);
      return;
    }

    std::vector<int> label;
    if (!classify_substrate_shell(
          atom,
          box,
          axis,
          cutoff,
          offset,
          eligible_ptr,
          has_region ? region_cubic : nullptr,
          label)) {
      PRINT_INPUT_ERROR("fix shell found no substrate connected to the minimum along that axis.");
    }
    for (int n = 0; n < N; ++n) {
      if (label[n] == 1) {
        fix_union.frozen[n] = 1;
      }
    }

    const char axis_name[3] = {'x', 'y', 'z'};
    printf(
      "Fix shell (union): substrate along %c, cutoff = %g, unfix shear offset = %g A.\n",
      axis_name[axis],
      cutoff,
      offset);
    if (!types.empty()) {
      printf("    substrate types:");
      for (const auto& symbol : types) {
        printf(" %s", symbol.c_str());
      }
      printf("\n");
    }
    if (has_region) {
      printf(
        "    region cubic [%g, %g] [%g, %g] [%g, %g]\n",
        region_cubic[0],
        region_cubic[1],
        region_cubic[2],
        region_cubic[3],
        region_cubic[4],
        region_cubic[5]);
    }
    commit_fix_union(group, atom);
    return;
  }

  std::vector<int> group_ids;
  std::unordered_set<std::string> types;
  int i = 1;
  while (i < num_param) {
    if (tokens[i] == "type") {
      int n_type = 0;
      ++i;
      while (i < num_param && tokens[i] != "group") {
        types.insert(tokens[i]);
        ++n_type;
        ++i;
      }
      if (n_type == 0) {
        PRINT_INPUT_ERROR("Usage: fix type <symbol> [symbol ...]");
      }
    } else if (tokens[i] == "group") {
      int n_group = 0;
      ++i;
      while (i < num_param && tokens[i] != "type") {
        int id = 0;
        if (!is_valid_int(tokens[i], &id)) {
          PRINT_INPUT_ERROR("Usage: fix group <group_id> [group_id ...]");
        }
        if (group.empty() || fix_union.created_group || id < 0 || id >= fix_union.user_number) {
          PRINT_INPUT_ERROR("fix group needs a group ID of grouping method 0.");
        }
        group_ids.push_back(id);
        ++n_group;
        ++i;
      }
      if (n_group == 0) {
        PRINT_INPUT_ERROR("Usage: fix group <group_id> [group_id ...]");
      }
    } else {
      PRINT_INPUT_ERROR("Usage: fix group <group_id> [...] [type <symbol> ...]");
    }
  }

  const std::unordered_set<int> id_set(group_ids.begin(), group_ids.end());
  int n_added = 0;
  for (int n = 0; n < N; ++n) {
    const bool by_group = !id_set.empty() && id_set.count(group[0].cpu_label_user[n]) > 0;
    const bool by_type = !types.empty() && types.count(atom.cpu_atom_symbol[n]) > 0;
    if ((by_group || by_type) && !fix_union.frozen[n]) {
      fix_union.frozen[n] = 1;
      ++n_added;
    }
  }
  printf("Fix union: +%d atoms.\n", n_added);
  commit_fix_union(group, atom);
}
