#!/bin/bash
# UF3 v2 (optimized) vs NEP Inference Speed Benchmark
# New benchmark folder created post-optimization (2026-05-30).
# Compares: UF3 post-opt (re-enabled 3B, float4 positions, ij-basis hoist,
#           direct B-spline, rsqrtf) vs NEP4.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUMD="$SCRIPT_DIR/../../src/gpumd"

CONFIGS=(
    "10x10x1  10 10 1   500"
    "20x20x1  20 20 1   500"
)

POTS=(
    "uf3   SiGe.uf3   UF3_2B_v2"
    "nep   nep.txt    NEP4"
)

REPEAT=3

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}  UF3 v2 vs NEP Inference Speed Benchmark${NC}"
echo -e "${BOLD}============================================${NC}"
echo ""
echo "GPU : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "GPUMD: $GPUMD"
echo "Repeat: ${REPEAT}x per test"
echo ""

build_run_in() {
    local rx=$1 ry=$2 rz=$3 steps=$4 pot_file=$5
    cat > "$SCRIPT_DIR/run.in" <<EOF
replicate $rx $ry $rz
potential   ./$pot_file

time_step   1

fix 1

dump_thermo $steps
dump_exyz   $steps 0 0

ensemble    nvt_ber 400 400 10
run         $steps
EOF
}

run_bench() {
    local out
    out=$(cd "$SCRIPT_DIR" && "$GPUMD" 2>&1)
    if echo "$out" | grep -q "^Error\|Fatal\|Segfault"; then
        echo "ERROR" >&2
        echo "$out" >&2
        return 1
    fi
    echo "$out" | grep "Time used for this run" | grep -oP '[\d.]+(?= second)'
}

declare -A best_times

printf "\n%-12s | %-8s | %10s | %6s | %10s | %14s\n" \
    "System" "Pot" "Atoms" "Steps" "Avg(s)" "Speed(at·st/s)"
printf -- "%-12s-+-%-8s-+-%-10s-+-%-6s-+-%-10s-+-%-14s\n" \
    "------------" "--------" "----------" "------" "----------" "--------------"

for cfg in "${CONFIGS[@]}"; do
    read -r label rx ry rz steps <<< "$cfg"
    atoms=$((3460 * rx * ry * rz))

    for pot in "${POTS[@]}"; do
        read -r pot_label pot_file pot_desc <<< "$pot"
        build_run_in "$rx" "$ry" "$rz" "$steps" "$pot_file"

        times=()
        for _ in $(seq 1 $REPEAT); do
            t=$(run_bench) && times+=("$t")
            rm -f "$SCRIPT_DIR/neighbor.out"
        done

        if [ ${#times[@]} -gt 0 ]; then
            sum=0
            for t in "${times[@]}"; do sum=$(echo "$sum + $t" | bc -l); done
            avg=$(echo "scale=6; $sum / ${#times[@]}" | bc -l)
            speed=$(echo "scale=4; $atoms * $steps / $avg" | bc -l)
            speed_fmt=$(printf "%.3e" "$speed")
            printf "%-12s | %-8s | %10d | %6d | %10.4f | %14s\n" \
                "$label" "$pot_label" "$atoms" "$steps" "$avg" "$speed_fmt"
            best_times["${label}_${pot_label}"]=$avg
        else
            printf "%-12s | %-8s | %10d | %6d | %10s | %14s\n" \
                "$label" "$pot_label" "$atoms" "$steps" "FAILED" "---"
        fi
    done
    echo ""
done

echo -e "\n${GREEN}Benchmark complete.${NC}"
