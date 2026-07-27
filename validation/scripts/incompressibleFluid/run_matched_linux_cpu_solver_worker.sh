#!/usr/bin/env bash
set -euo pipefail

mode="run"
rust_toolchain="1.94.0"
cpu_set="2"
build_variant="portable"
warmup_runs="2"
measured_runs="9"
pressure_solver="gamg"
source_archive=""
source_archive_sha256=""
templates_archive=""
templates_archive_sha256=""
manifest_path=""
output_archive=""
source_commit=""
source_tree=""
keep_workspace="0"

fail() {
    printf 'linux parity worker: %s\n' "$*" >&2
    exit 1
}

require_value() {
    [[ $# -ge 2 && -n "$2" ]] || fail "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preflight-only) mode="preflight"; shift ;;
        --rust-toolchain) require_value "$@"; rust_toolchain="$2"; shift 2 ;;
        --cpu-set) require_value "$@"; cpu_set="$2"; shift 2 ;;
        --build-variant) require_value "$@"; build_variant="$2"; shift 2 ;;
        --warmup-runs) require_value "$@"; warmup_runs="$2"; shift 2 ;;
        --measured-runs) require_value "$@"; measured_runs="$2"; shift 2 ;;
        --pressure-solver) require_value "$@"; pressure_solver="$2"; shift 2 ;;
        --source-archive) require_value "$@"; source_archive="$2"; shift 2 ;;
        --source-archive-sha256) require_value "$@"; source_archive_sha256="$2"; shift 2 ;;
        --templates-archive) require_value "$@"; templates_archive="$2"; shift 2 ;;
        --templates-archive-sha256) require_value "$@"; templates_archive_sha256="$2"; shift 2 ;;
        --manifest) require_value "$@"; manifest_path="$2"; shift 2 ;;
        --output-archive) require_value "$@"; output_archive="$2"; shift 2 ;;
        --source-commit) require_value "$@"; source_commit="$2"; shift 2 ;;
        --source-tree) require_value "$@"; source_tree="$2"; shift 2 ;;
        --keep-workspace) keep_workspace="1"; shift ;;
        *) fail "unknown argument: $1" ;;
    esac
done

if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

for command_name in bash cargo rustc jq git tar sha256sum taskset flock findmnt python3 /usr/bin/time; do
    command -v "$command_name" >/dev/null 2>&1 || fail "required command was not found: $command_name"
done

[[ -f /opt/openfoam13/etc/bashrc ]] || fail "OpenFOAM Foundation 13 bashrc was not found"
# shellcheck disable=SC1091
set +e +u
source /opt/openfoam13/etc/bashrc >/dev/null 2>&1
set -euo pipefail
[[ "${WM_PROJECT_VERSION:-}" == "13" ]] || fail "WM_PROJECT_VERSION is not 13"
command -v foamRun >/dev/null 2>&1 || fail "foamRun was not found after sourcing OpenFOAM 13"

[[ "$cpu_set" =~ ^[0-9]+([,-][0-9]+)*$ ]] || fail "invalid CPU set: $cpu_set"
taskset -c "$cpu_set" true >/dev/null 2>&1 || fail "CPU set is not available: $cpu_set"
[[ "$warmup_runs" =~ ^[0-9]+$ ]] || fail "warmup runs must be a non-negative integer"
[[ "$measured_runs" =~ ^[1-9][0-9]*$ ]] || fail "measured runs must be a positive integer"
[[ "$build_variant" == "portable" || "$build_variant" == "native" ]] || fail "unsupported build variant: $build_variant"
[[ "$pressure_solver" == "pcg" || "$pressure_solver" == "gamg" ]] || fail "unsupported pressure solver: $pressure_solver"

rustc_version="$(rustc "+$rust_toolchain" --version 2>/dev/null || true)"
[[ "$rustc_version" == "rustc $rust_toolchain "* ]] || fail "exact Rust $rust_toolchain is not installed"
cargo_version="$(cargo "+$rust_toolchain" --version 2>/dev/null || true)"
[[ -n "$cargo_version" ]] || fail "Cargo for Rust $rust_toolchain is not installed"

home_fstype="$(findmnt -T "$HOME" -n -o FSTYPE | tr -d '[:space:]')"
[[ "$home_fstype" == "ext4" ]] || fail "WSL home is not on ext4 (found '$home_fstype')"

if [[ "$mode" == "preflight" ]]; then
    printf 'preflight=pass\n'
    printf 'rustc=%s\n' "$rustc_version"
    printf 'cargo=%s\n' "$cargo_version"
    printf 'openfoam=%s\n' "${WM_PROJECT_VERSION}"
    printf 'wm_options=%s\n' "${WM_OPTIONS:-unknown}"
    printf 'filesystem=%s\n' "$home_fstype"
    printf 'cpu_set=%s\n' "$cpu_set"
    exit 0
fi

for required_file in "$source_archive" "$templates_archive" "$manifest_path"; do
    [[ -f "$required_file" ]] || fail "required staged input was not found: $required_file"
done
[[ -n "$source_archive_sha256" ]] || fail "source archive SHA-256 was not supplied"
[[ -n "$templates_archive_sha256" ]] || fail "templates archive SHA-256 was not supplied"
[[ -n "$output_archive" ]] || fail "output archive path was not supplied"
[[ -n "$source_commit" && -n "$source_tree" ]] || fail "source commit and tree are required"

cache_root="$HOME/.cache/ferrumcfd-linux-parity"
mkdir -p "$cache_root"
exec 9>"$cache_root/benchmark.lock"
flock -n 9 || fail "another Linux parity benchmark is active"
workspace="$(mktemp -d "$cache_root/run.XXXXXXXX")"
completed="0"

cleanup() {
    status=$?
    if [[ "$status" -eq 0 && "$completed" == "1" && "$keep_workspace" == "0" ]]; then
        case "$workspace" in
            "$cache_root"/run.*) rm -rf -- "$workspace" ;;
            *) printf 'refusing unsafe workspace cleanup: %s\n' "$workspace" >&2 ;;
        esac
    else
        printf 'linux parity workspace preserved: %s\n' "$workspace" >&2
    fi
}
trap cleanup EXIT

source_root="$workspace/source"
templates_root="$workspace/templates"
results_root="$workspace/export/raw"
metadata_root="$workspace/export/metadata"
mkdir -p "$source_root" "$templates_root" "$results_root" "$metadata_root"

validate_archive() {
    local archive_path="$1" description="$2"
    python3 - "$archive_path" "$description" <<'PY'
import pathlib
import re
import sys
import tarfile

archive_path, description = sys.argv[1:]
with tarfile.open(archive_path, mode="r:*") as archive:
    members = archive.getmembers()
    if not members:
        raise SystemExit(f"{description} archive is empty")
    for member in members:
        name = member.name.replace("\\", "/")
        path = pathlib.PurePosixPath(name)
        if (
            not name
            or name.startswith("/")
            or re.match(r"^[A-Za-z]:(?:/|$)", name)
            or ".." in path.parts
        ):
            raise SystemExit(f"{description} archive contains an unsafe path: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"{description} archive contains a non-regular entry: {member.name}")
PY
}

cp -- "$source_archive" "$workspace/source.tar"
actual_source_sha256="$(sha256sum "$workspace/source.tar" | awk '{print $1}')"
[[ "$actual_source_sha256" == "$source_archive_sha256" ]] || fail "source archive SHA-256 changed while staging"
validate_archive "$workspace/source.tar" "source"
tar --no-same-owner --no-same-permissions -xf "$workspace/source.tar" -C "$source_root"
cp -- "$templates_archive" "$workspace/templates.tar"
actual_templates_sha256="$(sha256sum "$workspace/templates.tar" | awk '{print $1}')"
[[ "$actual_templates_sha256" == "$templates_archive_sha256" ]] || fail "templates archive SHA-256 changed while staging"
validate_archive "$workspace/templates.tar" "templates"
tar --no-same-owner --no-same-permissions -xf "$workspace/templates.tar" -C "$templates_root"
cp -- "$manifest_path" "$metadata_root/input-manifest.json"

source_fstype="$(findmnt -T "$source_root" -n -o FSTYPE | tr -d '[:space:]')"
results_fstype="$(findmnt -T "$results_root" -n -o FSTYPE | tr -d '[:space:]')"
[[ "$source_fstype" == "ext4" && "$results_fstype" == "ext4" ]] || fail "source and results must remain on ext4"

for case_name in $(jq -r '.cases[].name' "$metadata_root/input-manifest.json"); do
    for engine_name in ferrum openfoam; do
        case_root="$templates_root/$case_name/$engine_name"
        [[ -d "$case_root" ]] || fail "staged case was not found: $case_root"
        for mesh_file in points faces owner neighbour boundary; do
            expected_hash="$(jq -r --arg case "$case_name" --arg file "$mesh_file" '.cases[] | select(.name == $case) | .canonicalPolyMeshSha256[$file]' "$metadata_root/input-manifest.json")"
            actual_hash="$(sha256sum "$case_root/constant/polyMesh/$mesh_file" | awk '{print $1}')"
            [[ "$actual_hash" == "$expected_hash" ]] || fail "$case_name $engine_name polyMesh differs in $mesh_file"
        done
    done
    for shared_name in velocity fvSchemes fvSolution; do
        case "$shared_name" in
            velocity) relative_path="0/U" ;;
            fvSchemes) relative_path="system/fvSchemes" ;;
            fvSolution) relative_path="system/fvSolution" ;;
        esac
        expected_hash="$(jq -r --arg case "$case_name" --arg file "$shared_name" '.cases[] | select(.name == $case) | .sharedFileSha256[$file]' "$metadata_root/input-manifest.json")"
        for engine_name in ferrum openfoam; do
            actual_hash="$(sha256sum "$templates_root/$case_name/$engine_name/$relative_path" | awk '{print $1}')"
            [[ "$actual_hash" == "$expected_hash" ]] || fail "$case_name $engine_name shared file differs: $relative_path"
        done
    done
done

target_root="$source_root/target/linux-parity-$build_variant"
build_timing="$metadata_root/build-timing.env"
build_log="$metadata_root/cargo-build-release.log"
build_format=$'elapsed_s=%e\nuser_s=%U\nsystem_s=%S\nmax_rss_kb=%M\nexit=%x'
set +e
if [[ "$build_variant" == "native" ]]; then
    (
        cd "$source_root"
        CARGO_TARGET_DIR="$target_root" RUSTFLAGS="-C target-cpu=native" \
            /usr/bin/time -q -f "$build_format" -o "$build_timing" \
            cargo "+$rust_toolchain" build --locked --release -p ferrum-run --bin ferrumRun \
            >"$build_log" 2>&1
    )
else
    (
        cd "$source_root"
        CARGO_TARGET_DIR="$target_root" \
            /usr/bin/time -q -f "$build_format" -o "$build_timing" \
            cargo "+$rust_toolchain" build --locked --release -p ferrum-run --bin ferrumRun \
            >"$build_log" 2>&1
    )
fi
build_status=$?
set -e
[[ "$build_status" -eq 0 ]] || fail "Ferrum Linux release build failed; see $build_log"
binary="$target_root/release/ferrumRun"
[[ -x "$binary" ]] || fail "Linux Ferrum executable was not produced: $binary"

printf '%s\n' "$source_commit" >"$metadata_root/source-commit.txt"
printf '%s\n' "$source_tree" >"$metadata_root/source-tree.txt"
printf '%s\n' "$actual_source_sha256" >"$metadata_root/source-archive-sha256.txt"
printf '%s\n' "$actual_templates_sha256" >"$metadata_root/templates-archive-sha256.txt"
sha256sum "$source_root/Cargo.lock" | awk '{print $1}' >"$metadata_root/cargo-lock-sha256.txt"
sha256sum "$binary" | awk '{print $1}' >"$metadata_root/ferrum-binary-sha256.txt"
rustc "+$rust_toolchain" -vV >"$metadata_root/rustc-vv.txt"
cargo "+$rust_toolchain" --version >"$metadata_root/cargo-version.txt"
uname -a >"$metadata_root/uname.txt"
grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '"' >"$metadata_root/distro-release.txt"
lscpu >"$metadata_root/lscpu.txt"
awk -F: '/model name/ {sub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo >"$metadata_root/cpu-model.txt"
printf '%s\n' "$cpu_set" >"$metadata_root/cpu-set.txt"
first_cpu="${cpu_set%%[,-]*}"
cat "/sys/devices/system/cpu/cpu$first_cpu/topology/thread_siblings_list" >"$metadata_root/cpu-siblings.txt"
printf '%s\n' "$home_fstype" >"$metadata_root/filesystem-type.txt"
printf '%s\n' "${WM_PROJECT_VERSION}" >"$metadata_root/openfoam-version.txt"
printf '%s\n' "${WM_OPTIONS:-unknown}" >"$metadata_root/openfoam-build-options.txt"
sha256sum "$(command -v foamRun)" | awk '{print $1}' >"$metadata_root/openfoam-binary-sha256.txt"
printf '%s\n' "$build_variant" >"$metadata_root/build-variant.txt"
printf '%s\n' "$workspace" >"$metadata_root/workspace-path.txt"

time_format=$'elapsed_s=%e\nuser_s=%U\nsystem_s=%S\nmax_rss_kb=%M\nexit=%x'
thread_environment=(
    LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
)
printf 'case\tkind\tordinal\tposition\tengine\n' >"$metadata_root/run-order.tsv"

numeric_output_count() {
    find "$1" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
        awk '$0 != "0" && $0 ~ /^[0-9]+([.][0-9]+)?$/ {count++} END {print count+0}'
}

run_ferrum() {
    local case_name="$1" kind="$2" ordinal="$3" fixed_iterations="$4" run_root="$5"
    local working_case="$run_root/case" log_path="$run_root/ferrum.log"
    local report_path="$run_root/solve-report.json" timing_path="$run_root/process-time.env"
    mkdir -p "$working_case"
    cp -a "$templates_root/$case_name/ferrum/." "$working_case/"
    set +e
    (
        cd "$run_root"
        /usr/bin/time -q -f "$time_format" -o "$timing_path" \
            taskset -c "$cpu_set" env "${thread_environment[@]}" \
            "$binary" -solver incompressibleFluid -case "$working_case" \
            --minSimpleIterations "$fixed_iterations" \
            --maxSimpleIterations "$fixed_iterations" \
            --solveReportJson "$report_path" >"$log_path" 2>&1
    )
    local status=$?
    set -e
    [[ "$status" -eq 0 ]] || fail "Ferrum run failed for $case_name ($kind $ordinal)"
    [[ "$(numeric_output_count "$working_case")" -eq 0 ]] || fail "Ferrum wrote an unexpected time directory"
}

run_openfoam() {
    local case_name="$1" kind="$2" ordinal="$3" run_root="$4"
    local working_case="$run_root/case" log_path="$run_root/openfoam.log"
    local timing_path="$run_root/process-time.env"
    mkdir -p "$working_case"
    cp -a "$templates_root/$case_name/openfoam/." "$working_case/"
    set +e
    (
        cd "$working_case"
        /usr/bin/time -q -f "$time_format" -o "$timing_path" \
            taskset -c "$cpu_set" env "${thread_environment[@]}" \
            foamRun -solver incompressibleFluid >"$log_path" 2>&1
    )
    local status=$?
    set -e
    [[ "$status" -eq 0 ]] || fail "OpenFOAM run failed for $case_name ($kind $ordinal)"
    [[ "$(numeric_output_count "$working_case")" -eq 0 ]] || fail "OpenFOAM wrote an unexpected time directory"
}

mapfile -t case_rows < <(jq -r '.cases[] | [.name, .fixedIterations] | @tsv' "$metadata_root/input-manifest.json")
total_runs=$((warmup_runs + measured_runs))
for case_row in "${case_rows[@]}"; do
    IFS=$'\t' read -r case_name fixed_iterations <<<"$case_row"
    for ((run_index=1; run_index<=total_runs; run_index++)); do
        if ((run_index <= warmup_runs)); then
            kind="warmup"; ordinal="$run_index"
        else
            kind="measured"; ordinal="$((run_index - warmup_runs))"
        fi
        if ((run_index % 2 == 1)); then engines=(ferrum openfoam); else engines=(openfoam ferrum); fi
        position=0
        for engine in "${engines[@]}"; do
            position=$((position + 1))
            printf '%s\t%s\t%s\t%s\t%s\n' "$case_name" "$kind" "$ordinal" "$position" "$engine" >>"$metadata_root/run-order.tsv"
            run_root="$results_root/$case_name/$kind-$ordinal-$engine"
            mkdir -p "$run_root"
            if [[ "$engine" == "ferrum" ]]; then
                run_ferrum "$case_name" "$kind" "$ordinal" "$fixed_iterations" "$run_root"
            else
                run_openfoam "$case_name" "$kind" "$ordinal" "$run_root"
            fi
        done
    done
done

archive_on_ext4="$workspace/linux-parity-results.tar"
tar -cf "$archive_on_ext4" -C "$workspace/export" .
archive_sha256="$(sha256sum "$archive_on_ext4" | awk '{print $1}')"
printf '%s\n' "$archive_sha256" >"$workspace/linux-parity-results.tar.sha256"
mkdir -p "$(dirname "$output_archive")"
cp -- "$archive_on_ext4" "$output_archive"
cp -- "$workspace/linux-parity-results.tar.sha256" "$output_archive.sha256"
completed="1"
printf 'output_archive=%s\n' "$output_archive"
printf 'output_archive_sha256=%s\n' "$archive_sha256"
printf 'workspace=%s\n' "$workspace"
