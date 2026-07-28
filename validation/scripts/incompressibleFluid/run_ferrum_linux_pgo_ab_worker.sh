#!/usr/bin/env bash
set -euo pipefail

mode="run"
rust_toolchain="1.94.0"
target_triple="x86_64-unknown-linux-gnu"
cpu_set="2"
warmup_runs="2"
measured_runs="20"
source_archive=""
source_archive_sha256=""
source_commit=""
source_tree=""
templates_archive=""
templates_archive_sha256=""
manifest_path=""
output_archive=""
controller_source=""
controller_sha256=""
worker_source=""
worker_sha256=""
keep_workspace="0"

fail() {
    printf 'Ferrum Linux Native-PGO A/B worker: %s\n' "$*" >&2
    exit 1
}

require_value() {
    [[ $# -ge 2 && -n "$2" ]] || fail "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preflight-only) mode="preflight"; shift ;;
        --rust-toolchain) require_value "$@"; rust_toolchain="$2"; shift 2 ;;
        --target-triple) require_value "$@"; target_triple="$2"; shift 2 ;;
        --cpu-set) require_value "$@"; cpu_set="$2"; shift 2 ;;
        --warmup-runs) require_value "$@"; warmup_runs="$2"; shift 2 ;;
        --measured-runs) require_value "$@"; measured_runs="$2"; shift 2 ;;
        --source-archive) require_value "$@"; source_archive="$2"; shift 2 ;;
        --source-archive-sha256) require_value "$@"; source_archive_sha256="$2"; shift 2 ;;
        --source-commit) require_value "$@"; source_commit="$2"; shift 2 ;;
        --source-tree) require_value "$@"; source_tree="$2"; shift 2 ;;
        --templates-archive) require_value "$@"; templates_archive="$2"; shift 2 ;;
        --templates-archive-sha256) require_value "$@"; templates_archive_sha256="$2"; shift 2 ;;
        --manifest) require_value "$@"; manifest_path="$2"; shift 2 ;;
        --output-archive) require_value "$@"; output_archive="$2"; shift 2 ;;
        --controller-source) require_value "$@"; controller_source="$2"; shift 2 ;;
        --controller-sha256) require_value "$@"; controller_sha256="$2"; shift 2 ;;
        --worker-source) require_value "$@"; worker_source="$2"; shift 2 ;;
        --worker-sha256) require_value "$@"; worker_sha256="$2"; shift 2 ;;
        --keep-workspace) keep_workspace="1"; shift ;;
        *) fail "unknown argument: $1" ;;
    esac
done

if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

for command_name in bash cargo rustc jq tar sha256sum taskset flock findmnt python3 readelf realpath stat cmp /usr/bin/time; do
    command -v "$command_name" >/dev/null 2>&1 || fail "required command was not found: $command_name"
done

[[ "$rust_toolchain" == "1.94.0" ]] || fail "this benchmark requires exact Rust 1.94.0"
[[ "$target_triple" == "x86_64-unknown-linux-gnu" ]] || fail "this benchmark requires target x86_64-unknown-linux-gnu"
[[ "$cpu_set" =~ ^[0-9]+([,-][0-9]+)*$ ]] || fail "invalid CPU set: $cpu_set"
taskset -c "$cpu_set" true >/dev/null 2>&1 || fail "CPU set is not available: $cpu_set"
[[ "$warmup_runs" =~ ^[0-9]+$ && "$measured_runs" =~ ^[1-9][0-9]*$ ]] || fail "run counts must be non-negative/positive integers"
((measured_runs % 2 == 0)) || fail "measured runs must be even"
if ! { [[ "$warmup_runs" == "0" && "$measured_runs" == "2" ]] || [[ "$warmup_runs" == "2" && "$measured_runs" == "20" ]]; }; then
    fail "only the 0+2 smoke or 2+20 decision protocol is allowed"
fi

rustc_vv="$(rustc "+$rust_toolchain" -vV 2>/dev/null || true)"
rustc_version="$(printf '%s\n' "$rustc_vv" | sed -n '1p')"
[[ "$rustc_version" == "rustc $rust_toolchain "* ]] || fail "exact Rust $rust_toolchain is not installed"
host_triple="$(printf '%s\n' "$rustc_vv" | sed -n 's/^host: //p')"
[[ "$host_triple" == "$target_triple" ]] || fail "Rust host '$host_triple' differs from required target '$target_triple'"
rustc "+$rust_toolchain" --print target-list | grep -Fx "$target_triple" >/dev/null || fail "required Rust target is unavailable"
cargo_version="$(cargo "+$rust_toolchain" --version 2>/dev/null || true)"
[[ -n "$cargo_version" ]] || fail "Cargo for Rust $rust_toolchain is not installed"
rust_sysroot="$(rustc "+$rust_toolchain" --print sysroot)"
[[ "$rust_sysroot" == /* ]] || fail "Rust sysroot is not absolute"
compgen -G "$rust_sysroot/lib/rustlib/$target_triple/lib/libstd-*.rlib" >/dev/null ||
    fail "standard library for target '$target_triple' is not installed in the exact toolchain"
llvm_profdata="$rust_sysroot/lib/rustlib/$host_triple/bin/llvm-profdata"
[[ -x "$llvm_profdata" ]] || fail "toolchain-bound llvm-profdata was not found: $llvm_profdata"
[[ "$(realpath -- "$llvm_profdata")" == "$llvm_profdata" ]] || fail "llvm-profdata path is not canonical"
llvm_profdata_version="$($llvm_profdata --version 2>/dev/null || true)"
[[ -n "$llvm_profdata_version" ]] || fail "toolchain-bound llvm-profdata did not report a version"
rustc_llvm_version="$(printf '%s\n' "$rustc_vv" | sed -n 's/^LLVM version: //p')"
llvm_profdata_numeric_version="$(printf '%s\n' "$llvm_profdata_version" | sed -n 's/.*LLVM version \([0-9][0-9.]*\).*/\1/p' | head -n 1)"
[[ -n "$rustc_llvm_version" && "$llvm_profdata_numeric_version" == "$rustc_llvm_version" ]] ||
    fail "llvm-profdata LLVM '$llvm_profdata_numeric_version' differs from rustc LLVM '$rustc_llvm_version'"
llvm_profdata_sha256="$(sha256sum "$llvm_profdata" | awk '{print $1}')"
home_fstype="$(findmnt -T "$HOME" -n -o FSTYPE | tr -d '[:space:]')"
[[ "$home_fstype" == "ext4" ]] || fail "WSL home is not on ext4 (found '$home_fstype')"

if [[ "$mode" == "preflight" ]]; then
    printf 'preflight=pass\n'
    printf 'rustc=%s\n' "$rustc_version"
    printf 'cargo=%s\n' "$cargo_version"
    printf 'target=%s\n' "$target_triple"
    printf 'filesystem=%s\n' "$home_fstype"
    printf 'cpu_set=%s\n' "$cpu_set"
    printf 'llvm_profdata_path=%s\n' "$llvm_profdata"
    printf 'llvm_profdata_sha256=%s\n' "$llvm_profdata_sha256"
    printf 'decision_eligible=%s\n' "$([[ "$warmup_runs" == "2" && "$measured_runs" == "20" ]] && printf true || printf false)"
    exit 0
fi

for required_file in "$source_archive" "$templates_archive" "$manifest_path" "$controller_source" "$worker_source"; do
    [[ -f "$required_file" ]] || fail "required staged input was not found: $required_file"
done
for required_value in "$source_archive_sha256" "$source_commit" "$source_tree" "$templates_archive_sha256" "$output_archive" "$controller_sha256" "$worker_sha256"; do
    [[ -n "$required_value" ]] || fail "a required binding value was empty"
done
[[ "$source_archive" == /* && "$templates_archive" == /* && "$manifest_path" == /* && "$output_archive" == /* &&
    "$controller_source" == /* && "$worker_source" == /* ]] || fail "all staged paths must be absolute"
[[ "$(sha256sum "$controller_source" | awk '{print $1}')" == "$controller_sha256" ]] || fail "controller source SHA-256 differs"
[[ "$(sha256sum "$worker_source" | awk '{print $1}')" == "$worker_sha256" ]] || fail "worker source SHA-256 differs"

cache_root="$HOME/.cache/ferrumcfd-linux-native-pgo-ab"
mkdir -p "$cache_root"
exec 9>"$cache_root/benchmark.lock"
flock -n 9 || fail "another Ferrum Linux Native-PGO A/B benchmark is active"
workspace="$(mktemp -d "$cache_root/run.XXXXXXXX")"
completed="0"

cleanup() {
    local status=$?
    if [[ "$status" -eq 0 && "$completed" == "1" && "$keep_workspace" == "0" ]]; then
        case "$workspace" in
            "$cache_root"/run.*) rm -rf -- "$workspace" ;;
            *) printf 'refusing unsafe workspace cleanup: %s\n' "$workspace" >&2 ;;
        esac
    else
        printf 'Ferrum Linux Native-PGO A/B workspace preserved: %s\n' "$workspace" >&2
    fi
}
trap cleanup EXIT

validate_archive() {
    local archive_path="$1" description="$2"
    python3 - "$archive_path" "$description" <<'PY'
import pathlib, re, sys, tarfile
archive_path, description = sys.argv[1:]
with tarfile.open(archive_path, mode="r:*") as archive:
    members = archive.getmembers()
    if not members:
        raise SystemExit(f"{description} archive is empty")
    for member in members:
        name = member.name.replace("\\", "/")
        path = pathlib.PurePosixPath(name)
        if not name or name.startswith("/") or re.match(r"^[A-Za-z]:(?:/|$)", name) or ".." in path.parts:
            raise SystemExit(f"{description} archive contains an unsafe path: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"{description} archive contains a non-regular entry: {member.name}")
PY
}

export_root="$workspace/export"
raw_root="$export_root/raw"
metadata_root="$export_root/metadata"
binary_export_root="$export_root/binaries"
profile_export_root="$export_root/profiles"
control_export_root="$export_root/controls"
source_root="$workspace/source"
templates_root="$workspace/templates"
mkdir -p "$raw_root" "$metadata_root" "$binary_export_root" "$profile_export_root" "$control_export_root" "$source_root" "$templates_root"
cp -- "$manifest_path" "$metadata_root/input-manifest.json"
cp -- "$controller_source" "$control_export_root/run_ferrum_linux_pgo_ab_benchmark.ps1"
cp -- "$worker_source" "$control_export_root/run_ferrum_linux_pgo_ab_worker.sh"
chmod 0444 "$control_export_root"/*

cp -- "$source_archive" "$workspace/source.tar"
actual_source_archive_sha256="$(sha256sum "$workspace/source.tar" | awk '{print $1}')"
[[ "$actual_source_archive_sha256" == "$source_archive_sha256" ]] || fail "source archive SHA-256 changed while staging"
validate_archive "$workspace/source.tar" "exact source"
tar --no-same-owner --no-same-permissions -xf "$workspace/source.tar" -C "$source_root"
cp -- "$templates_archive" "$workspace/templates.tar"
actual_templates_sha256="$(sha256sum "$workspace/templates.tar" | awk '{print $1}')"
[[ "$actual_templates_sha256" == "$templates_archive_sha256" ]] || fail "templates archive SHA-256 changed while staging"
validate_archive "$workspace/templates.tar" "matched Ferrum templates"
tar --no-same-owner --no-same-permissions -xf "$workspace/templates.tar" -C "$templates_root"

for root in "$source_root" "$templates_root" "$raw_root"; do
    [[ "$(findmnt -T "$root" -n -o FSTYPE | tr -d '[:space:]')" == "ext4" ]] || fail "benchmark path is not on ext4: $root"
done

[[ "$(jq -r '.source.commit' "$metadata_root/input-manifest.json")" == "$source_commit" ]] || fail "manifest source commit differs"
[[ "$(jq -r '.source.tree' "$metadata_root/input-manifest.json")" == "$source_tree" ]] || fail "manifest source tree differs"
[[ "$(jq -r '.source.archiveSha256' "$metadata_root/input-manifest.json")" == "$source_archive_sha256" ]] || fail "manifest source archive SHA differs"
[[ "$(jq -r '.rust.toolchain' "$metadata_root/input-manifest.json")" == "$rust_toolchain" ]] || fail "manifest Rust toolchain differs"
[[ "$(jq -r '.rust.target' "$metadata_root/input-manifest.json")" == "$target_triple" ]] || fail "manifest target differs"
[[ "$(jq -r '.pressureSolver' "$metadata_root/input-manifest.json")" == "gamg" ]] || fail "manifest pressure solver is not GAMG"
[[ "$(jq -r '.trainingOrder | map(.name + ":" + (.fixedIterations|tostring)) | join(",")' "$metadata_root/input-manifest.json")" == "laminarPipe:10,planeChannel:500" ]] || fail "manifest training order differs"
[[ "$(jq -r '.controls.controllerSha256' "$metadata_root/input-manifest.json")" == "$controller_sha256" ]] || fail "manifest controller SHA differs"
[[ "$(jq -r '.controls.workerSha256' "$metadata_root/input-manifest.json")" == "$worker_sha256" ]] || fail "manifest worker SHA differs"

cargo_lock_sha256="$(sha256sum "$source_root/Cargo.lock" | awk '{print $1}')"
[[ "$cargo_lock_sha256" == "$(jq -r '.cargoLock.sha256' "$metadata_root/input-manifest.json")" ]] || fail "Cargo.lock content differs from manifest"

for case_name in $(jq -r '.cases[].name' "$metadata_root/input-manifest.json"); do
    case_root="$templates_root/$case_name"
    [[ -d "$case_root" ]] || fail "staged case template was not found: $case_name"
    for mesh_file in points faces owner neighbour boundary; do
        expected_hash="$(jq -r --arg case "$case_name" --arg file "$mesh_file" '.cases[] | select(.name == $case) | .canonicalPolyMeshSha256[$file]' "$metadata_root/input-manifest.json")"
        actual_hash="$(sha256sum "$case_root/constant/polyMesh/$mesh_file" | awk '{print $1}')"
        [[ "$actual_hash" == "$expected_hash" ]] || fail "$case_name polyMesh differs in $mesh_file"
    done
    for shared_name in velocity pressure fvSchemes fvSolution; do
        case "$shared_name" in
            velocity) relative_path="0/U" ;;
            pressure) relative_path="0/p" ;;
            fvSchemes) relative_path="system/fvSchemes" ;;
            fvSolution) relative_path="system/fvSolution" ;;
        esac
        expected_hash="$(jq -r --arg case "$case_name" --arg file "$shared_name" '.cases[] | select(.name == $case) | .sharedFileSha256[$file]' "$metadata_root/input-manifest.json")"
        actual_hash="$(sha256sum "$case_root/$relative_path" | awk '{print $1}')"
        [[ "$actual_hash" == "$expected_hash" ]] || fail "$case_name shared file differs: $relative_path"
    done
done

base_rustflags="-C target-cpu=native"
profile_raw_root="$workspace/profiles/raw"
profile_merged="$workspace/profiles/merged.profdata"
mkdir -p "$workspace/targets/native" "$workspace/targets/instrumented" "$workspace/targets/pgo" "$workspace/binaries" "$workspace/profiles"
for root in "$workspace/targets/native" "$workspace/targets/instrumented" "$workspace/targets/pgo" "$workspace/binaries" "$workspace/profiles"; do
    [[ "$(findmnt -T "$root" -n -o FSTYPE | tr -d '[:space:]')" == "ext4" ]] || fail "build/profile path is not on ext4: $root"
done
build_format=$'elapsed_s=%e\nuser_s=%U\nsystem_s=%S\nmax_rss_kb=%M\nexit=%x'

build_binary() {
    local name="$1" rustflags="$2" target_root="$3"
    local build_log="$metadata_root/cargo-build-$name-release.log" build_time="$metadata_root/build-$name-time.env"
    set +e
    (
        cd "$source_root"
        /usr/bin/time -q -f "$build_format" -o "$build_time" \
            env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS -u CARGO_BUILD_RUSTFLAGS \
            -u CARGO_INCREMENTAL -u CARGO_PROFILE_RELEASE_INCREMENTAL \
            -u CARGO_PROFILE_RELEASE_CODEGEN_UNITS -u CARGO_PROFILE_RELEASE_LTO \
            -u CARGO_PROFILE_RELEASE_OPT_LEVEL -u CARGO_PROFILE_RELEASE_DEBUG \
            -u CARGO_PROFILE_RELEASE_PANIC -u CARGO_PROFILE_RELEASE_STRIP \
            -u RUSTC_WRAPPER -u RUSTC_WORKSPACE_WRAPPER \
            CARGO_INCREMENTAL=0 CARGO_TARGET_DIR="$target_root" RUSTFLAGS="$rustflags" \
            cargo "+$rust_toolchain" build --locked --release --target "$target_triple" \
                -p ferrum-run --bin ferrumRun >"$build_log" 2>&1
    )
    local status=$?
    set -e
    [[ "$status" -eq 0 ]] || fail "$name release build failed; see $build_log"
    local built="$target_root/$target_triple/release/ferrumRun"
    [[ -x "$built" ]] || fail "$name Linux executable was not produced"
    cp -- "$built" "$workspace/binaries/ferrumRun-$name"
    chmod 0555 "$workspace/binaries/ferrumRun-$name"
}

build_binary native "$base_rustflags" "$workspace/targets/native"
native_binary="$workspace/binaries/ferrumRun-native"
readelf -SW "$native_binary" >"$metadata_root/native-readelf-sections.txt"
if grep -Eq '(__llvm_prf_(cnts|data|names)|\.llvm_prf_(cnts|data|names))' "$metadata_root/native-readelf-sections.txt"; then
    fail "native baseline binary unexpectedly contains LLVM profile-generation sections"
fi

profile_generate_flag="-C profile-generate=$profile_raw_root"
build_binary instrumented "$base_rustflags $profile_generate_flag" "$workspace/targets/instrumented"
instrumented_binary="$workspace/binaries/ferrumRun-instrumented"
readelf -SW "$instrumented_binary" >"$metadata_root/instrumented-readelf-sections.txt"
for profile_section in cnts data names; do
    grep -Eq "(__llvm_prf_${profile_section}|\\.llvm_prf_${profile_section})" "$metadata_root/instrumented-readelf-sections.txt" ||
        fail "instrumented binary is missing LLVM profile section: $profile_section"
done

# The fresh raw directory is created exactly once, immediately before the
# fixed Pipe -> Channel training sequence. A reused directory is rejected.
case "$profile_raw_root" in
    "$workspace"/profiles/raw) ;;
    *) fail "unsafe raw profile path: $profile_raw_root" ;;
esac
[[ ! -e "$profile_raw_root" ]] || fail "raw profile directory existed before the fixed training sequence"
mkdir -p "$profile_raw_root"
[[ -z "$(find "$profile_raw_root" -mindepth 1 -maxdepth 1 -print -quit)" ]] || fail "raw profile directory is not empty before training"

thread_environment=(LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1)
printf 'ordinal\tcase\tfixedIterations\tprofrawCountAfter\n' >"$metadata_root/training-order.tsv"

numeric_output_count() {
    find "$1" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
        awk '$0 != "0" && $0 ~ /^[0-9]+([.][0-9]+)?$/ {count++} END {print count+0}'
}

validate_fixed_report() {
    local report_path="$1" fixed_iterations="$2" description="$3"
    jq -e --argjson iterations "$fixed_iterations" '
        .solve.simpleIterations == $iterations and
        .linearSolves.momentumComponentNonConvergedSolves == 0 and
        .linearSolves.pressureCorrectionNonConvergedSolves == 0 and
        (.history | length) == $iterations
    ' "$report_path" >/dev/null || fail "$description report violates the fixed-work contract"
}

train_case() {
    local ordinal="$1" case_name="$2" fixed_iterations="$3"
    local run_root="$workspace/training/$ordinal-$case_name" working_case="$workspace/training/$ordinal-$case_name/case"
    mkdir -p "$working_case"
    cp -a "$templates_root/$case_name/." "$working_case/"
    (
        cd "$run_root"
        taskset -c "$cpu_set" env "${thread_environment[@]}" \
            LLVM_PROFILE_FILE="$profile_raw_root/%m-%p.profraw" \
            "$instrumented_binary" -solver incompressibleFluid -case "$working_case" \
            --minSimpleIterations "$fixed_iterations" --maxSimpleIterations "$fixed_iterations" \
            --solveReportJson "$run_root/solve-report.json" >"$run_root/ferrum.log" 2>&1
    ) || fail "instrumented training failed for $case_name"
    [[ "$(numeric_output_count "$working_case")" -eq 0 ]] || fail "training wrote an unexpected time directory for $case_name"
    validate_fixed_report "$run_root/solve-report.json" "$fixed_iterations" "$case_name training"
    local count
    count="$(find "$profile_raw_root" -maxdepth 1 -type f -name '*.profraw' -printf '.\n' | wc -l | tr -d '[:space:]')"
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || fail "$case_name training produced no raw profile"
    printf '%s\t%s\t%s\t%s\n' "$ordinal" "$case_name" "$fixed_iterations" "$count" >>"$metadata_root/training-order.tsv"
    printf '%s\n' "$count"
}

pipe_raw_count="$(train_case 1 laminarPipe 10)"
channel_raw_count="$(train_case 2 planeChannel 500)"
((channel_raw_count > pipe_raw_count)) || fail "Channel training did not add a distinct raw profile"
mapfile -t raw_profiles < <(find "$profile_raw_root" -maxdepth 1 -type f -name '*.profraw' -print | LC_ALL=C sort)
[[ "${#raw_profiles[@]}" -eq "$channel_raw_count" ]] || fail "raw profile inventory changed before merge"
printf 'name\tsizeBytes\tsha256\n' >"$metadata_root/llvm-profraw-inventory.tsv"
for raw_profile in "${raw_profiles[@]}"; do
    printf '%s\t%s\t%s\n' "$(basename "$raw_profile")" "$(stat -c %s "$raw_profile")" \
        "$(sha256sum "$raw_profile" | awk '{print $1}')" >>"$metadata_root/llvm-profraw-inventory.tsv"
done
"$llvm_profdata" merge -sparse "${raw_profiles[@]}" -o "$profile_merged"
[[ -s "$profile_merged" ]] || fail "merged LLVM profile was not produced"
profile_merged_sha256="$(sha256sum "$profile_merged" | awk '{print $1}')"
"$llvm_profdata" show --all-functions --counts "$profile_merged" >"$metadata_root/llvm-profdata-show.txt"
[[ -s "$metadata_root/llvm-profdata-show.txt" ]] || fail "llvm-profdata show returned no proof"
grep -Eq 'Functions shown:[[:space:]]*[1-9][0-9]*' "$metadata_root/llvm-profdata-show.txt" ||
    fail "merged LLVM profile contains no demonstrated functions"

profile_use_flag="-C profile-use=$profile_merged"
missing_function_flag="-C llvm-args=-pgo-warn-missing-function"
build_binary pgo "$base_rustflags $profile_use_flag $missing_function_flag" "$workspace/targets/pgo"
pgo_binary="$workspace/binaries/ferrumRun-pgo"
readelf -SW "$pgo_binary" >"$metadata_root/pgo-readelf-sections.txt"
if grep -Eq '(__llvm_prf_(cnts|data|names)|\.llvm_prf_(cnts|data|names))' "$metadata_root/pgo-readelf-sections.txt"; then
    fail "final PGO binary still contains LLVM profile-generation sections"
fi

cp -- "$native_binary" "$binary_export_root/ferrumRun-native"
cp -- "$instrumented_binary" "$binary_export_root/ferrumRun-instrumented"
cp -- "$pgo_binary" "$binary_export_root/ferrumRun-pgo"
chmod 0555 "$binary_export_root"/ferrumRun-*
cp -- "$profile_merged" "$profile_export_root/merged.profdata"
chmod 0444 "$profile_export_root/merged.profdata"
mkdir -p "$profile_export_root/raw"
for raw_profile in "${raw_profiles[@]}"; do
    cp -- "$raw_profile" "$profile_export_root/raw/$(basename "$raw_profile")"
done
chmod 0444 "$profile_export_root/raw"/*.profraw

printf '%s\n' "$source_commit" >"$metadata_root/source-commit.txt"
printf '%s\n' "$source_tree" >"$metadata_root/source-tree.txt"
printf '%s\n' "$actual_source_archive_sha256" >"$metadata_root/source-archive-sha256.txt"
printf '%s\n' "$actual_templates_sha256" >"$metadata_root/templates-archive-sha256.txt"
printf '%s\n' "$cargo_lock_sha256" >"$metadata_root/cargo-lock-sha256.txt"
printf '%s\n' "$controller_sha256" >"$metadata_root/controller-script-sha256.txt"
printf '%s\n' "$worker_sha256" >"$metadata_root/worker-script-sha256.txt"
printf '%s\n' "$base_rustflags" >"$metadata_root/native-rustflags.txt"
printf '%s\n' "$base_rustflags $profile_generate_flag" >"$metadata_root/instrumented-rustflags.txt"
printf '%s\n' "$base_rustflags $profile_use_flag $missing_function_flag" >"$metadata_root/pgo-rustflags.txt"
printf '%s\n' "$rustc_vv" >"$metadata_root/rustc-vv.txt"
printf '%s\n' "$cargo_version" >"$metadata_root/cargo-version.txt"
printf '%s\n' "$target_triple" >"$metadata_root/target-triple.txt"
printf '%s\n' "$llvm_profdata" >"$metadata_root/llvm-profdata-path.txt"
printf '%s\n' "$llvm_profdata_version" >"$metadata_root/llvm-profdata-version.txt"
printf '%s\n' "$llvm_profdata_sha256" >"$metadata_root/llvm-profdata-sha256.txt"
printf '%s\n' "$channel_raw_count" >"$metadata_root/llvm-profraw-count.txt"
printf '%s\n' "$profile_merged_sha256" >"$metadata_root/llvm-profdata-merged-sha256.txt"
sha256sum "$native_binary" | awk '{print $1}' >"$metadata_root/native-binary-sha256.txt"
sha256sum "$instrumented_binary" | awk '{print $1}' >"$metadata_root/instrumented-binary-sha256.txt"
sha256sum "$pgo_binary" | awk '{print $1}' >"$metadata_root/pgo-binary-sha256.txt"
uname -a >"$metadata_root/uname.txt"
grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '"' >"$metadata_root/distro-release.txt"
awk -F: '/model name/ {sub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo >"$metadata_root/cpu-model.txt"
printf '%s\n' "$cpu_set" >"$metadata_root/cpu-set.txt"
first_cpu="${cpu_set%%[,-]*}"
cat "/sys/devices/system/cpu/cpu$first_cpu/topology/thread_siblings_list" >"$metadata_root/cpu-siblings.txt"
printf '%s\n' "$home_fstype" >"$metadata_root/filesystem-type.txt"
printf '%s\n' "$workspace" >"$metadata_root/workspace-path.txt"
printf 'case\tkind\tordinal\tposition\tbuild\n' >"$metadata_root/run-order.tsv"

canonicalize_report() {
    local report_path="$1" canonical_path="$2" hash_path="$3"
    python3 - "$report_path" "$canonical_path" "$hash_path" <<'PY'
import hashlib, json, pathlib, sys
report_path, canonical_path, hash_path = map(pathlib.Path, sys.argv[1:])
with report_path.open("r", encoding="utf-8-sig") as handle:
    report = json.load(handle)
removed = []
def canonical(value, path=()):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            child_path = path + (key,)
            if key == "caseDir" or key.endswith("Seconds"):
                removed.append(".".join(child_path)); continue
            result[key] = canonical(child, child_path)
        return result
    if isinstance(value, list):
        return [canonical(child, path + (str(index),)) for index, child in enumerate(value)]
    return value
result = canonical(report)
if "caseDir" not in report or "solve.wallClockSeconds" not in removed or not any(item.startswith("timing.") and item.endswith("Seconds") for item in removed):
    raise SystemExit("report did not expose the expected path/timing fields")
if "pressureMatrixVectorProducts" not in result.get("timing", {}) or "pressurePreconditionerApplications" not in result.get("timing", {}):
    raise SystemExit("canonical report dropped deterministic timing counters")
payload = (json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
canonical_path.write_bytes(payload)
hash_path.write_text(hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii")
PY
}

write_field_oracle() {
    local report_path="$1" fields_root="$2" output_path="$3"
    python3 - "$report_path" "$fields_root" "$output_path" <<'PY'
import hashlib, json, math, pathlib, re, struct, sys
report_path, fields_root, output_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
report = json.loads(report_path.read_text(encoding="utf-8-sig"))
cells = report.get("mesh", {}).get("cells")
if not isinstance(cells, int) or cells <= 0:
    raise SystemExit("oracle report has no positive integer cell count")
def read_field(name, kind, components):
    path = fields_root / name
    raw = path.read_bytes(); text = raw.decode("utf-8")
    if len(re.findall(rf"(?m)^\s*class\s+vol{'Vector' if components == 3 else 'Scalar'}Field\s*;", text)) != 1:
        raise SystemExit(f"{name} field class is not exact")
    if len(re.findall(rf"(?m)^\s*object\s+{name}\s*;", text)) != 1:
        raise SystemExit(f"{name} field object is not exact")
    matches = list(re.finditer(rf"\binternalField\s+nonuniform\s+List<{kind}>\s+(\d+)\s*\((.*?)\)\s*;", text, re.S))
    if len(matches) != 1 or int(matches[0].group(1)) != cells:
        raise SystemExit(f"{name} internalField shape differs")
    payload = matches[0].group(2)
    if components == 1:
        values = [float(token) for token in payload.split()]
    else:
        vector_pattern = re.compile(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)")
        vectors = list(vector_pattern.finditer(payload))
        if vector_pattern.sub("", payload).strip() or len(vectors) != cells:
            raise SystemExit(f"{name} vector payload is malformed")
        values = [float(token) for vector in vectors for token in vector.groups()]
    if len(values) != cells * components or not all(math.isfinite(value) for value in values):
        raise SystemExit(f"{name} parsed value contract failed")
    bits = b"".join(struct.pack(">d", value) for value in values)
    roundtrip = [struct.unpack(">d", bits[offset:offset + 8])[0] for offset in range(0, len(bits), 8)]
    if any(struct.pack(">d", left) != struct.pack(">d", right) for left, right in zip(values, roundtrip)):
        raise SystemExit(f"{name} IEEE-754 round-trip failed")

    number = r"[-+0-9.eE]+"
    if components == 1:
        boundary_pattern = re.compile(
            rf"\bvalue\s+(?:uniform\s+({number})\s*;|nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;)", re.S)
    else:
        boundary_pattern = re.compile(
            rf"\bvalue\s+(?:uniform\s+\(\s*({number})\s+({number})\s+({number})\s*\)\s*;|"
            rf"nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;)", re.S)
    boundary_matches = list(boundary_pattern.finditer(text))
    if len(re.findall(r"\bvalue\b", text)) != len(boundary_matches):
        raise SystemExit(f"{name} contains an unsupported boundary value entry")
    boundary_parts = []
    boundary_scalar_slots = 0
    for entry in boundary_matches:
        groups = entry.groups()
        if components == 1:
            if groups[0] is not None:
                entry_values = [float(groups[0])]; declared = 1; marker = b"uniform\0"
            else:
                declared = int(groups[1]); tokens = groups[2].split()
                if len(tokens) != declared: raise SystemExit(f"{name} boundary scalar payload count differs")
                entry_values = [float(token) for token in tokens]; marker = b"nonuniform\0"
        else:
            if groups[0] is not None:
                entry_values = [float(groups[0]), float(groups[1]), float(groups[2])]; declared = 1; marker = b"uniform\0"
            else:
                declared = int(groups[3])
                vectors = list(vector_pattern.finditer(groups[4]))
                if vector_pattern.sub("", groups[4]).strip() or len(vectors) != declared:
                    raise SystemExit(f"{name} boundary vector payload is malformed")
                entry_values = [float(token) for vector in vectors for token in vector.groups()]; marker = b"nonuniform\0"
        if len(entry_values) != declared * components or not all(math.isfinite(value) for value in entry_values):
            raise SystemExit(f"{name} boundary value contract failed")
        entry_bits = b"".join(struct.pack(">d", value) for value in entry_values)
        entry_roundtrip = [struct.unpack(">d", entry_bits[offset:offset + 8])[0]
                           for offset in range(0, len(entry_bits), 8)]
        if any(struct.pack(">d", left) != struct.pack(">d", right)
               for left, right in zip(entry_values, entry_roundtrip)):
            raise SystemExit(f"{name} boundary IEEE-754 round-trip failed")
        boundary_parts.append(marker + struct.pack(">Q", declared) + entry_bits)
        boundary_scalar_slots += len(entry_values)
    boundary_bits = b"".join(boundary_parts)
    full_bits = b"FerrumCFD-field-v1\0" + name.encode("ascii") + b"\0" + bits + boundary_bits
    return {
        "name": name, "declaredValues": cells, "components": components,
        "scalarSlots": len(values), "boundaryValueEntries": len(boundary_matches),
        "boundaryScalarSlots": boundary_scalar_slots,
        "textSha256": hashlib.sha256(raw).hexdigest(),
        "ieee754BigEndianSha256": hashlib.sha256(bits).hexdigest(),
        "boundaryIeee754BigEndianSha256": hashlib.sha256(boundary_bits).hexdigest(),
        "fullFieldIeee754BigEndianSha256": hashlib.sha256(full_bits).hexdigest(),
    }, full_bits
u_summary, u_bits = read_field("U", "vector", 3)
p_summary, p_bits = read_field("p", "scalar", 1)
combined = b"FerrumCFD-field-oracle-v2\0" + struct.pack(">Q", cells) + u_bits + p_bits
result = {"schemaVersion": 2, "cellCount": cells, "encoding": "IEEE-754 binary64 big-endian",
          "U": u_summary, "p": p_summary, "combinedIeee754Sha256": hashlib.sha256(combined).hexdigest()}
output_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
}

time_format=$'elapsed_s=%e\nuser_s=%U\nsystem_s=%S\nmax_rss_kb=%M\nexit=%x'

run_timed_build() {
    local build_name="$1" binary="$2" case_name="$3" kind="$4" ordinal="$5" fixed_iterations="$6" run_root="$7"
    local working_case="$run_root/case"
    mkdir -p "$working_case"
    cp -a "$templates_root/$case_name/." "$working_case/"
    set +e
    (
        cd "$run_root"
        /usr/bin/time -q -f "$time_format" -o "$run_root/process-time.env" \
            taskset -c "$cpu_set" env "${thread_environment[@]}" \
            "$binary" -solver incompressibleFluid -case "$working_case" \
            --minSimpleIterations "$fixed_iterations" --maxSimpleIterations "$fixed_iterations" \
            --solveReportJson "$run_root/solve-report.json" >"$run_root/ferrum.log" 2>&1
    )
    local status=$?
    set -e
    [[ "$status" -eq 0 ]] || fail "$build_name run failed for $case_name ($kind $ordinal)"
    [[ "$(numeric_output_count "$working_case")" -eq 0 ]] || fail "$build_name wrote an unexpected time directory"
    validate_fixed_report "$run_root/solve-report.json" "$fixed_iterations" "$build_name $case_name $kind $ordinal"
    canonicalize_report "$run_root/solve-report.json" "$run_root/canonical-report.json" "$run_root/canonical-report.sha256"
}

run_oracle_build() {
    local build_name="$1" binary="$2" case_name="$3" fixed_iterations="$4" run_root="$5"
    local working_case="$run_root/case" fields_root="$run_root/final-fields"
    mkdir -p "$working_case"
    cp -a "$templates_root/$case_name/." "$working_case/"
    (
        cd "$run_root"
        taskset -c "$cpu_set" env "${thread_environment[@]}" \
            "$binary" -solver incompressibleFluid -case "$working_case" \
            --minSimpleIterations "$fixed_iterations" --maxSimpleIterations "$fixed_iterations" \
            --solveReportJson "$run_root/solve-report.json" --writeFinalFields final-fields >"$run_root/ferrum.log" 2>&1
    ) || fail "$build_name oracle failed for $case_name"
    [[ "$(numeric_output_count "$working_case")" -eq 0 ]] || fail "$build_name oracle wrote an unexpected time directory"
    [[ -f "$fields_root/U" && -f "$fields_root/p" ]] || fail "$build_name oracle did not write U and p"
    [[ "$(find "$fields_root" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | sort | tr '\n' ' ')" == "U p " ]] || fail "$build_name oracle inventory was not exactly U and p"
    validate_fixed_report "$run_root/solve-report.json" "$fixed_iterations" "$build_name $case_name oracle"
    canonicalize_report "$run_root/solve-report.json" "$run_root/canonical-report.json" "$run_root/canonical-report.sha256"
    write_field_oracle "$run_root/solve-report.json" "$fields_root" "$run_root/field-oracle.json"
}

mapfile -t case_rows < <(jq -r '.cases[] | [.name, .fixedIterations] | @tsv' "$metadata_root/input-manifest.json")
for case_row in "${case_rows[@]}"; do
    IFS=$'\t' read -r case_name fixed_iterations <<<"$case_row"
    for kind in warmup measured; do
        [[ "$kind" == "warmup" ]] && count="$warmup_runs" || count="$measured_runs"
        for ((ordinal=1; ordinal<=count; ordinal++)); do
            if ((ordinal % 2 == 1)); then builds=(native pgo); else builds=(pgo native); fi
            position=0
            for build_name in "${builds[@]}"; do
                position=$((position + 1))
                printf '%s\t%s\t%s\t%s\t%s\n' "$case_name" "$kind" "$ordinal" "$position" "$build_name" >>"$metadata_root/run-order.tsv"
                run_root="$raw_root/$case_name/$kind-$ordinal-$build_name"
                mkdir -p "$run_root"
                [[ "$build_name" == "native" ]] && binary="$native_binary" || binary="$pgo_binary"
                run_timed_build "$build_name" "$binary" "$case_name" "$kind" "$ordinal" "$fixed_iterations" "$run_root"
            done
        done
    done
    mkdir -p "$raw_root/$case_name/oracle-native" "$raw_root/$case_name/oracle-pgo"
    run_oracle_build native "$native_binary" "$case_name" "$fixed_iterations" "$raw_root/$case_name/oracle-native"
    run_oracle_build pgo "$pgo_binary" "$case_name" "$fixed_iterations" "$raw_root/$case_name/oracle-pgo"
done

[[ "$(sha256sum "$native_binary" | awk '{print $1}')" == "$(cat "$metadata_root/native-binary-sha256.txt")" ]] || fail "native binary changed during execution"
[[ "$(sha256sum "$pgo_binary" | awk '{print $1}')" == "$(cat "$metadata_root/pgo-binary-sha256.txt")" ]] || fail "PGO binary changed during execution"
printf 'name\tsizeBytes\tsha256\n' >"$workspace/llvm-profraw-inventory-after.tsv"
mapfile -t raw_profiles_after < <(find "$profile_raw_root" -maxdepth 1 -type f -name '*.profraw' -print | LC_ALL=C sort)
[[ "${#raw_profiles_after[@]}" -eq "${#raw_profiles[@]}" ]] || fail "raw profile count changed after training"
for raw_profile in "${raw_profiles_after[@]}"; do
    [[ -f "$raw_profile" ]] || fail "raw profile disappeared during timing"
    printf '%s\t%s\t%s\n' "$(basename "$raw_profile")" "$(stat -c %s "$raw_profile")" \
        "$(sha256sum "$raw_profile" | awk '{print $1}')" >>"$workspace/llvm-profraw-inventory-after.tsv"
done
cmp -s "$metadata_root/llvm-profraw-inventory.tsv" "$workspace/llvm-profraw-inventory-after.tsv" ||
    fail "raw profile inventory changed after training"
[[ "$(sha256sum "$profile_export_root/merged.profdata" | awk '{print $1}')" == "$profile_merged_sha256" ]] ||
    fail "exported merged profile differs from its bound SHA-256"
for exported_binary in native instrumented pgo; do
    [[ "$(sha256sum "$binary_export_root/ferrumRun-$exported_binary" | awk '{print $1}')" == \
        "$(cat "$metadata_root/$exported_binary-binary-sha256.txt")" ]] || fail "exported $exported_binary binary differs"
done
[[ "$(sha256sum "$control_export_root/run_ferrum_linux_pgo_ab_benchmark.ps1" | awk '{print $1}')" == "$controller_sha256" ]] ||
    fail "exported controller source differs"
[[ "$(sha256sum "$control_export_root/run_ferrum_linux_pgo_ab_worker.sh" | awk '{print $1}')" == "$worker_sha256" ]] ||
    fail "exported worker source differs"
for raw_profile in "${raw_profiles_after[@]}"; do
    exported_raw="$profile_export_root/raw/$(basename "$raw_profile")"
    [[ "$(sha256sum "$exported_raw" | awk '{print $1}')" == "$(sha256sum "$raw_profile" | awk '{print $1}')" ]] ||
        fail "exported raw profile differs: $(basename "$raw_profile")"
done

archive_on_ext4="$workspace/ferrum-linux-native-pgo-ab-results.tar"
tar -cf "$archive_on_ext4" -C "$export_root" .
archive_sha256="$(sha256sum "$archive_on_ext4" | awk '{print $1}')"
printf '%s\n' "$archive_sha256" >"$workspace/ferrum-linux-native-pgo-ab-results.tar.sha256"
mkdir -p "$(dirname "$output_archive")"
cp -- "$archive_on_ext4" "$output_archive"
cp -- "$workspace/ferrum-linux-native-pgo-ab-results.tar.sha256" "$output_archive.sha256"
completed="1"
printf 'output_archive=%s\noutput_archive_sha256=%s\nworkspace=%s\n' "$output_archive" "$archive_sha256" "$workspace"
