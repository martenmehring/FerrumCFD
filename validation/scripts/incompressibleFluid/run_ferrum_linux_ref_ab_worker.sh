#!/usr/bin/env bash
set -euo pipefail

mode="run"
rust_toolchain="1.94.0"
cpu_set="2"
build_variant="portable"
warmup_runs="2"
measured_runs="10"
pressure_solver="gamg"
baseline_archive=""
baseline_archive_sha256=""
baseline_commit=""
baseline_tree=""
candidate_archive=""
candidate_archive_sha256=""
candidate_commit=""
candidate_tree=""
templates_archive=""
templates_archive_sha256=""
manifest_path=""
output_archive=""
keep_workspace="0"

fail() {
    printf 'Ferrum Linux ref A/B worker: %s\n' "$*" >&2
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
        --baseline-archive) require_value "$@"; baseline_archive="$2"; shift 2 ;;
        --baseline-archive-sha256) require_value "$@"; baseline_archive_sha256="$2"; shift 2 ;;
        --baseline-commit) require_value "$@"; baseline_commit="$2"; shift 2 ;;
        --baseline-tree) require_value "$@"; baseline_tree="$2"; shift 2 ;;
        --candidate-archive) require_value "$@"; candidate_archive="$2"; shift 2 ;;
        --candidate-archive-sha256) require_value "$@"; candidate_archive_sha256="$2"; shift 2 ;;
        --candidate-commit) require_value "$@"; candidate_commit="$2"; shift 2 ;;
        --candidate-tree) require_value "$@"; candidate_tree="$2"; shift 2 ;;
        --templates-archive) require_value "$@"; templates_archive="$2"; shift 2 ;;
        --templates-archive-sha256) require_value "$@"; templates_archive_sha256="$2"; shift 2 ;;
        --manifest) require_value "$@"; manifest_path="$2"; shift 2 ;;
        --output-archive) require_value "$@"; output_archive="$2"; shift 2 ;;
        --keep-workspace) keep_workspace="1"; shift ;;
        *) fail "unknown argument: $1" ;;
    esac
done

if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

for command_name in bash cargo rustc jq tar sha256sum taskset flock findmnt python3 /usr/bin/time; do
    command -v "$command_name" >/dev/null 2>&1 || fail "required command was not found: $command_name"
done

[[ "$cpu_set" =~ ^[0-9]+([,-][0-9]+)*$ ]] || fail "invalid CPU set: $cpu_set"
taskset -c "$cpu_set" true >/dev/null 2>&1 || fail "CPU set is not available: $cpu_set"
[[ "$warmup_runs" =~ ^[0-9]+$ ]] || fail "warmup runs must be a non-negative integer"
[[ "$measured_runs" =~ ^[1-9][0-9]*$ ]] || fail "measured runs must be a positive integer"
((measured_runs % 2 == 0)) || fail "measured runs must be even"
[[ "$pressure_solver" == "pcg" || "$pressure_solver" == "gamg" ]] || fail "unsupported pressure solver: $pressure_solver"

case "$build_variant" in
    portable)
        build_rustflags=""; build_codegen_units=""; build_lto="" ;;
    native)
        build_rustflags="-C target-cpu=native"; build_codegen_units=""; build_lto="" ;;
    native-codegen1)
        build_rustflags="-C target-cpu=native"; build_codegen_units="1"; build_lto="" ;;
    native-thin-lto)
        build_rustflags="-C target-cpu=native"; build_codegen_units=""; build_lto="thin" ;;
    native-fat-lto)
        build_rustflags="-C target-cpu=native"; build_codegen_units=""; build_lto="fat" ;;
    *) fail "unsupported build variant: $build_variant" ;;
esac

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
    printf 'filesystem=%s\n' "$home_fstype"
    printf 'cpu_set=%s\n' "$cpu_set"
    printf 'build_variant=%s\n' "$build_variant"
    printf 'rustflags=%s\n' "${build_rustflags:-<unset>}"
    printf 'cargo_profile_release_codegen_units=%s\n' "${build_codegen_units:-<unset>}"
    printf 'cargo_profile_release_lto=%s\n' "${build_lto:-<unset>}"
    exit 0
fi

for required_file in "$baseline_archive" "$candidate_archive" "$templates_archive" "$manifest_path"; do
    [[ -f "$required_file" ]] || fail "required staged input was not found: $required_file"
done
for required_value in "$baseline_archive_sha256" "$baseline_commit" "$baseline_tree" \
    "$candidate_archive_sha256" "$candidate_commit" "$candidate_tree" \
    "$templates_archive_sha256" "$output_archive"; do
    [[ -n "$required_value" ]] || fail "a required binding value was empty"
done

cache_root="$HOME/.cache/ferrumcfd-linux-ref-ab"
mkdir -p "$cache_root"
exec 9>"$cache_root/benchmark.lock"
flock -n 9 || fail "another Ferrum Linux ref A/B benchmark is active"
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
        printf 'Ferrum Linux ref A/B workspace preserved: %s\n' "$workspace" >&2
    fi
}
trap cleanup EXIT

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
        if not name or name.startswith("/") or re.match(r"^[A-Za-z]:(?:/|$)", name) or ".." in path.parts:
            raise SystemExit(f"{description} archive contains an unsafe path: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"{description} archive contains a non-regular entry: {member.name}")
PY
}

export_root="$workspace/export"
raw_root="$export_root/raw"
metadata_root="$export_root/metadata"
templates_root="$workspace/templates"
mkdir -p "$raw_root" "$metadata_root" "$templates_root"
cp -- "$manifest_path" "$metadata_root/input-manifest.json"

stage_archive() {
    local slot="$1" source_archive="$2" expected_sha="$3"
    local slot_root="$workspace/slots/$slot" archive_copy="$workspace/$slot.tar"
    mkdir -p "$slot_root/source"
    cp -- "$source_archive" "$archive_copy"
    local actual_sha
    actual_sha="$(sha256sum "$archive_copy" | awk '{print $1}')"
    [[ "$actual_sha" == "$expected_sha" ]] || fail "$slot source archive SHA-256 changed while staging"
    validate_archive "$archive_copy" "$slot source"
    tar --no-same-owner --no-same-permissions -xf "$archive_copy" -C "$slot_root/source"
    printf '%s\n' "$actual_sha"
}

actual_baseline_archive_sha256="$(stage_archive A "$baseline_archive" "$baseline_archive_sha256")"
actual_candidate_archive_sha256="$(stage_archive B "$candidate_archive" "$candidate_archive_sha256")"
cp -- "$templates_archive" "$workspace/templates.tar"
actual_templates_sha256="$(sha256sum "$workspace/templates.tar" | awk '{print $1}')"
[[ "$actual_templates_sha256" == "$templates_archive_sha256" ]] || fail "templates archive SHA-256 changed while staging"
validate_archive "$workspace/templates.tar" "matched Ferrum templates"
tar --no-same-owner --no-same-permissions -xf "$workspace/templates.tar" -C "$templates_root"

for root in "$workspace/slots/A/source" "$workspace/slots/B/source" "$templates_root" "$raw_root"; do
    [[ "$(findmnt -T "$root" -n -o FSTYPE | tr -d '[:space:]')" == "ext4" ]] || fail "benchmark path is not on ext4: $root"
done

manifest_baseline="$(jq -r '.baseline.commit' "$metadata_root/input-manifest.json")"
manifest_candidate="$(jq -r '.candidate.commit' "$metadata_root/input-manifest.json")"
[[ "$manifest_baseline" == "$baseline_commit" && "$manifest_candidate" == "$candidate_commit" ]] || fail "manifest commit binding differs"
[[ "$(jq -r '.baseline.tree' "$metadata_root/input-manifest.json")" == "$baseline_tree" ]] || fail "manifest baseline tree differs"
[[ "$(jq -r '.candidate.tree' "$metadata_root/input-manifest.json")" == "$candidate_tree" ]] || fail "manifest candidate tree differs"
[[ "$(jq -r '.pressureSolver' "$metadata_root/input-manifest.json")" == "$pressure_solver" ]] || fail "manifest pressure solver differs"

baseline_lock_sha="$(sha256sum "$workspace/slots/A/source/Cargo.lock" | awk '{print $1}')"
candidate_lock_sha="$(sha256sum "$workspace/slots/B/source/Cargo.lock" | awk '{print $1}')"
expected_lock_sha="$(jq -r '.cargoLock.sha256' "$metadata_root/input-manifest.json")"
[[ "$baseline_lock_sha" == "$candidate_lock_sha" && "$baseline_lock_sha" == "$expected_lock_sha" ]] || fail "Cargo.lock content differs between refs or manifest"

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

build_environment=("CARGO_INCREMENTAL=0")
if [[ -n "$build_rustflags" ]]; then build_environment+=("RUSTFLAGS=$build_rustflags"); fi
if [[ -n "$build_codegen_units" ]]; then build_environment+=("CARGO_PROFILE_RELEASE_CODEGEN_UNITS=$build_codegen_units"); fi
if [[ -n "$build_lto" ]]; then build_environment+=("CARGO_PROFILE_RELEASE_LTO=$build_lto"); fi
build_format=$'elapsed_s=%e\nuser_s=%U\nsystem_s=%S\nmax_rss_kb=%M\nexit=%x'

build_slot() {
    local slot="$1" ref_name="$2"
    local source_root="$workspace/slots/$slot/source"
    local target_root="$workspace/slots/$slot/target"
    local build_timing="$metadata_root/build-$ref_name-time.env"
    local build_log="$metadata_root/cargo-build-$ref_name-release.log"
    set +e
    (
        cd "$source_root"
        /usr/bin/time -q -f "$build_format" -o "$build_timing" \
            env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS \
            -u CARGO_PROFILE_RELEASE_CODEGEN_UNITS -u CARGO_PROFILE_RELEASE_LTO \
            -u CARGO_INCREMENTAL \
            "${build_environment[@]}" "CARGO_TARGET_DIR=$target_root" \
            cargo "+$rust_toolchain" build --locked --release -p ferrum-run --bin ferrumRun \
            >"$build_log" 2>&1
    )
    local status=$?
    set -e
    [[ "$status" -eq 0 ]] || fail "$ref_name Ferrum Linux release build failed; see $build_log"
    [[ -x "$target_root/release/ferrumRun" ]] || fail "$ref_name Linux Ferrum executable was not produced"
    mkdir -p "$workspace/binaries/$slot"
    cp -- "$target_root/release/ferrumRun" "$workspace/binaries/$slot/ferrumRun"
    chmod 0555 "$workspace/binaries/$slot/ferrumRun"
}

build_slot A baseline
build_slot B candidate
baseline_binary="$workspace/binaries/A/ferrumRun"
candidate_binary="$workspace/binaries/B/ferrumRun"

printf '%s\n' "$baseline_commit" >"$metadata_root/baseline-commit.txt"
printf '%s\n' "$baseline_tree" >"$metadata_root/baseline-tree.txt"
printf '%s\n' "$actual_baseline_archive_sha256" >"$metadata_root/baseline-archive-sha256.txt"
printf '%s\n' "$candidate_commit" >"$metadata_root/candidate-commit.txt"
printf '%s\n' "$candidate_tree" >"$metadata_root/candidate-tree.txt"
printf '%s\n' "$actual_candidate_archive_sha256" >"$metadata_root/candidate-archive-sha256.txt"
printf '%s\n' "$actual_templates_sha256" >"$metadata_root/templates-archive-sha256.txt"
printf '%s\n' "$baseline_lock_sha" >"$metadata_root/cargo-lock-sha256.txt"
sha256sum "$baseline_binary" | awk '{print $1}' >"$metadata_root/baseline-binary-sha256.txt"
sha256sum "$candidate_binary" | awk '{print $1}' >"$metadata_root/candidate-binary-sha256.txt"
rustc "+$rust_toolchain" -vV >"$metadata_root/rustc-vv.txt"
cargo "+$rust_toolchain" --version >"$metadata_root/cargo-version.txt"
uname -a >"$metadata_root/uname.txt"
grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '"' >"$metadata_root/distro-release.txt"
awk -F: '/model name/ {sub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo >"$metadata_root/cpu-model.txt"
printf '%s\n' "$cpu_set" >"$metadata_root/cpu-set.txt"
first_cpu="${cpu_set%%[,-]*}"
cat "/sys/devices/system/cpu/cpu$first_cpu/topology/thread_siblings_list" >"$metadata_root/cpu-siblings.txt"
printf '%s\n' "$home_fstype" >"$metadata_root/filesystem-type.txt"
printf '%s\n' "$build_variant" >"$metadata_root/build-variant.txt"
printf '%s\n' "$build_rustflags" >"$metadata_root/build-rustflags.txt"
printf '%s\n' "$build_codegen_units" >"$metadata_root/build-cargo-profile-release-codegen-units.txt"
printf '%s\n' "$build_lto" >"$metadata_root/build-cargo-profile-release-lto.txt"
printf '%s\n' "$workspace" >"$metadata_root/workspace-path.txt"

time_format=$'elapsed_s=%e\nuser_s=%U\nsystem_s=%S\nmax_rss_kb=%M\nexit=%x'
thread_environment=(LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1)
printf 'case\tkind\tordinal\tposition\tref\n' >"$metadata_root/run-order.tsv"

numeric_output_count() {
    find "$1" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
        awk '$0 != "0" && $0 ~ /^[0-9]+([.][0-9]+)?$/ {count++} END {print count+0}'
}

canonicalize_report() {
    local report_path="$1" canonical_path="$2" hash_path="$3"
    python3 - "$report_path" "$canonical_path" "$hash_path" <<'PY'
import hashlib
import json
import pathlib
import sys

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
                removed.append(".".join(child_path))
                continue
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

run_timed_ref() {
    local ref_name="$1" binary="$2" case_name="$3" kind="$4" ordinal="$5" fixed_iterations="$6" run_root="$7"
    local working_case="$run_root/case" log_path="$run_root/ferrum.log"
    local report_path="$run_root/solve-report.json" timing_path="$run_root/process-time.env"
    mkdir -p "$working_case"
    cp -a "$templates_root/$case_name/." "$working_case/"
    set +e
    (
        cd "$run_root"
        /usr/bin/time -q -f "$time_format" -o "$timing_path" \
            taskset -c "$cpu_set" env "${thread_environment[@]}" \
            "$binary" -solver incompressibleFluid -case "$working_case" \
            --minSimpleIterations "$fixed_iterations" --maxSimpleIterations "$fixed_iterations" \
            --solveReportJson "$report_path" >"$log_path" 2>&1
    )
    local status=$?
    set -e
    [[ "$status" -eq 0 ]] || fail "$ref_name run failed for $case_name ($kind $ordinal)"
    [[ "$(numeric_output_count "$working_case")" -eq 0 ]] || fail "$ref_name wrote an unexpected time directory"
    [[ ! -e "$run_root/final-fields" ]] || fail "$ref_name timing run wrote final fields"
    canonicalize_report "$report_path" "$run_root/canonical-report.json" "$run_root/canonical-report.sha256"
}

write_field_oracle() {
    local report_path="$1" fields_root="$2" output_path="$3"
    python3 - "$report_path" "$fields_root" "$output_path" <<'PY'
import hashlib
import json
import math
import pathlib
import re
import struct
import sys

report_path, fields_root, output_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
report = json.loads(report_path.read_text(encoding="utf-8-sig"))
cells = report.get("mesh", {}).get("cells")
if not isinstance(cells, int) or cells <= 0:
    raise SystemExit("oracle report has no positive integer cell count")

def read_field(name, kind, components):
    path = fields_root / name
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if len(re.findall(rf"(?m)^\s*class\s+vol{'Vector' if components == 3 else 'Scalar'}Field\s*;", text)) != 1:
        raise SystemExit(f"{name} field class is not exact")
    if len(re.findall(rf"(?m)^\s*object\s+{name}\s*;", text)) != 1:
        raise SystemExit(f"{name} field object is not exact")
    pattern = re.compile(rf"\binternalField\s+nonuniform\s+List<{kind}>\s+(\d+)\s*\((.*?)\)\s*;", re.S)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise SystemExit(f"{name} must contain exactly one nonuniform internalField")
    declared = int(matches[0].group(1))
    if declared != cells:
        raise SystemExit(f"{name} declares {declared} values, expected {cells}")
    payload = matches[0].group(2)
    values = []
    if components == 1:
        tokens = payload.split()
        if len(tokens) != declared:
            raise SystemExit(f"{name} payload count differs")
        values = [float(token) for token in tokens]
    else:
        vector_pattern = re.compile(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)")
        vectors = list(vector_pattern.finditer(payload))
        residue = vector_pattern.sub("", payload)
        if residue.strip() or len(vectors) != declared:
            raise SystemExit(f"{name} vector payload is malformed")
        values = [float(token) for vector in vectors for token in vector.groups()]
    if len(values) != declared * components or not all(math.isfinite(value) for value in values):
        raise SystemExit(f"{name} parsed value contract failed")
    bits = b"".join(struct.pack(">d", value) for value in values)
    roundtrip = [struct.unpack(">d", bits[offset:offset + 8])[0] for offset in range(0, len(bits), 8)]
    if any(struct.pack(">d", left) != struct.pack(">d", right) for left, right in zip(values, roundtrip)):
        raise SystemExit(f"{name} IEEE-754 round-trip failed")

    number = r"[-+0-9.eE]+"
    if components == 1:
        boundary_pattern = re.compile(
            rf"\bvalue\s+(?:uniform\s+({number})\s*;|nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;)",
            re.S,
        )
    else:
        boundary_pattern = re.compile(
            rf"\bvalue\s+(?:uniform\s+\(\s*({number})\s+({number})\s+({number})\s*\)\s*;|"
            rf"nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;)",
            re.S,
        )
    boundary_matches = list(boundary_pattern.finditer(text))
    if len(re.findall(r"\bvalue\b", text)) != len(boundary_matches):
        raise SystemExit(f"{name} contains an unsupported boundary value entry")
    boundary_bits_parts = []
    boundary_scalar_slots = 0
    for entry in boundary_matches:
        groups = entry.groups()
        if components == 1:
            if groups[0] is not None:
                entry_values = [float(groups[0])]
                declared_boundary = 1
                marker = b"uniform\0"
            else:
                declared_boundary = int(groups[1])
                tokens = groups[2].split()
                if len(tokens) != declared_boundary:
                    raise SystemExit(f"{name} boundary scalar payload count differs")
                entry_values = [float(token) for token in tokens]
                marker = b"nonuniform\0"
        else:
            if groups[0] is not None:
                entry_values = [float(groups[0]), float(groups[1]), float(groups[2])]
                declared_boundary = 1
                marker = b"uniform\0"
            else:
                declared_boundary = int(groups[3])
                vector_pattern = re.compile(rf"\(\s*({number})\s+({number})\s+({number})\s*\)")
                vectors = list(vector_pattern.finditer(groups[4]))
                residue = vector_pattern.sub("", groups[4])
                if residue.strip() or len(vectors) != declared_boundary:
                    raise SystemExit(f"{name} boundary vector payload is malformed")
                entry_values = [float(token) for vector in vectors for token in vector.groups()]
                marker = b"nonuniform\0"
        if len(entry_values) != declared_boundary * components or not all(math.isfinite(value) for value in entry_values):
            raise SystemExit(f"{name} boundary value contract failed")
        entry_bits = b"".join(struct.pack(">d", value) for value in entry_values)
        entry_roundtrip = [struct.unpack(">d", entry_bits[offset:offset + 8])[0] for offset in range(0, len(entry_bits), 8)]
        if any(struct.pack(">d", left) != struct.pack(">d", right) for left, right in zip(entry_values, entry_roundtrip)):
            raise SystemExit(f"{name} boundary IEEE-754 round-trip failed")
        boundary_bits_parts.append(marker + struct.pack(">Q", declared_boundary) + entry_bits)
        boundary_scalar_slots += len(entry_values)
    boundary_bits = b"".join(boundary_bits_parts)
    full_field_bits = b"FerrumCFD-field-v1\0" + name.encode("ascii") + b"\0" + bits + boundary_bits
    return {
        "name": name,
        "declaredValues": declared,
        "components": components,
        "scalarSlots": len(values),
        "boundaryValueEntries": len(boundary_matches),
        "boundaryScalarSlots": boundary_scalar_slots,
        "textSha256": hashlib.sha256(raw).hexdigest(),
        "ieee754BigEndianSha256": hashlib.sha256(bits).hexdigest(),
        "boundaryIeee754BigEndianSha256": hashlib.sha256(boundary_bits).hexdigest(),
        "fullFieldIeee754BigEndianSha256": hashlib.sha256(full_field_bits).hexdigest(),
    }, full_field_bits

u_summary, u_bits = read_field("U", "vector", 3)
p_summary, p_bits = read_field("p", "scalar", 1)
combined = b"FerrumCFD-field-oracle-v2\0" + struct.pack(">Q", cells) + u_bits + p_bits
result = {
    "schemaVersion": 2,
    "cellCount": cells,
    "encoding": "IEEE-754 binary64 big-endian",
    "U": u_summary,
    "p": p_summary,
    "combinedIeee754Sha256": hashlib.sha256(combined).hexdigest(),
}
output_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
}

run_oracle_ref() {
    local ref_name="$1" binary="$2" case_name="$3" fixed_iterations="$4" run_root="$5"
    local working_case="$run_root/case" report_path="$run_root/solve-report.json" fields_root="$run_root/final-fields"
    mkdir -p "$working_case"
    cp -a "$templates_root/$case_name/." "$working_case/"
    set +e
    (
        cd "$run_root"
        taskset -c "$cpu_set" env "${thread_environment[@]}" \
            "$binary" -solver incompressibleFluid -case "$working_case" \
            --minSimpleIterations "$fixed_iterations" --maxSimpleIterations "$fixed_iterations" \
            --solveReportJson "$report_path" --writeFinalFields "final-fields" >"$run_root/ferrum.log" 2>&1
    )
    local status=$?
    set -e
    [[ "$status" -eq 0 ]] || fail "$ref_name final-field oracle failed for $case_name"
    [[ "$(numeric_output_count "$working_case")" -eq 0 ]] || fail "$ref_name oracle wrote an unexpected time directory"
    [[ -f "$fields_root/U" && -f "$fields_root/p" ]] || fail "$ref_name oracle did not write exact U and p fields"
    [[ "$(find "$fields_root" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | sort | tr '\n' ' ')" == "U p " ]] || fail "$ref_name oracle field inventory was not exactly U and p"
    canonicalize_report "$report_path" "$run_root/canonical-report.json" "$run_root/canonical-report.sha256"
    write_field_oracle "$report_path" "$fields_root" "$run_root/field-oracle.json"
}

mapfile -t case_rows < <(jq -r '.cases[] | [.name, .fixedIterations] | @tsv' "$metadata_root/input-manifest.json")
for case_row in "${case_rows[@]}"; do
    IFS=$'\t' read -r case_name fixed_iterations <<<"$case_row"
    for kind in warmup measured; do
        if [[ "$kind" == "warmup" ]]; then count="$warmup_runs"; else count="$measured_runs"; fi
        for ((ordinal=1; ordinal<=count; ordinal++)); do
            if ((ordinal % 2 == 1)); then refs=(baseline candidate); else refs=(candidate baseline); fi
            position=0
            for ref_name in "${refs[@]}"; do
                position=$((position + 1))
                printf '%s\t%s\t%s\t%s\t%s\n' "$case_name" "$kind" "$ordinal" "$position" "$ref_name" >>"$metadata_root/run-order.tsv"
                run_root="$raw_root/$case_name/$kind-$ordinal-$ref_name"
                mkdir -p "$run_root"
                if [[ "$ref_name" == "baseline" ]]; then binary="$baseline_binary"; else binary="$candidate_binary"; fi
                run_timed_ref "$ref_name" "$binary" "$case_name" "$kind" "$ordinal" "$fixed_iterations" "$run_root"
            done
        done
    done
    run_oracle_ref baseline "$baseline_binary" "$case_name" "$fixed_iterations" "$raw_root/$case_name/oracle-baseline"
    run_oracle_ref candidate "$candidate_binary" "$case_name" "$fixed_iterations" "$raw_root/$case_name/oracle-candidate"
done

archive_on_ext4="$workspace/ferrum-linux-ref-ab-results.tar"
tar -cf "$archive_on_ext4" -C "$export_root" .
archive_sha256="$(sha256sum "$archive_on_ext4" | awk '{print $1}')"
printf '%s\n' "$archive_sha256" >"$workspace/ferrum-linux-ref-ab-results.tar.sha256"
mkdir -p "$(dirname "$output_archive")"
cp -- "$archive_on_ext4" "$output_archive"
cp -- "$workspace/ferrum-linux-ref-ab-results.tar.sha256" "$output_archive.sha256"
completed="1"
printf 'output_archive=%s\n' "$output_archive"
printf 'output_archive_sha256=%s\n' "$archive_sha256"
printf 'workspace=%s\n' "$workspace"
