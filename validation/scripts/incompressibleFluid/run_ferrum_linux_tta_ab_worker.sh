#!/usr/bin/env bash
set -euo pipefail

mode="run"
rust_toolchain="1.94.0"
cpu_set="2"
build_variant="native"
warmup_runs="2"
measured_runs="10"
pressure_solver="gamg"
max_simple_iterations="2000"
candidate_pressure_reltol="0.05"
candidate_momentum_reltol="0.05"
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
controls_archive=""
controls_archive_sha256=""
manifest_path=""
output_archive=""
keep_workspace="0"

fail() {
    printf 'Ferrum Linux TTA A/B worker: %s\n' "$*" >&2
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
        --max-simple-iterations) require_value "$@"; max_simple_iterations="$2"; shift 2 ;;
        --candidate-pressure-reltol) require_value "$@"; candidate_pressure_reltol="$2"; shift 2 ;;
        --candidate-momentum-reltol) require_value "$@"; candidate_momentum_reltol="$2"; shift 2 ;;
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
        --controls-archive) require_value "$@"; controls_archive="$2"; shift 2 ;;
        --controls-archive-sha256) require_value "$@"; controls_archive_sha256="$2"; shift 2 ;;
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

run_reltol_tool() {
    python3 - "$@" <<'PY'
import math
import pathlib
import sys
from dataclasses import dataclass

class ContractError(RuntimeError):
    pass

@dataclass(frozen=True)
class Token:
    kind: str
    text: bytes
    start: int
    end: int

WHITESPACE = b" \t\r\n\f\v"

def lex(raw: bytes):
    tokens = []
    index = 3 if raw.startswith(b"\xef\xbb\xbf") else 0
    while index < len(raw):
        byte = raw[index]
        if byte in WHITESPACE:
            index += 1
            continue
        if raw[index:index + 2] == b"//":
            newline = raw.find(b"\n", index + 2)
            index = len(raw) if newline < 0 else newline + 1
            continue
        if raw[index:index + 2] == b"/*":
            closing = raw.find(b"*/", index + 2)
            if closing < 0:
                raise ContractError(f"unterminated block comment at byte {index}")
            index = closing + 2
            continue
        if byte == ord("#"):
            line_start = max(raw.rfind(b"\n", 0, index), raw.rfind(b"\r", 0, index)) + 1
            if raw[line_start:index].strip(b" \t"):
                raise ContractError(f"directive marker outside logical-line start at byte {index}")
            newline = raw.find(b"\n", index + 1)
            end = len(raw) if newline < 0 else newline + 1
            tokens.append(Token("directive", raw[index:end], index, end))
            index = end
            continue
        if byte in (ord('"'), ord("'")):
            quote = byte
            start = index
            index += 1
            while index < len(raw):
                if raw[index] == ord("\\"):
                    if index + 1 >= len(raw):
                        break
                    index += 2
                    continue
                if raw[index] == quote:
                    index += 1
                    tokens.append(Token("string", raw[start:index], start, index))
                    break
                index += 1
            else:
                raise ContractError(f"unterminated quoted string at byte {start}")
            if not tokens or tokens[-1].start != start:
                raise ContractError(f"unterminated quoted string at byte {start}")
            continue
        if byte in (ord("{"), ord("}"), ord(";")):
            tokens.append(Token("punct", raw[index:index + 1], index, index + 1))
            index += 1
            continue
        start = index
        while index < len(raw):
            byte = raw[index]
            if byte in WHITESPACE or byte in (ord("{"), ord("}"), ord(";"), ord('"'), ord("'"), ord("#")):
                break
            if raw[index:index + 2] in (b"//", b"/*"):
                break
            index += 1
        if index == start:
            raise ContractError(f"unsupported token at byte {index}")
        tokens.append(Token("word", raw[start:index], start, index))
    return tokens

def brace_pairs(tokens):
    stack = []
    pairs = {}
    for index, token in enumerate(tokens):
        if token.kind != "punct":
            continue
        if token.text == b"{":
            stack.append(index)
        elif token.text == b"}":
            if not stack:
                raise ContractError(f"unmatched closing brace at byte {token.start}")
            pairs[stack.pop()] = index
    if stack:
        raise ContractError(f"unterminated dictionary block at byte {tokens[stack[-1]].start}")
    return pairs

def direct_entries(tokens, start, end, pairs, description):
    entries = []
    index = start
    while index < end:
        key = tokens[index]
        if key.kind not in ("word", "string"):
            raise ContractError(f"{description} expected an entry key at byte {key.start}")
        index += 1
        if index >= end:
            raise ContractError(f"{description}.{key.text!r} is missing a value or dictionary")
        if tokens[index].kind == "punct" and tokens[index].text == b"{":
            if index not in pairs or pairs[index] >= end:
                raise ContractError(f"{description}.{key.text!r} has an invalid dictionary block")
            closing = pairs[index]
            entries.append({"key": key, "kind": "dictionary", "opening": index, "closing": closing, "values": []})
            index = closing + 1
            if index < end and tokens[index].kind == "punct" and tokens[index].text == b";":
                index += 1
            continue
        value_start = index
        while index < end and not (tokens[index].kind == "punct" and tokens[index].text == b";"):
            if tokens[index].kind == "punct" and tokens[index].text in (b"{", b"}"):
                raise ContractError(f"{description}.{key.text!r} contains an unexpected block")
            index += 1
        if index >= end:
            raise ContractError(f"{description}.{key.text!r} is missing its semicolon")
        entries.append({"key": key, "kind": "scalar", "opening": None, "closing": None, "values": tokens[value_start:index]})
        index += 1
    return entries

def ordinary(entries, name: bytes):
    return [entry for entry in entries if entry["key"].kind == "word" and entry["key"].text == name]

def locate_reltol_tokens(raw: bytes):
    tokens = lex(raw)
    directives = [token for token in tokens if token.kind == "directive"]
    if directives:
        raise ContractError(f"active OpenFOAM directive at byte {directives[0].start}")
    pairs = brace_pairs(tokens)
    top = direct_entries(tokens, 0, len(tokens), pairs, "fvSolution")
    solvers = ordinary(top, b"solvers")
    if len(solvers) != 1 or solvers[0]["kind"] != "dictionary":
        raise ContractError("fvSolution must contain exactly one direct ordinary solvers dictionary")
    solvers_entry = solvers[0]
    children = direct_entries(tokens, solvers_entry["opening"] + 1, solvers_entry["closing"], pairs, "solvers")
    result = {}
    for name in (b"p", b"U"):
        sections = ordinary(children, name)
        label = name.decode("ascii")
        if len(sections) != 1 or sections[0]["kind"] != "dictionary":
            raise ContractError(f"solvers.{label} must be exactly one direct ordinary dictionary")
        section = sections[0]
        options = direct_entries(tokens, section["opening"] + 1, section["closing"], pairs, f"solvers.{label}")
        reltols = ordinary(options, b"relTol")
        if len(reltols) != 1 or reltols[0]["kind"] != "scalar":
            raise ContractError(f"solvers.{label}.relTol must be exactly one direct scalar")
        values = reltols[0]["values"]
        if len(values) != 1 or values[0].kind != "word":
            raise ContractError(f"solvers.{label}.relTol must have one unquoted numeric token")
        try:
            parsed = float(values[0].text.decode("ascii"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ContractError(f"solvers.{label}.relTol is not numeric") from exc
        if not math.isfinite(parsed) or parsed < 0.0:
            raise ContractError(f"solvers.{label}.relTol must be finite and non-negative")
        result[label] = values[0]
    return result

def expected_value(raw: str, label: str):
    try:
        value = float(raw)
        encoded = raw.encode("ascii")
    except (ValueError, UnicodeEncodeError) as exc:
        raise ContractError(f"expected {label} relTol is invalid") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ContractError(f"expected {label} relTol must be finite and non-negative")
    return value, encoded

def verify_bytes(raw: bytes, expected_p: str, expected_u: str):
    locations = locate_reltol_tokens(raw)
    for name, expected_raw in (("p", expected_p), ("U", expected_u)):
        expected, _ = expected_value(expected_raw, name)
        actual = float(locations[name].text.decode("ascii"))
        if actual != expected:
            raise ContractError(f"solvers.{name}.relTol differs from exact expected value")

def patch_bytes(raw: bytes, expected_p: str, expected_u: str):
    locations = locate_reltol_tokens(raw)
    replacements = []
    for name, expected_raw in (("p", expected_p), ("U", expected_u)):
        _, encoded = expected_value(expected_raw, name)
        token = locations[name]
        replacements.append((token.start, token.end, encoded))
    result = raw
    for start, end, replacement in sorted(replacements, reverse=True):
        result = result[:start] + replacement + result[end:]
    verify_bytes(result, expected_p, expected_u)
    return result

def expect_reject(raw: bytes, description: str):
    try:
        locate_reltol_tokens(raw)
    except ContractError:
        return
    raise ContractError(f"self-test accepted malformed {description}")

def self_test():
    sample = b'''// p { relTol 9; }\n/* U { relTol 8; } } */\n"solvers" { p { relTol 7; } U { relTol 7; } }\nsolvers\n{\n p\n {\n  solver GAMG;\n  note "} relTol 6; \\\"quoted\\\"";\n  relTol 0;\n }\n U { solver smoothSolver; relTol 0; }\n T { solver smoothSolver; relTol 0.25; }\n}\n'''
    locations = locate_reltol_tokens(sample)
    expected = sample
    for start, end, replacement in sorted(((locations["p"].start, locations["p"].end, b"0.05"),
                                            (locations["U"].start, locations["U"].end, b"0.125")), reverse=True):
        expected = expected[:start] + replacement + expected[end:]
    patched = patch_bytes(sample, "0.05", "0.125")
    if patched != expected or b"T { solver smoothSolver; relTol 0.25; }" not in patched:
        raise ContractError("self-test changed bytes outside direct p/U relTol numeric tokens")
    verify_bytes(patched, "0.05", "0.125")
    malformed = {
        "missing U": b"solvers { p { relTol 0; } }",
        "duplicate p": b"solvers { p { relTol 0; } p { relTol 0; } U { relTol 0; } }",
        "nested p/U": b"solvers { group { p { relTol 0; } U { relTol 0; } } }",
        "nested relTol": b"solvers { p { controls { relTol 0; } } U { relTol 0; } }",
        "duplicate relTol": b"solvers { p { relTol 0; relTol 0; } U { relTol 0; } }",
        "non-scalar relTol": b"solvers { p { relTol { value 0; } } U { relTol 0; } }",
        "active directive": b"#include \"other\"\nsolvers { p { relTol 0; } U { relTol 0; } }",
        "unterminated comment": b"solvers { p { relTol 0; } U { relTol 0; } } /*",
        "unterminated string": b"solvers { p { note \"bad; relTol 0; } U { relTol 0; } }",
        "unterminated block": b"solvers { p { relTol 0; } U { relTol 0; }",
    }
    for description, raw in malformed.items():
        expect_reject(raw, description)

mode = sys.argv[1]
if mode == "self-test":
    self_test()
elif mode in ("patch", "verify"):
    if len(sys.argv) != 5:
        raise SystemExit("reltol tool requires mode, path, p, U")
    path = pathlib.Path(sys.argv[2])
    before = path.read_bytes()
    if mode == "verify":
        verify_bytes(before, sys.argv[3], sys.argv[4])
    else:
        after = patch_bytes(before, sys.argv[3], sys.argv[4])
        if after == before and (float(sys.argv[3]) != 0.0 or float(sys.argv[4]) != 0.0):
            raise ContractError("requested nonzero patch made no byte change")
        path.write_bytes(after)
        if path.read_bytes() != after:
            raise ContractError("patched fvSolution bytes differ after write")
else:
    raise SystemExit(f"unknown reltol tool mode: {mode}")
PY
}

run_report_proof_tool() {
    python3 - "$@" <<'PY'
import hashlib, json, math, os, pathlib, re, sys, tempfile

class ProofError(RuntimeError):
    pass

ENTRY_KEYS = {"case", "kind", "ordinal", "ref", "relativePath", "sha256"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

def strict_json_loads(text):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ProofError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject_constant(value):
        raise ProofError(f"non-finite JSON constant: {value}")
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ProofError("non-finite JSON number")
        if isinstance(value, dict):
            for child in value.values(): finite(child)
        elif isinstance(value, list):
            for child in value: finite(child)
    result = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
    finite(result)
    return result

def identity(case, kind, ordinal, ref):
    if type(case) is not str or not SAFE_NAME.fullmatch(case) or case in (".", ".."):
        raise ProofError("report proof case name is unsafe")
    if ref not in ("baseline", "candidate"):
        raise ProofError("report proof ref is invalid")
    if type(ordinal) is not int or ordinal < 0:
        raise ProofError("report proof ordinal is invalid")
    if kind == "oracle":
        if ordinal != 0: raise ProofError("oracle report proof ordinal must be zero")
        run = f"oracle-{ref}"
    elif kind in ("warmup", "measured"):
        if ordinal < 1: raise ProofError("timed report proof ordinal must be positive")
        run = f"{kind}-{ordinal}-{ref}"
    else:
        raise ProofError("report proof kind is invalid")
    relative = f"raw/{case}/{run}/solve-report.json"
    return {"case": case, "kind": kind, "ordinal": ordinal, "ref": ref, "relativePath": relative}

def validate_entry(entry):
    if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
        raise ProofError("report proof entry shape differs")
    expected = identity(entry["case"], entry["kind"], entry["ordinal"], entry["ref"])
    if entry["relativePath"] != expected["relativePath"]:
        raise ProofError("report proof path differs from run identity")
    if type(entry["sha256"]) is not str or not SHA256.fullmatch(entry["sha256"]):
        raise ProofError("report proof SHA-256 is invalid")
    return entry

def load_entries(path):
    if not path.exists(): return []
    if path.is_symlink() or not path.is_file(): raise ProofError("report proof journal is not a regular file")
    entries, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line: raise ProofError(f"report proof journal line {number} is empty")
        try: entry = validate_entry(strict_json_loads(line))
        except (json.JSONDecodeError, ProofError) as exc: raise ProofError(f"report proof journal line {number} is invalid") from exc
        if entry["relativePath"] in seen: raise ProofError("report proof journal contains a duplicate path")
        seen.add(entry["relativePath"]); entries.append(entry)
    return entries

def regular_report(export_root, report_path, expected_relative):
    root = export_root.resolve(strict=True)
    if export_root.is_symlink() or not export_root.is_dir(): raise ProofError("report proof export root is unsafe")
    if report_path.is_symlink() or not report_path.is_file(): raise ProofError("validated report is not a regular file")
    resolved = report_path.resolve(strict=True)
    if resolved != report_path.absolute(): raise ProofError("validated report traverses a symbolic link")
    try: relative = resolved.relative_to(root).as_posix()
    except ValueError as exc: raise ProofError("validated report escaped export root") from exc
    if relative != expected_relative: raise ProofError("validated report path differs from run identity")
    return resolved

def expected_identities(manifest, warmup, measured):
    cases = manifest.get("cases") if isinstance(manifest, dict) else None
    if not isinstance(cases, list) or not cases: raise ProofError("report proof manifest cases are invalid")
    result = {}
    for case_row in cases:
        if not isinstance(case_row, dict): raise ProofError("report proof manifest case is not an object")
        case = case_row.get("name")
        for kind, count in (("warmup", warmup), ("measured", measured)):
            for ordinal in range(1, count + 1):
                for ref in ("baseline", "candidate"):
                    row = identity(case, kind, ordinal, ref); result[row["relativePath"]] = row
        for ref in ("baseline", "candidate"):
            row = identity(case, "oracle", 0, ref); result[row["relativePath"]] = row
    if len(result) != len(cases) * (2 * (warmup + measured) + 2):
        raise ProofError("report proof expected identity set is not unique")
    return result

def finalize(entries_path, export_root, proof_path, proof_hash_path, manifest_path,
             controls_sha, warmup, measured, max_simple, pressure_solver, pressure_reltol, momentum_reltol):
    if min(warmup, measured, max_simple) < 1: raise ProofError("report proof run policy is invalid")
    if not SHA256.fullmatch(controls_sha): raise ProofError("report proof controls SHA-256 is invalid")
    manifest_bytes = manifest_path.read_bytes()
    manifest = strict_json_loads(manifest_bytes.decode("utf-8-sig"))
    if manifest.get("benchmark") != "ferrum-linux-time-to-accuracy-ab": raise ProofError("report proof benchmark differs")
    if manifest.get("controls", {}).get("archiveSha256") != controls_sha: raise ProofError("report proof controls binding differs")
    if manifest.get("maxSimpleIterations") != max_simple or manifest.get("pressureSolver") != pressure_solver:
        raise ProofError("report proof solver policy differs from manifest")
    if manifest.get("candidateRelTol") != {"p": pressure_reltol, "U": momentum_reltol}:
        raise ProofError("report proof relTol policy differs from manifest")
    expected = expected_identities(manifest, warmup, measured)
    entries = load_entries(entries_path)
    recorded = {entry["relativePath"]: entry for entry in entries}
    if set(recorded) != set(expected): raise ProofError("report proof journal does not exactly cover expected runs")
    raw_root = export_root / "raw"
    if raw_root.is_symlink() or not raw_root.is_dir(): raise ProofError("report proof raw root is unsafe")
    actual = set()
    for path in raw_root.rglob("solve-report.json"):
        resolved = regular_report(export_root, path, path.relative_to(export_root).as_posix())
        relative = resolved.relative_to(export_root.resolve(strict=True)).as_posix()
        if relative in actual: raise ProofError("raw report inventory contains a duplicate path")
        actual.add(relative)
    if actual != set(expected): raise ProofError("raw report inventory does not exactly match expected runs")
    verified = []
    for relative in sorted(expected):
        entry = recorded[relative]
        for key, value in expected[relative].items():
            if entry[key] != value: raise ProofError("report proof identity differs from expected run")
        report = regular_report(export_root, export_root / pathlib.PurePosixPath(relative), relative)
        if hashlib.sha256(report.read_bytes()).hexdigest() != entry["sha256"]:
            raise ProofError("validated report changed after exact contract validation")
        verified.append(entry)
    proof = {
        "benchmark": "ferrum-linux-time-to-accuracy-ab",
        "controlsArchiveSha256": controls_sha,
        "inputManifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "reportCount": len(verified),
        "reports": verified,
        "runPolicy": {"candidateRelTol": {"U": momentum_reltol, "p": pressure_reltol},
                      "maxSimpleIterations": max_simple, "measuredRuns": measured,
                      "pressureSolver": pressure_solver, "warmupRuns": warmup},
        "schemaVersion": 1,
        "validator": "worker-python-exact-report-contract-v1",
    }
    payload = (json.dumps(proof, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    for target in (proof_path, proof_hash_path):
        if target.exists() and (target.is_symlink() or not target.is_file()): raise ProofError("report proof output path is unsafe")
        target.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    proof_hash_path.write_text(digest + "\n", encoding="ascii")
    if hashlib.sha256(proof_path.read_bytes()).hexdigest() != digest: raise ProofError("report proof write verification failed")

def expect_failure(action, description):
    try: action()
    except (ProofError, json.JSONDecodeError): return
    raise ProofError(f"report proof self-test accepted {description}")

def self_test():
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary) / "export"; root.mkdir()
        metadata = root / "metadata"; metadata.mkdir()
        manifest = {"benchmark": "ferrum-linux-time-to-accuracy-ab", "candidateRelTol": {"p": "0.1", "U": "0.2"},
                    "controls": {"archiveSha256": "a" * 64}, "maxSimpleIterations": 5,
                    "pressureSolver": "gamg", "cases": [{"name": "caseA"}]}
        manifest_path = metadata / "input-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        entries_path = pathlib.Path(temporary) / "entries.jsonl"
        expected = expected_identities(manifest, 1, 1)
        entries = []
        for index, row in enumerate(expected.values()):
            path = root / pathlib.PurePosixPath(row["relativePath"]); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"report-{index}".encode("ascii"))
            entry = dict(row); entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest(); entries.append(entry)
        entries_path.write_text("".join(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n" for entry in entries), encoding="utf-8")
        first = next(iter(expected.values())); first_path = root / pathlib.PurePosixPath(first["relativePath"])
        expect_failure(lambda: identity("../bad", "oracle", 0, "baseline"), "an unsafe report identity")
        proof_path = metadata / "exact-report-validation.json"; hash_path = metadata / "exact-report-validation.sha256"
        finalize(entries_path, root, proof_path, hash_path, manifest_path, "a" * 64, 1, 1, 5, "gamg", "0.1", "0.2")
        proof = strict_json_loads(proof_path.read_text(encoding="utf-8"))
        if proof["reportCount"] != 6 or hashlib.sha256(proof_path.read_bytes()).hexdigest() != hash_path.read_text().strip():
            raise ProofError("report proof self-test output differs")
        expect_failure(lambda: strict_json_loads('{"x":1,"x":2}'), "a duplicate JSON key")
        expect_failure(lambda: strict_json_loads('{"x":NaN}'), "a NaN JSON constant")
        expect_failure(lambda: strict_json_loads('{"x":Infinity}'), "an infinite JSON constant")
        expect_failure(lambda: strict_json_loads('{"x":1e999}'), "an overflowing JSON number")
        valid_journal = entries_path.read_bytes(); entries_path.write_bytes(valid_journal + valid_journal.splitlines(keepends=True)[0])
        expect_failure(lambda: finalize(entries_path, root, proof_path, hash_path, manifest_path, "a" * 64, 1, 1, 5, "gamg", "0.1", "0.2"), "a duplicate report")
        entries_path.write_bytes(valid_journal)
        original = first_path.read_bytes(); first_path.write_bytes(b"tampered")
        expect_failure(lambda: finalize(entries_path, root, proof_path, hash_path, manifest_path, "a" * 64, 1, 1, 5, "gamg", "0.1", "0.2"), "a changed validated report")
        first_path.write_bytes(original)
        extra = root / "raw/caseA/unexpected/solve-report.json"; extra.parent.mkdir(parents=True); extra.write_text("extra")
        expect_failure(lambda: finalize(entries_path, root, proof_path, hash_path, manifest_path, "a" * 64, 1, 1, 5, "gamg", "0.1", "0.2"), "an extra raw report")

mode = sys.argv[1]
if mode == "self-test":
    if len(sys.argv) != 2: raise SystemExit("report proof self-test takes no arguments")
    self_test()
elif mode == "finalize":
    if len(sys.argv) != 14: raise SystemExit("report proof finalize argument count differs")
    finalize(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]), pathlib.Path(sys.argv[5]),
             pathlib.Path(sys.argv[6]), sys.argv[7], int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10]),
             sys.argv[11], sys.argv[12], sys.argv[13])
else:
    raise SystemExit(f"unknown report proof tool mode: {mode}")
PY
}

run_reltol_tool self-test
run_report_proof_tool self-test

[[ "$cpu_set" =~ ^[0-9]+([,-][0-9]+)*$ ]] || fail "invalid CPU set: $cpu_set"
taskset -c "$cpu_set" true >/dev/null 2>&1 || fail "CPU set is not available: $cpu_set"
[[ "$warmup_runs" =~ ^[0-9]+$ ]] && ((warmup_runs >= 2)) || fail "warmup runs must be at least two"
[[ "$measured_runs" =~ ^[1-9][0-9]*$ ]] && ((measured_runs >= 10)) || fail "measured runs must be at least ten"
((measured_runs % 2 == 0)) || fail "measured runs must be even"
[[ "$max_simple_iterations" =~ ^[1-9][0-9]*$ ]] || fail "max SIMPLE iterations must be positive"
[[ "$pressure_solver" == "pcg" || "$pressure_solver" == "gamg" ]] || fail "unsupported pressure solver: $pressure_solver"

python3 - "$candidate_pressure_reltol" "$candidate_momentum_reltol" <<'PY'
import math, sys
for label, raw in zip(("pressure", "momentum"), sys.argv[1:]):
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"candidate {label} relTol is not numeric: {raw}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise SystemExit(f"candidate {label} relTol must be finite and non-negative")
if float(sys.argv[1]) == 0.0 and float(sys.argv[2]) == 0.0:
    raise SystemExit("candidate p and U relTol must not both be zero")
def active(is_gamg, value):
    return value > (0.0 if is_gamg else 1.0e-15)
if active(False, 1.0e-15) or not active(False, math.nextafter(1.0e-15, math.inf)):
    raise SystemExit("non-GAMG relTol activation boundary self-check failed")
if active(True, 0.0) or not active(True, math.nextafter(0.0, math.inf)):
    raise SystemExit("GAMG relTol activation boundary self-check failed")
def exact_positive_integer(value):
    return type(value) is int and value >= 1
if exact_positive_integer(True) or exact_positive_integer(0) or not exact_positive_integer(1):
    raise SystemExit("exact positive-integer contract self-check failed")
def matching_field_shapes(baseline, candidate):
    return (baseline[0] == candidate[0] and baseline[1] == candidate[1] and baseline[2] == candidate[2]
            and baseline[1] == 3 * baseline[0] and baseline[2] == baseline[0]
            and candidate[1] == 3 * candidate[0] and candidate[2] == candidate[0])
if matching_field_shapes((2, 6, 2), (2, 3, 2)) or not matching_field_shapes((2, 6, 2), (2, 6, 2)):
    raise SystemExit("cross-ref field-shape contract self-check failed")
PY

case "$build_variant" in
    portable) build_rustflags=""; build_codegen_units=""; build_lto="" ;;
    native) build_rustflags="-C target-cpu=native"; build_codegen_units=""; build_lto="" ;;
    native-codegen1) build_rustflags="-C target-cpu=native"; build_codegen_units="1"; build_lto="" ;;
    native-thin-lto) build_rustflags="-C target-cpu=native"; build_codegen_units=""; build_lto="thin" ;;
    native-fat-lto) build_rustflags="-C target-cpu=native"; build_codegen_units=""; build_lto="fat" ;;
    *) fail "unsupported build variant: $build_variant" ;;
esac

rustc_version="$(rustc "+$rust_toolchain" --version 2>/dev/null || true)"
[[ "$rustc_version" == "rustc $rust_toolchain "* ]] || fail "exact Rust $rust_toolchain is not installed"
cargo_version="$(cargo "+$rust_toolchain" --version 2>/dev/null || true)"
[[ -n "$cargo_version" ]] || fail "Cargo for Rust $rust_toolchain is not installed"
home_fstype="$(findmnt -T "$HOME" -n -o FSTYPE | tr -d '[:space:]')"
[[ "$home_fstype" == "ext4" ]] || fail "WSL home is not on ext4 (found '$home_fstype')"

if [[ "$mode" == "preflight" ]]; then
    printf 'preflight=pass\nrustc=%s\ncargo=%s\nfilesystem=%s\ncpu_set=%s\nbuild_variant=%s\nreltol_boundary_self_test=pass\ncontract_negative_self_test=pass\nreltol_token_mutation_self_test=pass\nexact_report_proof_self_test=pass\n' \
        "$rustc_version" "$cargo_version" "$home_fstype" "$cpu_set" "$build_variant"
    exit 0
fi

for required_file in "$baseline_archive" "$candidate_archive" "$templates_archive" "$controls_archive" "$manifest_path"; do
    [[ -f "$required_file" ]] || fail "required staged input was not found: $required_file"
done
for required_value in "$baseline_archive_sha256" "$baseline_commit" "$baseline_tree" \
    "$candidate_archive_sha256" "$candidate_commit" "$candidate_tree" \
    "$templates_archive_sha256" "$controls_archive_sha256" "$output_archive"; do
    [[ -n "$required_value" ]] || fail "a required binding value was empty"
done

cache_root="$HOME/.cache/ferrumcfd-linux-tta-ab"
mkdir -p "$cache_root"
exec 9>"$cache_root/benchmark.lock"
flock -n 9 || fail "another Ferrum Linux TTA A/B benchmark is active"
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
        printf 'Ferrum Linux TTA A/B workspace preserved: %s\n' "$workspace" >&2
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
templates_root="$workspace/templates"
controls_root="$export_root/controls"
report_proof_entries_path="$workspace/exact-report-validation.entries.jsonl"
mkdir -p "$raw_root" "$metadata_root" "$templates_root" "$controls_root"
: >"$report_proof_entries_path"
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
validate_archive "$workspace/templates.tar" "TTA templates"
tar --no-same-owner --no-same-permissions -xf "$workspace/templates.tar" -C "$templates_root"
cp -- "$controls_archive" "$workspace/controls.tar"
actual_controls_sha256="$(sha256sum "$workspace/controls.tar" | awk '{print $1}')"
[[ "$actual_controls_sha256" == "$controls_archive_sha256" ]] || fail "controls archive SHA-256 changed while staging"
[[ "$(jq -r '.controls.archiveSha256' "$metadata_root/input-manifest.json")" == "$controls_archive_sha256" ]] || fail "manifest controls archive hash differs"
validate_archive "$workspace/controls.tar" "TTA controls"
tar --no-same-owner --no-same-permissions -xf "$workspace/controls.tar" -C "$controls_root"
mapfile -t expected_control_names < <(jq -r '.controls.files[].name' "$metadata_root/input-manifest.json" | sort)
mapfile -t actual_control_names < <(find "$controls_root" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | sort)
[[ "${expected_control_names[*]}" == "${actual_control_names[*]}" ]] || fail "control inventory differs from manifest"
[[ "$(find "$controls_root" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 0 ]] || fail "control archive contains an unexpected directory"
for control_name in "${expected_control_names[@]}"; do
    [[ "$control_name" =~ ^[A-Za-z0-9._-]+$ ]] || fail "unsafe control filename in manifest: $control_name"
    expected_hash="$(jq -r --arg name "$control_name" '.controls.files[] | select(.name == $name) | .sha256' "$metadata_root/input-manifest.json")"
    [[ "$expected_hash" =~ ^[0-9a-f]{64}$ ]] || fail "invalid control hash in manifest: $control_name"
    [[ "$(sha256sum "$controls_root/$control_name" | awk '{print $1}')" == "$expected_hash" ]] || fail "control hash differs: $control_name"
done

for root in "$workspace/slots/A/source" "$workspace/slots/B/source" "$templates_root" "$raw_root"; do
    [[ "$(findmnt -T "$root" -n -o FSTYPE | tr -d '[:space:]')" == "ext4" ]] || fail "benchmark path is not on ext4: $root"
done

[[ "$(jq -r '.baseline.commit' "$metadata_root/input-manifest.json")" == "$baseline_commit" ]] || fail "manifest baseline commit differs"
[[ "$(jq -r '.candidate.commit' "$metadata_root/input-manifest.json")" == "$candidate_commit" ]] || fail "manifest candidate commit differs"
[[ "$(jq -r '.baseline.tree' "$metadata_root/input-manifest.json")" == "$baseline_tree" ]] || fail "manifest baseline tree differs"
[[ "$(jq -r '.candidate.tree' "$metadata_root/input-manifest.json")" == "$candidate_tree" ]] || fail "manifest candidate tree differs"
[[ "$(jq -r '.pressureSolver' "$metadata_root/input-manifest.json")" == "$pressure_solver" ]] || fail "manifest pressure solver differs"
[[ "$(jq -r '.candidateRelTol.p' "$metadata_root/input-manifest.json")" == "$candidate_pressure_reltol" ]] || fail "manifest candidate p relTol differs"
[[ "$(jq -r '.candidateRelTol.U' "$metadata_root/input-manifest.json")" == "$candidate_momentum_reltol" ]] || fail "manifest candidate U relTol differs"

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
    expected_solution_hash="$(jq -r --arg case "$case_name" '.cases[] | select(.name == $case) | .baselineFvSolutionSha256' "$metadata_root/input-manifest.json")"
    [[ "$(sha256sum "$case_root/system/fvSolution" | awk '{print $1}')" == "$expected_solution_hash" ]] || fail "$case_name fvSolution differs"
done

build_environment=("CARGO_INCREMENTAL=0")
if [[ -n "$build_rustflags" ]]; then build_environment+=("RUSTFLAGS=$build_rustflags"); fi
if [[ -n "$build_codegen_units" ]]; then build_environment+=("CARGO_PROFILE_RELEASE_CODEGEN_UNITS=$build_codegen_units"); fi
if [[ -n "$build_lto" ]]; then build_environment+=("CARGO_PROFILE_RELEASE_LTO=$build_lto"); fi
build_format=$'elapsed_s=%e\nuser_s=%U\nsystem_s=%S\nmax_rss_kb=%M\nexit=%x'

build_slot() {
    local slot="$1" ref_name="$2"
    local source_root="$workspace/slots/$slot/source" target_root="$workspace/slots/$slot/target"
    local build_timing="$metadata_root/build-$ref_name-time.env" build_log="$metadata_root/cargo-build-$ref_name-release.log"
    set +e
    (
        cd "$source_root"
        /usr/bin/time -q -f "$build_format" -o "$build_timing" \
            env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS \
            -u CARGO_PROFILE_RELEASE_CODEGEN_UNITS -u CARGO_PROFILE_RELEASE_LTO -u CARGO_INCREMENTAL \
            "${build_environment[@]}" "CARGO_TARGET_DIR=$target_root" \
            cargo "+$rust_toolchain" build --locked --release -p ferrum-run --bin ferrumRun >"$build_log" 2>&1
    )
    local status=$?
    set -e
    [[ "$status" -eq 0 ]] || fail "$ref_name release build failed; see $build_log"
    [[ -x "$target_root/release/ferrumRun" ]] || fail "$ref_name executable was not produced"
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
printf '%s\n' "$actual_controls_sha256" >"$metadata_root/controls-archive-sha256.txt"
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
    find "$1" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | awk '$0 != "0" && $0 ~ /^[0-9]+([.][0-9]+)?$/ {count++} END {print count+0}'
}

patch_candidate_reltol() {
    local solution_path="$1"
    run_reltol_tool patch "$solution_path" "$candidate_pressure_reltol" "$candidate_momentum_reltol"
}

verify_case_reltol() {
    local solution_path="$1" expected_p="$2" expected_u="$3"
    run_reltol_tool verify "$solution_path" "$expected_p" "$expected_u"
}

canonicalize_report() {
    local report_path="$1" canonical_path="$2" hash_path="$3"
    python3 - "$report_path" "$canonical_path" "$hash_path" <<'PY'
import hashlib, json, math, pathlib, sys
report_path, canonical_path, hash_path = map(pathlib.Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8-sig"))
def canonical(value):
    if isinstance(value, dict):
        return {key: canonical(child) for key, child in value.items() if key != "caseDir" and not key.endswith("Seconds")}
    if isinstance(value, list): return [canonical(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value): raise SystemExit("report contains a non-finite number")
    return value
result = canonical(report)
if "caseDir" not in report or "wallClockSeconds" not in report.get("solve", {}):
    raise SystemExit("report did not expose expected path/timing fields")
payload = (json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
canonical_path.write_bytes(payload)
hash_path.write_text(hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii")
PY
}

write_field_values() {
    local report_path="$1" fields_root="$2" output_path="$3"
    python3 - "$report_path" "$fields_root" "$output_path" <<'PY'
import hashlib, json, math, pathlib, re, sys
report_path, fields_root, output_path = map(pathlib.Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8-sig"))
cells = report.get("mesh", {}).get("cells")
if type(cells) is not int or cells <= 0: raise SystemExit("invalid cell count")
def read(name, kind, components):
    path = fields_root / name; raw = path.read_bytes(); text = raw.decode("utf-8")
    pattern = re.compile(rf"\binternalField\s+nonuniform\s+List<{kind}>\s+(\d+)\s*\((.*?)\)\s*;", re.S)
    matches = list(pattern.finditer(text))
    if len(matches) != 1 or int(matches[0].group(1)) != cells: raise SystemExit(f"{name} internalField shape differs")
    payload = matches[0].group(2)
    if components == 1:
        values = [float(token) for token in payload.split()]
    else:
        vector = re.compile(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)")
        entries = list(vector.finditer(payload)); residue = vector.sub("", payload)
        if residue.strip() or len(entries) != cells: raise SystemExit(f"{name} vector payload malformed")
        values = [float(token) for entry in entries for token in entry.groups()]
    if len(values) != cells * components or not all(math.isfinite(value) for value in values): raise SystemExit(f"{name} values invalid")
    return {"components": components, "textSha256": hashlib.sha256(raw).hexdigest(), "values": values}
result = {"schemaVersion": 1, "cellCount": cells, "U": read("U", "vector", 3), "p": read("p", "scalar", 1)}
output_path.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
PY
}

prepare_case() {
    local ref_name="$1" case_name="$2" working_case="$3"
    mkdir -p "$working_case"
    cp -a "$templates_root/$case_name/." "$working_case/"
    if [[ "$ref_name" == "candidate" ]]; then
        patch_candidate_reltol "$working_case/system/fvSolution"
        verify_case_reltol "$working_case/system/fvSolution" "$candidate_pressure_reltol" "$candidate_momentum_reltol"
    else
        verify_case_reltol "$working_case/system/fvSolution" 0 0
    fi
}

validate_report_contract() {
    local report_path="$1" ref_name="$2" case_name="$3" kind="$4" ordinal="$5"
    python3 - "$report_path" "$ref_name" "$pressure_solver" "$candidate_pressure_reltol" "$candidate_momentum_reltol" "$max_simple_iterations" \
        "$case_name" "$kind" "$ordinal" "$report_proof_entries_path" "$export_root" <<'PY'
import hashlib, json, math, os, pathlib, re, sys
report_path = pathlib.Path(sys.argv[1])
report_bytes = report_path.read_bytes()
ref_name, requested_pressure_solver = sys.argv[2], sys.argv[3]
expected_pressure_reltol = float(sys.argv[4]) if ref_name == "candidate" else 0.0
expected_momentum_reltol = float(sys.argv[5]) if ref_name == "candidate" else 0.0
max_simple_iterations = int(sys.argv[6])
case_name, kind, ordinal = sys.argv[7], sys.argv[8], int(sys.argv[9])
entries_path, export_root = pathlib.Path(sys.argv[10]), pathlib.Path(sys.argv[11])

def strict_json_loads(text):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise SystemExit(f"duplicate report JSON key: {key}")
            result[key] = value
        return result
    def reject_constant(value):
        raise SystemExit(f"non-finite report JSON constant: {value}")
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value): raise SystemExit("report contains a non-finite JSON number")
        if isinstance(value, dict):
            for child in value.values(): finite(child)
        elif isinstance(value, list):
            for child in value: finite(child)
    result = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
    finite(result)
    return result

report = strict_json_loads(report_bytes.decode("utf-8-sig"))

def object_value(parent, key, path):
    if not isinstance(parent, dict) or key not in parent or not isinstance(parent[key], dict):
        raise SystemExit(f"{path}.{key} must be a present object")
    return parent[key]
def array_value(parent, key, path):
    if not isinstance(parent, dict) or key not in parent or not isinstance(parent[key], list):
        raise SystemExit(f"{path}.{key} must be a present array")
    return parent[key]
def bool_value(parent, key, path):
    if not isinstance(parent, dict) or key not in parent or type(parent[key]) is not bool:
        raise SystemExit(f"{path}.{key} must be a present boolean")
    return parent[key]
def int_value(parent, key, path):
    if not isinstance(parent, dict) or key not in parent or type(parent[key]) is not int or parent[key] < 0:
        raise SystemExit(f"{path}.{key} must be a present non-negative integer")
    return parent[key]
def number_value(parent, key, path):
    if not isinstance(parent, dict) or key not in parent or type(parent[key]) not in (int, float):
        raise SystemExit(f"{path}.{key} must be a present numeric scalar")
    value = float(parent[key])
    if not math.isfinite(value): raise SystemExit(f"{path}.{key} must be finite")
    return value
def string_value(parent, key, path):
    if not isinstance(parent, dict) or key not in parent or type(parent[key]) is not str or not parent[key]:
        raise SystemExit(f"{path}.{key} must be a present non-empty string")
    return parent[key]
def array_number(values, index, path):
    if not isinstance(values, list) or index < 0 or index >= len(values) or type(values[index]) not in (int, float):
        raise SystemExit(f"{path} must be a present numeric scalar")
    value = float(values[index])
    if not math.isfinite(value): raise SystemExit(f"{path} must be finite")
    return value
def active(solver, reltol):
    return reltol > (0.0 if solver.casefold() == "gamg" else 1.0e-15)
def validate_one_solve(solve, absolute, reltol, solver, path):
    if not isinstance(solve, dict): raise SystemExit(f"{path} must be an object")
    iterations = int_value(solve, "iterations", path)
    if bool_value(solve, "converged", path) is not True: raise SystemExit(f"{path} is non-converged")
    initial = number_value(solve, "initialNormalizedResidual", path)
    residual = number_value(solve, "residualNorm", path)
    final = number_value(solve, "normalizedResidual", path)
    reported_target = number_value(solve, "effectiveNormalizedTolerance", path)
    if min(initial, residual, final, reported_target) < 0.0: raise SystemExit(f"{path} contains a negative residual/tolerance")
    relative_limit = reltol * initial if active(solver, reltol) else 0.0
    expected_target = max(absolute, relative_limit)
    if reported_target != expected_target: raise SystemExit(f"{path} effective target differs")
    if not final < reported_target: raise SystemExit(f"{path} misses strict final < target")
    expected_reason = "ExactZero" if solver.casefold() == "gamg" and iterations == 0 and final == 0.0 else (
        "RelativeTolerance" if relative_limit > absolute else "AbsoluteTolerance"
    )
    if string_value(solve, "stopReason", path) != expected_reason: raise SystemExit(f"{path} stopReason differs")
    return iterations

if not isinstance(report, dict): raise SystemExit("report root must be an object")
solve = object_value(report, "solve", "$")
outer = object_value(report, "outerConvergence", "$")
linear = object_value(report, "linearSolves", "$")
options = object_value(report, "options", "$")
history = array_value(report, "history", "$")
if not history: raise SystemExit("report history is empty")
if bool_value(solve, "converged", "$.solve") is not True or string_value(solve, "stopReason", "$.solve") != "Converged":
    raise SystemExit("outer SIMPLE solve did not converge exactly")
if string_value(outer, "status", "$.outerConvergence") != "converged" or string_value(outer, "reason", "$.outerConvergence") != "Converged":
    raise SystemExit("outer convergence status/reason differs")
for key in ("configured", "evaluated", "converged"):
    if bool_value(outer, key, "$.outerConvergence") is not True: raise SystemExit(f"outer convergence {key} is not true")
simple_iterations = int_value(solve, "simpleIterations", "$.solve")
solve_momentum_total = int_value(solve, "momentumLinearIterations", "$.solve")
solve_pressure_total = int_value(solve, "pressureLinearIterations", "$.solve")
if simple_iterations != len(history) or simple_iterations > max_simple_iterations:
    raise SystemExit("SIMPLE iteration count differs from history or configured maximum")
if int_value(linear, "momentumComponentNonConvergedSolves", "$.linearSolves") != 0 or int_value(linear, "pressureCorrectionNonConvergedSolves", "$.linearSolves") != 0:
    raise SystemExit("non-converged linear solve counter is nonzero")
momentum_solver = string_value(options, "momentumLinearSolver", "$.options")
pressure_solver = string_value(options, "pressureLinearSolver", "$.options")
if pressure_solver.casefold() != requested_pressure_solver.casefold(): raise SystemExit("pressure solver differs from request")
momentum_absolute = number_value(options, "momentumLinearTolerance", "$.options")
pressure_absolute = number_value(options, "pressureLinearTolerance", "$.options")
if min(momentum_absolute, pressure_absolute) < 0.0: raise SystemExit("linear absolute tolerance is negative")
if ref_name == "candidate":
    if number_value(options, "momentumLinearRelativeTolerance", "$.options") != expected_momentum_reltol:
        raise SystemExit("candidate momentum relTol differs")
    if number_value(options, "pressureLinearRelativeTolerance", "$.options") != expected_pressure_reltol:
        raise SystemExit("candidate pressure relTol differs")
if pressure_solver.casefold() == "gamg":
    pressure_gamg = object_value(options, "pressureGamg", "$.options")
    if number_value(pressure_gamg, "relTol", "$.options.pressureGamg") != expected_pressure_reltol:
        raise SystemExit("GAMG relTol differs")

history_momentum_total = 0
history_pressure_total = 0
for row_index, row in enumerate(history):
    path = f"$.history[{row_index}]"
    if not isinstance(row, dict): raise SystemExit(f"{path} must be an object")
    if bool_value(row, "pressureCorrectionAccepted", path) is not True or bool_value(row, "momentumLinearConverged", path) is not True or bool_value(row, "pressureLinearConverged", path) is not True:
        raise SystemExit(f"{path} contains a reject or non-converged solve")
    momentum_total = int_value(row, "momentumLinearIterations", path)
    pressure_total = int_value(row, "pressureLinearIterations", path)
    pressure_count = int_value(row, "pressureLinearSolves", path)
    if pressure_count < 1: raise SystemExit(f"{path}.pressureLinearSolves must be at least one")
    history_momentum_total += momentum_total
    history_pressure_total += pressure_total
    has_momentum = "momentumComponentLinearSolves" in row
    has_pressure = "pressureCorrectionLinearSolves" in row
    if has_momentum != has_pressure: raise SystemExit(f"{path} telemetry is partially missing")
    if ref_name == "candidate" and not has_momentum: raise SystemExit(f"{path} candidate telemetry is missing")
    if has_momentum:
        momentum = array_value(row, "momentumComponentLinearSolves", path)
        pressure = array_value(row, "pressureCorrectionLinearSolves", path)
        if len(momentum) != 3: raise SystemExit(f"{path} momentum telemetry count differs")
        momentum_sum = 0
        for component_index, component in enumerate(("x", "y", "z")):
            entry_path = f"{path}.momentumComponentLinearSolves[{component_index}]"
            entry = momentum[component_index]
            if not isinstance(entry, dict) or string_value(entry, "component", entry_path) != component:
                raise SystemExit(f"{entry_path} component differs")
            momentum_sum += validate_one_solve(object_value(entry, "solve", entry_path), momentum_absolute, expected_momentum_reltol, momentum_solver, entry_path + ".solve")
        if momentum_sum != momentum_total: raise SystemExit(f"{path} momentum iteration sum differs")
        if len(pressure) != pressure_count: raise SystemExit(f"{path} pressure telemetry count differs")
        pressure_sum = 0
        for correction_index, entry in enumerate(pressure, start=1):
            entry_path = f"{path}.pressureCorrectionLinearSolves[{correction_index - 1}]"
            if not isinstance(entry, dict) or int_value(entry, "correction", entry_path) != correction_index:
                raise SystemExit(f"{entry_path} correction index differs")
            pressure_sum += validate_one_solve(object_value(entry, "solve", entry_path), pressure_absolute, expected_pressure_reltol, pressure_solver, entry_path + ".solve")
        if pressure_sum != pressure_total: raise SystemExit(f"{path} pressure iteration sum differs")
    else:
        initial_values = array_value(row, "momentumComponentInitialResiduals", path)
        final_values = array_value(row, "momentumComponentNormalizedResidualNorms", path)
        if len(initial_values) != 3 or len(final_values) != 3 or pressure_count < 1:
            raise SystemExit(f"{path} legacy baseline residual shape differs")
        for component_index in range(3):
            initial = array_number(initial_values, component_index, f"{path}.momentumComponentInitialResiduals[{component_index}]")
            final = array_number(final_values, component_index, f"{path}.momentumComponentNormalizedResidualNorms[{component_index}]")
            if min(initial, final) < 0.0: raise SystemExit(f"{path} legacy momentum residual is negative")
            relative_limit = expected_momentum_reltol * initial if active(momentum_solver, expected_momentum_reltol) else 0.0
            if not final < max(momentum_absolute, relative_limit):
                raise SystemExit(f"{path} legacy momentum component misses strict target")
        pressure_initial = number_value(row, "pressureCorrectionInitialResidual", path)
        pressure_final = number_value(row, "pressureCorrectionNormalizedResidualNorm", path)
        if min(pressure_initial, pressure_final) < 0.0: raise SystemExit(f"{path} legacy pressure residual is negative")
        pressure_relative_limit = expected_pressure_reltol * pressure_initial if active(pressure_solver, expected_pressure_reltol) else 0.0
        if not pressure_final < max(pressure_absolute, pressure_relative_limit):
            raise SystemExit(f"{path} legacy pressure aggregate misses strict target")
if history_momentum_total != solve_momentum_total or history_pressure_total != solve_pressure_total:
    raise SystemExit("history linear-iteration totals differ from solve totals")

# Registration occurs only after the exact contract above has accepted these exact bytes.
if not re.fullmatch(r"[A-Za-z0-9._-]+", case_name) or case_name in (".", ".."):
    raise SystemExit("validated report case name is unsafe")
if ref_name not in ("baseline", "candidate"):
    raise SystemExit("validated report ref is invalid")
if kind == "oracle":
    if ordinal != 0: raise SystemExit("oracle report ordinal must be zero")
    run_identity = f"oracle-{ref_name}"
elif kind in ("warmup", "measured"):
    if ordinal < 1: raise SystemExit("timed report ordinal must be positive")
    run_identity = f"{kind}-{ordinal}-{ref_name}"
else:
    raise SystemExit("validated report kind is invalid")
expected_relative = f"raw/{case_name}/{run_identity}/solve-report.json"
root = export_root.resolve(strict=True)
if export_root.is_symlink() or not export_root.is_dir() or report_path.is_symlink() or not report_path.is_file():
    raise SystemExit("validated report path is not regular")
resolved = report_path.resolve(strict=True)
if resolved != report_path.absolute(): raise SystemExit("validated report path traverses a symbolic link")
try: relative = resolved.relative_to(root).as_posix()
except ValueError as exc: raise SystemExit("validated report escaped export root") from exc
if relative != expected_relative: raise SystemExit("validated report path differs from run identity")
entry = {"case": case_name, "kind": kind, "ordinal": ordinal, "ref": ref_name,
         "relativePath": relative, "sha256": hashlib.sha256(report_bytes).hexdigest()}
if entries_path.exists():
    if entries_path.is_symlink() or not entries_path.is_file(): raise SystemExit("validated report journal is unsafe")
    for line in entries_path.read_text(encoding="utf-8").splitlines():
        if not line: raise SystemExit("validated report journal contains an empty line")
        if strict_json_loads(line).get("relativePath") == relative: raise SystemExit("validated report path was already registered")
payload = (json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(str(entries_path), flags, 0o600)
try:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0: raise SystemExit("validated report journal write made no progress")
        remaining = remaining[written:]
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
}

run_timed_ref() {
    local ref_name="$1" binary="$2" case_name="$3" kind="$4" ordinal="$5" run_root="$6"
    local working_case="$run_root/case" report_path="$run_root/solve-report.json" timing_path="$run_root/process-time.env"
    prepare_case "$ref_name" "$case_name" "$working_case"
    sha256sum "$working_case/system/fvSolution" | awk '{print $1}' >"$run_root/case-fvSolution.sha256"
    set +e
    (
        cd "$run_root"
        /usr/bin/time -q -f "$time_format" -o "$timing_path" \
            taskset -c "$cpu_set" env "${thread_environment[@]}" \
            "$binary" -solver incompressibleFluid -case "$working_case" \
            --minSimpleIterations 2 --maxSimpleIterations "$max_simple_iterations" \
            --solveReportJson "$report_path" >"$run_root/ferrum.log" 2>&1
    )
    local status=$?
    set -e
    [[ "$status" -eq 0 ]] || fail "$ref_name run failed for $case_name ($kind $ordinal)"
    [[ "$(numeric_output_count "$working_case")" -eq 0 ]] || fail "$ref_name wrote an unexpected time directory"
    [[ ! -e "$run_root/final-fields" ]] || fail "$ref_name timing run wrote final fields"
    validate_report_contract "$report_path" "$ref_name" "$case_name" "$kind" "$ordinal"
    canonicalize_report "$report_path" "$run_root/canonical-report.json" "$run_root/canonical-report.sha256"
}

run_oracle_ref() {
    local ref_name="$1" binary="$2" case_name="$3" run_root="$4"
    local working_case="$run_root/case" report_path="$run_root/solve-report.json" fields_root="$run_root/final-fields"
    prepare_case "$ref_name" "$case_name" "$working_case"
    sha256sum "$working_case/system/fvSolution" | awk '{print $1}' >"$run_root/case-fvSolution.sha256"
    set +e
    (
        cd "$run_root"
        taskset -c "$cpu_set" env "${thread_environment[@]}" \
            "$binary" -solver incompressibleFluid -case "$working_case" \
            --minSimpleIterations 2 --maxSimpleIterations "$max_simple_iterations" \
            --solveReportJson "$report_path" --writeFinalFields final-fields >"$run_root/ferrum.log" 2>&1
    )
    local status=$?
    set -e
    [[ "$status" -eq 0 ]] || fail "$ref_name final-field oracle failed for $case_name"
    [[ "$(numeric_output_count "$working_case")" -eq 0 ]] || fail "$ref_name oracle wrote an unexpected time directory"
    [[ -f "$fields_root/U" && -f "$fields_root/p" ]] || fail "$ref_name oracle did not write U and p"
    [[ "$(find "$fields_root" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | sort | tr '\n' ' ')" == "U p " ]] || fail "$ref_name oracle inventory was not exactly U and p"
    validate_report_contract "$report_path" "$ref_name" "$case_name" oracle 0
    canonicalize_report "$report_path" "$run_root/canonical-report.json" "$run_root/canonical-report.sha256"
    write_field_values "$report_path" "$fields_root" "$run_root/field-values.json"
}

mapfile -t case_rows < <(jq -r '.cases[].name' "$metadata_root/input-manifest.json")
for case_name in "${case_rows[@]}"; do
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
                run_timed_ref "$ref_name" "$binary" "$case_name" "$kind" "$ordinal" "$run_root"
            done
        done
    done
    run_oracle_ref baseline "$baseline_binary" "$case_name" "$raw_root/$case_name/oracle-baseline"
    run_oracle_ref candidate "$candidate_binary" "$case_name" "$raw_root/$case_name/oracle-candidate"
done

run_report_proof_tool finalize "$report_proof_entries_path" "$export_root" \
    "$metadata_root/exact-report-validation.json" "$metadata_root/exact-report-validation.sha256" \
    "$metadata_root/input-manifest.json" "$actual_controls_sha256" "$warmup_runs" "$measured_runs" \
    "$max_simple_iterations" "$pressure_solver" "$candidate_pressure_reltol" "$candidate_momentum_reltol"

archive_on_ext4="$workspace/ferrum-linux-tta-ab-results.tar"
tar -cf "$archive_on_ext4" -C "$export_root" .
archive_sha256="$(sha256sum "$archive_on_ext4" | awk '{print $1}')"
printf '%s\n' "$archive_sha256" >"$workspace/ferrum-linux-tta-ab-results.tar.sha256"
mkdir -p "$(dirname "$output_archive")"
cp -- "$archive_on_ext4" "$output_archive"
cp -- "$workspace/ferrum-linux-tta-ab-results.tar.sha256" "$output_archive.sha256"
completed="1"
printf 'output_archive=%s\noutput_archive_sha256=%s\nworkspace=%s\n' "$output_archive" "$archive_sha256" "$workspace"
