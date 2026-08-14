"""
Photon Coincidence & CHSH Bell-Test Analysis Pipeline
======================================================
Author: Tony E. Ford | QCAUS Lab Analysis Tools

PURPOSE
-------
Analyzes real timestamped single-photon detection data from a coincidence
counting setup (e.g. an SPDC entangled-photon source + polarization
analyzers + single-photon avalanche diodes (SPADs) + a time tagger) and
computes:
  1. Coincidence counts between detector pairs within a timing window.
  2. Accidental coincidence rate (for signal-to-noise / accidental subtraction).
  3. Polarization correlation E(a,b) at each analyzer angle setting.
  4. The CHSH S-parameter, the standard Bell-inequality test statistic.
  5. Statistical significance (standard error, sigma above classical bound).

PHYSICS BACKGROUND
-------------------
The CHSH inequality (Clauser-Horne-Shimony-Holt, 1969) states that for any
local hidden-variable theory:
    S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')| <= 2
Quantum mechanics predicts entangled photon pairs can violate this bound,
up to the Tsirelson bound S <= 2*sqrt(2) ~= 2.828.
A measured S > 2 by a statistically significant margin (typically >5 sigma
in a real experiment) is evidence of quantum entanglement / violation of
local realism -- this is the actual, real test physicists run (Aspect 1982,
Zeilinger et al., and the 2015 loophole-free Bell tests).

E(a,b) is computed from four coincidence counts at each pair of analyzer
angles (a, b):
    E(a,b) = [N++ + N-- - N+- - N-+] / [N++ + N-- + N+- + N-+]
where N++ etc. are coincidence counts for each combination of polarization
outcomes (+/-) at detector settings a and b.

INPUT DATA FORMAT
------------------
Expects a CSV of timestamped detection events, one row per detected photon,
with columns:
    timestamp_ns   : detection time in nanoseconds (monotonic, from the
                      time tagger's internal clock)
    channel        : integer detector channel ID (e.g. 1-4)
This is the standard raw output format for time-to-digital converters
(Swabian Instruments Time Tagger, PicoQuant HydraHarp/MultiHarp, ID
Quantique ID900, etc. all support export to this kind of flat timestamp
list, sometimes as .ttbin/.ptu that must first be converted with the
vendor's own export tool -- see CONVERTING VENDOR FILES below).

A settings/angle log CSV maps time ranges to analyzer angle settings:
    run_id, start_ns, end_ns, angle_a_deg, angle_b_deg

USAGE
-----
    python bell_test_pipeline.py --events events.csv --settings settings.csv \
        --coincidence-window-ns 2.0 --channel-map "1:A+,2:A-,3:B+,4:B-"

Or import as a library and call analyze_run() / compute_chsh() directly.

CONVERTING VENDOR FILES
------------------------
Swabian Time Tagger: use their Python API (`TimeTagger` package) to stream
    or export to CSV directly -- avoids needing a separate conversion step.
PicoQuant .ptu files: use the `readPTU_FLIM` or PicoQuant's official
    `phconvert` / `ptufile` Python packages to export timestamps to CSV.
ID Quantique: their acquisition software exports directly to CSV/TSV.
This pipeline is deliberately hardware-agnostic -- it only needs the plain
timestamp+channel CSV, so it works with any tagger once you've done that
one-time export/conversion step with the vendor's own tool.
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ============================================================================
# DATA LOADING
# ============================================================================

def load_events(path: str) -> pd.DataFrame:
    """Load a timestamped detection-event CSV. Validates required columns."""
    df = pd.read_csv(path)
    required = {"timestamp_ns", "channel"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"events file missing required column(s): {missing}")
    df = df.sort_values("timestamp_ns").reset_index(drop=True)
    if df["timestamp_ns"].isna().any():
        raise ValueError("events file contains NaN timestamps")
    if (df["timestamp_ns"].diff().dropna() < 0).any():
        raise ValueError("timestamps are not monotonic after sorting -- check source data")
    return df


def load_settings(path: str) -> pd.DataFrame:
    """Load the run/angle-setting log CSV."""
    df = pd.read_csv(path)
    required = {"run_id", "start_ns", "end_ns", "angle_a_deg", "angle_b_deg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"settings file missing required column(s): {missing}")
    return df


def parse_channel_map(spec: str) -> dict:
    """
    Parses '1:A+,2:A-,3:B+,4:B-' into {1:'A+', 2:'A-', 3:'B+', 4:'B-'}.
    Channel labels must be exactly one of A+, A-, B+, B- -- one detector
    each for the '+' and '-' polarization outcome on each of the two
    entangled arms (A and B), which is the standard 4-detector CHSH setup.
    """
    mapping = {}
    for part in spec.split(","):
        ch_str, label = part.split(":")
        ch = int(ch_str.strip())
        label = label.strip()
        if label not in ("A+", "A-", "B+", "B-"):
            raise ValueError(f"channel label '{label}' invalid -- must be one of A+, A-, B+, B-")
        mapping[ch] = label
    required_labels = {"A+", "A-", "B+", "B-"}
    if set(mapping.values()) != required_labels:
        raise ValueError(f"channel map must cover exactly {required_labels}, got {set(mapping.values())}")
    return mapping


# ============================================================================
# COINCIDENCE COUNTING
# ============================================================================

@dataclass
class CoincidenceResult:
    window_ns: float
    counts: dict            # {(labelA, labelB): coincidence_count}
    singles: dict            # {label: singles_count}
    accidental_rate: dict     # {(labelA, labelB): estimated accidental coincidences}
    run_duration_s: float


def find_coincidences(times_a: np.ndarray, times_b: np.ndarray, window_ns: float) -> int:
    """
    Counts coincidences between two sorted timestamp arrays (channel A events,
    channel B events) within +/- window_ns of each other. Uses a two-pointer
    sweep -- O(n+m), correct for sorted timestamps, and this is the standard
    algorithm real coincidence-counting software uses (equivalent to what a
    hardware coincidence unit does with a fixed gate window).
    """
    i, j = 0, 0
    n, m = len(times_a), len(times_b)
    count = 0
    while i < n and j < m:
        dt = times_a[i] - times_b[j]
        if abs(dt) <= window_ns:
            count += 1
            # advance both -- assumes at most one legitimate coincidence per
            # photon pair; for high count rates where multiple B events could
            # fall in one A's window, advance the earlier one only (standard
            # practice to avoid double-counting)
            if times_a[i] <= times_b[j]:
                i += 1
            else:
                j += 1
        elif dt < -window_ns:
            i += 1
        else:
            j += 1
    return count


def estimate_accidentals(times_a: np.ndarray, times_b: np.ndarray, window_ns: float,
                          duration_s: float) -> float:
    """
    Standard accidental-coincidence estimate: R_acc = R_a * R_b * (2*window),
    where R_a, R_b are the singles count rates. This is the textbook formula
    used to subtract uncorrelated background coincidences (dark counts,
    unpaired photons) from the raw coincidence count -- see e.g. any
    quantum-optics lab manual on SPDC coincidence measurements.
    """
    rate_a = len(times_a) / duration_s if duration_s > 0 else 0.0
    rate_b = len(times_b) / duration_s if duration_s > 0 else 0.0
    window_s = (2 * window_ns) * 1e-9
    return rate_a * rate_b * window_s * duration_s


def analyze_coincidences(events: pd.DataFrame, channel_map: dict, window_ns: float,
                          start_ns: float = None, end_ns: float = None) -> CoincidenceResult:
    """Runs full pairwise coincidence analysis for one run/angle-setting window."""
    df = events.copy()
    if start_ns is not None:
        df = df[df["timestamp_ns"] >= start_ns]
    if end_ns is not None:
        df = df[df["timestamp_ns"] <= end_ns]
    if len(df) == 0:
        raise ValueError("no events found in the given time range -- check settings file alignment")

    duration_s = (df["timestamp_ns"].max() - df["timestamp_ns"].min()) * 1e-9
    df["label"] = df["channel"].map(channel_map)
    if df["label"].isna().any():
        unmapped = sorted(df.loc[df["label"].isna(), "channel"].unique())
        raise ValueError(f"channel(s) {unmapped} present in data but not in --channel-map")

    times = {lbl: df.loc[df["label"] == lbl, "timestamp_ns"].to_numpy() for lbl in ("A+", "A-", "B+", "B-")}
    singles = {lbl: len(t) for lbl, t in times.items()}

    counts = {}
    accidentals = {}
    for a_lbl in ("A+", "A-"):
        for b_lbl in ("B+", "B-"):
            c = find_coincidences(times[a_lbl], times[b_lbl], window_ns)
            counts[(a_lbl, b_lbl)] = c
            accidentals[(a_lbl, b_lbl)] = estimate_accidentals(times[a_lbl], times[b_lbl], window_ns, duration_s)

    return CoincidenceResult(window_ns=window_ns, counts=counts, singles=singles,
                              accidental_rate=accidentals, run_duration_s=duration_s)


# ============================================================================
# CORRELATION & CHSH
# ============================================================================

def correlation_E(coinc: CoincidenceResult, subtract_accidentals: bool = True) -> tuple:
    """
    Computes the polarization correlation coefficient E(a,b) for one
    analyzer-angle setting pair from the four coincidence counts:
        E = (N++ + N-- - N+- - N-+) / (N++ + N-- + N+- + N-+)
    Returns (E, standard_error, raw_counts_dict).
    Standard error uses Poisson counting statistics: sigma_N = sqrt(N) for
    each count, propagated through the E formula -- the standard treatment
    in coincidence-counting Bell-test analysis.
    """
    def net(key):
        raw = coinc.counts[key]
        acc = coinc.accidental_rate[key] if subtract_accidentals else 0.0
        return max(0.0, raw - acc)

    Npp = net(("A+", "B+"))
    Nmm = net(("A-", "B-"))
    Npm = net(("A+", "B-"))
    Nmp = net(("A-", "B+"))
    total = Npp + Nmm + Npm + Nmp
    if total == 0:
        raise ValueError("zero total coincidences for this setting -- check window size and channel map")

    E = (Npp + Nmm - Npm - Nmp) / total

    # Poisson error propagation on E = (X - Y) / (X + Y) form, X = Npp+Nmm, Y = Npm+Nmp
    X, Y = Npp + Nmm, Npm + Nmp
    sigma_X = np.sqrt(max(X, 1e-9))
    sigma_Y = np.sqrt(max(Y, 1e-9))
    # dE/dX = 2Y/(X+Y)^2 ; dE/dY = -2X/(X+Y)^2
    dEdX = 2 * Y / total**2
    dEdY = -2 * X / total**2
    sigma_E = np.sqrt((dEdX * sigma_X)**2 + (dEdY * sigma_Y)**2)

    return E, sigma_E, {"N++": Npp, "N--": Nmm, "N+-": Npm, "N-+": Nmp, "total": total}


@dataclass
class CHSHResult:
    S: float
    sigma_S: float
    sigma_above_classical: float   # (S - 2) / sigma_S
    E_values: dict                 # {(setting_label): (E, sigma_E)}
    violates_classical_bound: bool
    within_tsirelson_bound: bool


def compute_chsh(E_ab: float, sig_ab: float,
                  E_abp: float, sig_abp: float,
                  E_apb: float, sig_apb: float,
                  E_apbp: float, sig_apbp: float) -> CHSHResult:
    """
    Computes the CHSH S parameter from four correlation measurements at
    four analyzer angle combinations (a,b), (a,b'), (a',b), (a',b'):
        S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
    Classical (local hidden variable) bound: |S| <= 2
    Quantum (Tsirelson) bound:               |S| <= 2*sqrt(2) ~= 2.8284
    """
    S = E_ab - E_abp + E_apb + E_apbp
    sigma_S = np.sqrt(sig_ab**2 + sig_abp**2 + sig_apb**2 + sig_apbp**2)
    sigma_above = (abs(S) - 2.0) / sigma_S if sigma_S > 0 else float("inf")
    return CHSHResult(
        S=S, sigma_S=sigma_S, sigma_above_classical=sigma_above,
        E_values={
            "E(a,b)": (E_ab, sig_ab), "E(a,b')": (E_abp, sig_abp),
            "E(a',b)": (E_apb, sig_apb), "E(a',b')": (E_apbp, sig_apbp),
        },
        violates_classical_bound=(abs(S) > 2.0),
        within_tsirelson_bound=(abs(S) <= 2 * np.sqrt(2) + 1e-9),
    )


# ============================================================================
# FULL RUN ANALYSIS (multi-angle-setting experiment)
# ============================================================================

def analyze_run(events: pd.DataFrame, settings: pd.DataFrame, channel_map: dict,
                 window_ns: float, subtract_accidentals: bool = True) -> dict:
    """
    Full pipeline: for each of the 4 required angle-setting runs
    (a,b), (a,b'), (a',b), (a',b') in the settings file, computes
    coincidences and E(theta), then combines into the CHSH result.
    settings file must contain exactly 4 run_id rows labeled
    'ab', 'abp', 'apb', 'apbp' (case-insensitive) marking which angle
    combination each timing window corresponds to.
    """
    required_runs = {"ab", "abp", "apb", "apbp"}
    settings = settings.copy()
    settings["run_key"] = settings["run_id"].astype(str).str.lower()
    present = set(settings["run_key"])
    missing = required_runs - present
    if missing:
        raise ValueError(f"settings file missing required run(s) {missing} "
                          f"(need exactly: ab, abp, apb, apbp)")

    per_run = {}
    for _, row in settings.iterrows():
        key = row["run_key"]
        if key not in required_runs:
            continue
        coinc = analyze_coincidences(events, channel_map, window_ns,
                                      start_ns=row["start_ns"], end_ns=row["end_ns"])
        E, sigE, raw = correlation_E(coinc, subtract_accidentals=subtract_accidentals)
        per_run[key] = {
            "E": E, "sigma_E": sigE, "raw_counts": raw,
            "angle_a_deg": row["angle_a_deg"], "angle_b_deg": row["angle_b_deg"],
            "singles": coinc.singles, "duration_s": coinc.run_duration_s,
        }

    chsh = compute_chsh(
        per_run["ab"]["E"], per_run["ab"]["sigma_E"],
        per_run["abp"]["E"], per_run["abp"]["sigma_E"],
        per_run["apb"]["E"], per_run["apb"]["sigma_E"],
        per_run["apbp"]["E"], per_run["apbp"]["sigma_E"],
    )
    return {"per_run": per_run, "chsh": chsh}


def print_report(result: dict):
    chsh = result["chsh"]
    print("=" * 70)
    print("CHSH BELL-TEST ANALYSIS REPORT")
    print("=" * 70)
    for key in ("ab", "abp", "apb", "apbp"):
        r = result["per_run"][key]
        print(f"\n[{key}]  a={r['angle_a_deg']}deg  b={r['angle_b_deg']}deg  "
              f"duration={r['duration_s']:.2f}s")
        print(f"    E = {r['E']:+.4f} +/- {r['sigma_E']:.4f}")
        print(f"    raw counts: {r['raw_counts']}")
        print(f"    singles: {r['singles']}")
    print("\n" + "-" * 70)
    print(f"S = {chsh.S:+.4f} +/- {chsh.sigma_S:.4f}")
    print(f"Classical (local realism) bound: |S| <= 2")
    print(f"Quantum (Tsirelson) bound:       |S| <= {2*np.sqrt(2):.4f}")
    if chsh.violates_classical_bound:
        print(f"RESULT: Classical bound VIOLATED by {chsh.sigma_above_classical:.2f} sigma")
        if chsh.sigma_above_classical < 5:
            print("        NOTE: below the conventional 5-sigma significance threshold --")
            print("        longer acquisition / more counts needed for a strong claim.")
    else:
        print("RESULT: Classical bound NOT violated at this statistics level")
    if not chsh.within_tsirelson_bound:
        print("WARNING: S exceeds the Tsirelson bound (2*sqrt(2)) -- this indicates a "
              "systematic error in the setup (timing window, channel mapping, or "
              "accidental subtraction), not a real physics result. No real quantum "
              "system can exceed this bound.")
    print("=" * 70)


# ============================================================================
# SELF-TEST: synthetic data with a known, injected entanglement correlation
# ============================================================================

def _generate_synthetic_test_data(n_pairs_per_run: int = 20000, window_ns: float = 2.0,
                                   seed: int = 42) -> tuple:
    """
    Generates synthetic coincidence data for the four canonical CHSH angles
    (a=0, a'=45, b=22.5, b'=67.5 degrees) that reproduces the quantum
    mechanical prediction E(a,b) = -cos(2*(a-b)) for a maximally entangled
    state, plus realistic Poisson-distributed accidental background. This
    lets the pipeline be verified end-to-end without real hardware -- if
    the recovered S comes out near 2*sqrt(2) as expected, the analysis code
    itself is confirmed correct.
    """
    rng = np.random.default_rng(seed)
    angles = {"ab": (0, 22.5), "abp": (0, 67.5), "apb": (45, 22.5), "apbp": (45, 67.5)}
    events_rows = []
    settings_rows = []
    t_cursor = 0.0
    channel_map = {1: "A+", 2: "A-", 3: "B+", 4: "B-"}
    inv_map = {v: k for k, v in channel_map.items()}

    for run_id, (a, b) in angles.items():
        run_duration_ns = n_pairs_per_run * 1000.0  # spread pairs over the run
        start_ns = t_cursor
        E_true = -np.cos(np.radians(2 * (a - b)))
        p_correlated = (1 + E_true) / 2  # probability outcome matches (both + or both -)
        pair_times = np.sort(rng.uniform(0, run_duration_ns, n_pairs_per_run)) + start_ns
        for pt in pair_times:
            correlated = rng.random() < p_correlated
            a_outcome = rng.choice(["+", "-"])
            b_outcome = a_outcome if correlated else ("-" if a_outcome == "+" else "+")
            jitter_a = rng.normal(0, 0.3)
            jitter_b = rng.normal(0, 0.3)
            events_rows.append((pt + jitter_a, inv_map[f"A{a_outcome}"]))
            events_rows.append((pt + jitter_b, inv_map[f"B{b_outcome}"]))
        # sprinkle in uncorrelated accidental singles (dark counts / unpaired photons)
        n_acc = int(n_pairs_per_run * 0.02)
        for _ in range(n_acc):
            ch = rng.choice([1, 2, 3, 4])
            events_rows.append((start_ns + rng.uniform(0, run_duration_ns), ch))
        end_ns = start_ns + run_duration_ns
        settings_rows.append((run_id, start_ns, end_ns, a, b))
        t_cursor = end_ns + 1e6  # gap between runs

    events_df = pd.DataFrame(events_rows, columns=["timestamp_ns", "channel"]).sort_values("timestamp_ns").reset_index(drop=True)
    settings_df = pd.DataFrame(settings_rows, columns=["run_id", "start_ns", "end_ns", "angle_a_deg", "angle_b_deg"])
    return events_df, settings_df, channel_map


def run_self_test():
    print("Running self-test on synthetic entangled-pair data...")
    events_df, settings_df, channel_map = _generate_synthetic_test_data()
    result = analyze_run(events_df, settings_df, channel_map, window_ns=2.0)
    print_report(result)
    S = result["chsh"].S
    assert 2.0 < abs(S) <= 2 * np.sqrt(2) + 0.05, f"self-test S={S} outside expected range -- pipeline bug"
    assert result["chsh"].sigma_above_classical > 5, "self-test should show a clear >5-sigma violation"
    print("\nSELF-TEST PASSED: recovered S is within the expected quantum range "
          "and shows a clear statistically significant classical-bound violation, "
          "confirming the coincidence-counting and CHSH math is correct.")


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Photon coincidence & CHSH Bell-test analysis pipeline")
    p.add_argument("--events", help="path to timestamped events CSV")
    p.add_argument("--settings", help="path to run/angle settings CSV")
    p.add_argument("--channel-map", help="e.g. '1:A+,2:A-,3:B+,4:B-'")
    p.add_argument("--coincidence-window-ns", type=float, default=2.0,
                    help="coincidence timing window in nanoseconds (typical SPAD setups: 0.5-5 ns)")
    p.add_argument("--no-accidental-subtraction", action="store_true",
                    help="disable accidental coincidence background subtraction")
    p.add_argument("--self-test", action="store_true",
                    help="run on synthetic data to verify the pipeline is working correctly")
    args = p.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not (args.events and args.settings and args.channel_map):
        p.error("--events, --settings, and --channel-map are required unless using --self-test")

    events = load_events(args.events)
    settings = load_settings(args.settings)
    channel_map = parse_channel_map(args.channel_map)
    result = analyze_run(events, settings, channel_map, args.coincidence_window_ns,
                          subtract_accidentals=not args.no_accidental_subtraction)
    print_report(result)


if __name__ == "__main__":
    main()
