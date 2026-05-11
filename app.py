#!/usr/bin/env python3
"""
Streamlit GUI for the Genetic Algorithm Rig Scheduler.

Run locally:
    streamlit run app.py

Imports the engine directly from the parent project folder
(`v1_rig_scheduler_genetic_algorithm.py`), so this file is a thin UI shim.
"""

from __future__ import annotations

import io
import os
import sys
import time
import threading
import contextlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Make the GA engine importable.  The engine lives one folder up from this
# file as `v1_rig_scheduler_genetic_algorithm.py`.
# ---------------------------------------------------------------------------
_ENGINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ENGINE_DIR))

from v1_2_rig_scheduler_genetic_algorithm import (  # noqa: E402
    SimConfig,
    GAConfig,
    GeneticAlgorithmOptimizer,
    load_pads_from_csv,
    load_wells_from_csv,
    assign_wells_to_pads,
    load_base_production,
    load_base_water,
    load_minimum_volumes,
    run_single_simulation,
    build_pvi_greedy_order,
    _evaluate_ordering,
    _add_net_fcf_columns,
    make_diagnostic_plots,
    plot_final_results,
    WATER_OUTLET_CASCADE,
)


# =============================================================================
# Streamlit page
# =============================================================================
st.set_page_config(
    page_title="Rig Scheduler GA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("🧬 Rig Scheduler — Genetic Algorithm")
st.caption(
    "GA over pad orderings; daily simulator enforces crew/capex/water "
    "constraints; fitness = PV(FCF) − PV(water) − PV(shortfall) − PV(PVI delay)."
)


# =============================================================================
# Live stdout capture (so the user can watch generation logs)
# =============================================================================
class LiveStreamCapture(io.StringIO):
    """Pipe stdout into a Streamlit code block in (mostly) real time."""

    def __init__(self, container, max_lines: int = 400):
        super().__init__()
        self._container = container
        self._lines: List[str] = []
        self._max_lines = max_lines
        self._lock = threading.Lock()

    def write(self, s: str):
        super().write(s)
        if not s:
            return len(s)
        with self._lock:
            for line in s.splitlines():
                if line.strip():
                    self._lines.append(line)
            if len(self._lines) > self._max_lines:
                self._lines = self._lines[-self._max_lines :]
            try:
                self._container.code("\n".join(self._lines[-60:]), language="text")
            except Exception:
                pass
        return len(s)

    def flush(self):
        super().flush()


# =============================================================================
# Helpers
# =============================================================================
def _save_uploaded(uploaded, fallback_path: str, tmp_dir: str) -> str:
    """If a file was uploaded, persist it to tmp_dir and return that path.
    Otherwise return the typed-in fallback path (may be empty)."""
    if uploaded is None:
        return fallback_path
    out_path = os.path.join(tmp_dir, uploaded.name)
    with open(out_path, "wb") as f:
        f.write(uploaded.getbuffer())
    return out_path


def _df_download_button(label: str, df: pd.DataFrame, filename: str, key: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv_bytes, file_name=filename, mime="text/csv", key=key)


# =============================================================================
# Sidebar — configuration (organized into collapsible groups)
# =============================================================================
st.sidebar.title("Settings")
st.sidebar.caption("Click any section to expand and edit its settings.")

_NET_BASE = (
    r"\\coterra.com\data\Legacy\Tulsa\Departments\ProductionOperations\GRG"
    r"\Optimization Engineering\Digital Innovation\Operations Research\MBU"
    r"\v2_Rig Scheduler"
)
default_paths = {
    "pad":        rf"{_NET_BASE}\v2_Schedule.csv",
    "well":       rf"{_NET_BASE}\v2_Schedule_decline_curve_parameters_with_water.csv",
    "base_prod":  rf"{_NET_BASE}\v2_base_production.csv",
    "min_vol":    rf"{_NET_BASE}\v1_minimum_volumes.csv",
}

# ----- 📁 Input CSVs --------------------------------------------------------
with st.sidebar.expander("📁 Input CSVs", expanded=False):
    st.caption(
        "Defaults match the production network paths used by the engine. "
        "**Upload** a CSV to override any one, or edit the path text box."
    )
    up_pad   = st.file_uploader("Pad schedule CSV (override)",        type=["csv"], key="up_pad")
    pad_path = st.text_input("…or pad CSV path",                      value=default_paths["pad"])

    up_well   = st.file_uploader("Well decline curves CSV (override)", type=["csv"], key="up_well")
    well_path = st.text_input("…or well CSV path",                     value=default_paths["well"])

    up_base   = st.file_uploader("Base production CSV (override)",     type=["csv"], key="up_base")
    base_path = st.text_input("…or base production path",              value=default_paths["base_prod"])

    up_min   = st.file_uploader("Minimum volumes CSV (override)",      type=["csv"], key="up_min")
    min_path = st.text_input("…or minimum volumes path",               value=default_paths["min_vol"])


# ----- � Output Destination ------------------------------------------------
# Default to the same network share as the inputs, under a `GA_Outputs` subfolder.
# Each run is written into a timestamped subfolder so prior runs are preserved.
default_output_root = rf"{_NET_BASE}\GA_Outputs"
with st.sidebar.expander("📤 Output Destination", expanded=False):
    st.caption(
        "Folder where this run's plots and CSVs will be written. "
        "A timestamped subfolder is created for each run."
    )
    output_root = st.text_input(
        "Output root folder",
        value=default_output_root,
        help=(
            "Default is the production network share. Point this at any local or "
            "network folder (e.g. `C:\\Users\\you\\Downloads\\GA_runs`)."
        ),
    )
    use_timestamp_subfolder = st.checkbox(
        "Create timestamped subfolder per run",
        value=True,
        help="Off = write directly into the folder above (overwrites previous run).",
    )
    if use_timestamp_subfolder:
        st.caption(f"This run will write to: `{output_root}\\run_<timestamp>\\`")
    else:
        st.caption(f"This run will write to: `{output_root}\\`")


# ----- �👷 Crews & Resources -------------------------------------------------
with st.sidebar.expander("👷 Crews & Resources", expanded=False):
    c1, c2 = st.columns(2)
    num_rigs      = c1.number_input("Rigs", 1, 50, 2)
    num_frac      = c2.number_input("Frac crews", 1, 50, 1)
    num_land      = c1.number_input("Land crews", 1, 50, 3)
    num_permit    = c2.number_input("Permit crews", 1, 50, 3)
    num_construct = c1.number_input("Construction crews", 1, 50, 3)


# ----- 💰 Capital & Production Limits ---------------------------------------
with st.sidebar.expander("💰 Capital & Production Limits", expanded=False):
    st.markdown("**Annual capex (base year)**")
    annual_capex     = st.number_input("Annual capex limit ($MM)", 0.0, 5000.0, 350.0, step=25.0)
    capex_tol        = st.number_input("Capex tolerance ($MM)", 0.0, 500.0, 10.0, step=5.0)
    year_1_override  = st.number_input(
        "Year 1 capex limit override ($MM, 0 = use formula)", 0.0, 5000.0, 0.0, step=25.0,
        help="Explicit capex ceiling for year 1 (sim start \u2192 12/31). "
             "0 = use the standard annual limit / CAGR formula. "
             "Useful when pre-approved pads drive a different year-1 capital profile.",
    )
    capex_cagr_pct   = st.number_input("Capex CAGR (% / yr)", -50.0, 50.0, 0.0, step=0.5)
    capex_cagr_years = st.number_input("Capex CAGR years", 0, 50, 0)

    st.markdown("**Overland / Midstream sub-budget**")
    ol_ms_budget     = st.number_input("OL/MS sub-budget ($MM/yr, 0 = no sub-cap)", 0.0, 1000.0, 25.0, step=5.0)
    ol_ms_cagr_pct   = st.number_input("OL/MS CAGR (% / yr)", -50.0, 50.0, 0.0, step=0.5)
    ol_ms_cagr_years = st.number_input("OL/MS CAGR years", 0, 50, 0)

    st.markdown("**Production ceiling**")
    prod_ceiling        = st.number_input(
        "Production ceiling (MCFD avg/yr, 0 = none)", 0.0, 50_000_000.0, 5_000_000.0, step=100_000.0
    )
    production_cagr_pct   = st.number_input("Production CAGR (% / yr)", -50.0, 50.0, 0.0, step=0.5)
    production_cagr_years = st.number_input("Production CAGR years", 0, 50, 0)

    st.markdown("**Capex disbursement (v1.1)**")
    capex_disbursement = st.selectbox(
        "Capex disbursement mode", ["even", "lump_start"], index=0,
        help="`even` spreads each milestone's capex evenly across its duration. "
             "`lump_start` charges the full amount on the milestone start day (v1.0 behavior).",
    )
    reserve_mandatory_capex = st.checkbox(
        "Pre-reserve capex for mandatory pads", value=True,
        help="Back-chain land→permit→pad_con→drill→frac from each mandatory pad's start day "
             "and commit its capex up-front so non-mandatory pads cannot consume it.",
    )
    mandatory_pad_grace_days = st.number_input(
        "Mandatory pad grace days", 0, 365, 0,
        help="Allow each mandatory pad to slip this many days past `mandatory_start_day` "
             "before counting as a violation.",
    )


# ----- 💵 Pricing & Shortfall -----------------------------------------------
with st.sidebar.expander("💵 Pricing & Shortfall", expanded=False):
    price_per_mcf  = st.number_input("Gas price ($/MCF, revenue)", 0.0, 25.0, 3.0, step=0.25)
    shortfall_tol  = st.number_input("Shortfall tolerance (MCFD)", 0.0, 500_000.0, 10_000.0, step=1_000.0)
    gas_repl_price = st.number_input(
        "Gas replacement price ($/MCF, shortfall cost)", 0.0, 25.0, 3.0, step=0.25,
        help="Used in GA fitness: shortfall cost = shortfall_mcfd × days × this price.",
    )


# ----- 🗓️ Simulation Horizon ------------------------------------------------
with st.sidebar.expander("🗓️ Simulation Horizon", expanded=False):
    sim_days        = st.number_input("Simulation days", 365, 7300, 3650, step=365)
    sim_start_date  = st.text_input("Simulation start date", "2026-06-01")


# ----- 💧 Water — Storage & Outlets -----------------------------------------
with st.sidebar.expander("💧 Water — Storage & Outlets", expanded=False):
    water_enabled   = st.checkbox("Water management enabled", value=True)
    rainfall_bwpd   = st.number_input("Field rainfall load (BWPD)", 0.0, 1e6, 400.0, step=50.0)

    st.markdown("**Storage capacity (bbl)**")
    s1, s2 = st.columns(2)
    storage_company_cap     = s1.number_input("Company storage cap", 0.0, 1e8, 200_000.0, step=10_000.0)
    storage_thirdparty_cap  = s2.number_input("3rd-party storage cap", 0.0, 1e8, 75_000.0, step=10_000.0)

    st.markdown("**Per-bbl costs ($/bbl)**")
    water_to_frac_cost      = st.number_input("Water to frac cost", 0.0, 100.0, 2.75, step=0.25)
    storage_company_cost    = st.number_input("Company storage cost", 0.0, 100.0, 4.50, step=0.25)
    storage_thirdparty_cost = st.number_input("3rd-party storage cost", 0.0, 100.0, 13.00, step=0.25)

    st.markdown("**Outlet tier 1 — water sharing**")
    o1, o2 = st.columns(2)
    water_sharing_cap   = o1.number_input("Sharing cap (BWPD)", 0.0, 1e6, 1500.0, step=100.0)
    water_sharing_cost  = o2.number_input("Sharing $/bbl",      0.0, 200.0, 8.00, step=0.25)

    st.markdown("**Outlet tier 2 — Select / rail**")
    r1, r2 = st.columns(2)
    select_rail_cap     = r1.number_input("Rail cap (BWPD)",    0.0, 1e6, 3000.0, step=100.0)
    select_rail_cost    = r2.number_input("Rail $/bbl",         0.0, 200.0, 13.35, step=0.25)

    st.markdown("**Outlet tier 3 — PA SWD**")
    p1, p2 = st.columns(2)
    pa_swd_cap          = p1.number_input("PA SWD cap (BWPD)",  0.0, 1e6, 1000.0, step=100.0)
    pa_swd_cost         = p2.number_input("PA SWD $/bbl",       0.0, 200.0, 18.09, step=0.25)

    st.markdown("**Outlet tier 4 — remainder**")
    rm1, rm2 = st.columns(2)
    remainder_cap       = rm1.number_input("Remainder cap (BWPD)", 0.0, 1e7, 15000.0, step=500.0)
    remainder_cost      = rm2.number_input("Remainder $/bbl",      0.0, 500.0, 100.00, step=1.0)


# ----- 💧 Water — Production Deferral ---------------------------------------
with st.sidebar.expander("💧 Water — Production Deferral", expanded=False):
    defer_enabled = st.checkbox(
        "Enable water-takeaway deferral", value=False,
        help=("If checked, a finished pad is held in WAITING_PRODUCTION on any day "
              "where a same-day predictive water balance forecasts overflow above "
              "the trigger. Mandatory pads are never deferred."),
    )
    defer_max_days = st.number_input(
        "Max deferral days/pad", 0, 365, 45,
        help="Hard cap on cumulative days a single pad can be deferred before being forced online.",
    )
    defer_trigger = st.number_input(
        "Trigger: forecast overflow bbl >", 0.0, 1e7, 0.0, step=100.0,
        help=("Same-day predictive forecast = (existing producers' water + base "
              "water + rainfall + this pad's day-1 water) − frac demand − total "
              "outlet takeaway capacity − remaining storage headroom. If the "
              "forecast exceeds this many bbl, the pad is deferred. 0 = defer on "
              "any predicted overflow."),
    )


# ----- 🧬 GA — Population & Operators ---------------------------------------
with st.sidebar.expander("🧬 GA — Population & Operators", expanded=False):
    pop_size       = st.number_input("Population size", 4, 1000, 60)
    n_gens         = st.number_input("Generations", 1, 1000, 10)
    elitism        = st.number_input("Elitism count", 0, 100, 4)
    tourn_size     = st.number_input("Tournament size", 2, 20, 3)
    crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.85, 0.05)
    swap_rate      = st.slider("Swap mutation rate", 0.0, 1.0, 0.20, 0.05)
    insert_rate    = st.slider("Insert mutation rate", 0.0, 1.0, 0.10, 0.05)
    pvi_warm_frac  = st.slider(
        "PVI warm-start fraction", 0.0, 1.0, 0.20, 0.05,
        help="Fraction of initial population seeded with the PVI-sorted ordering.",
    )


# ----- 🎯 GA — Objective & Penalties ----------------------------------------
with st.sidebar.expander("🎯 GA — Objective & Penalties", expanded=False):
    discount_rate = st.number_input("Discount rate (annual)", 0.0, 0.50, 0.10, step=0.01)
    pvi_penalty   = st.number_input(
        "PVI delay penalty constant", 0.0, 1e10, 100_000_000.0, step=1e7, format="%.0f",
        help="penalty_$ = pad.pvi × this constant × (1 − discount_factor(first_production_day)).",
    )


# ----- 🚦 GA — Hard Constraints ---------------------------------------------
with st.sidebar.expander("🚦 GA — Hard Constraints", expanded=False):
    enf_capex   = st.checkbox("Enforce capex budget",      True)
    enf_ceiling = st.checkbox("Enforce production ceiling", True)
    enf_mand    = st.checkbox("Enforce mandatory pads",     True)
    enf_minvol  = st.checkbox("Enforce minimum volumes",    False)
    enf_water   = st.checkbox("Enforce water takeaway",     True)
    water_tol   = st.number_input(
        "Water unhandled tolerance (bbl)", 0.0, 1e9, 0.0, step=1000.0,
        help="Total bbl of unhandled water allowed across the full simulation.",
    )


# ----- 🛠️ GA — Runtime ------------------------------------------------------
with st.sidebar.expander("🛠️ GA — Runtime", expanded=False):
    random_seed      = st.number_input("Random seed", 0, 1_000_000, 42)
    parallel_workers = st.number_input(
        "Parallel workers (0 = auto, 1 = serial)", 0, 64, 0,
        help="0 = CPU count − 1; 1 = single-threaded; >1 = explicit worker count.",
    )

    st.markdown("**Per-generation pad-schedule CSV**")
    write_per_gen_csv = st.checkbox(
        "Write per-generation pad schedule", value=False,
        help="When on, re-runs the simulator on each captured generation's best ordering "
             "and appends those pad schedules to `pad_schedule_GA.csv`. Only those generations "
             "will be available in the Explore-Generations tab. Off = final solution only (much faster).",
    )
    per_gen_every_n = 1
    if write_per_gen_csv:
        write_every_gen = st.checkbox(
            "Every generation", value=True,
            help="Off = only every Nth generation (set N below).",
        )
        if not write_every_gen:
            per_gen_every_n = int(st.number_input(
                "N (write every Nth generation)", 1, 10_000, 5,
            ))


# =============================================================================
# Run button
# =============================================================================
st.markdown("---")
run_clicked = st.button("▶️ Run Genetic Algorithm", type="primary", use_container_width=True)

if run_clicked:
    # ---- Persist any uploaded CSVs to a temp dir so the engine loaders can
    #      read them off disk (their loaders take a path, not a buffer).
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp_uploads")
    os.makedirs(tmp_dir, exist_ok=True)

    pad_fp  = _save_uploaded(up_pad,  pad_path,  tmp_dir)
    well_fp = _save_uploaded(up_well, well_path, tmp_dir)
    base_fp = _save_uploaded(up_base, base_path, tmp_dir)
    min_fp  = _save_uploaded(up_min,  min_path,  tmp_dir)

    if not pad_fp or not well_fp or not base_fp or not min_fp:
        st.error("Pad, well, base-production and minimum-volumes CSVs are all required.")
        st.stop()

    # ---- Output paths — write into the user-selected output root (Settings tab),
    #      optionally under a per-run timestamped subfolder.
    out_root = (output_root or "").strip()
    if not out_root:
        st.error("Output root folder is empty — set one in the '📤 Output Destination' section.")
        st.stop()
    if use_timestamp_subfolder:
        out_dir = os.path.join(out_root, time.strftime("run_%Y%m%d_%H%M%S"))
    else:
        out_dir = out_root
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        st.error(f"Could not create output folder `{out_dir}`: {e}")
        st.stop()
    st.info(f"Writing outputs to: `{out_dir}`")

    config = SimConfig(
        # Resources
        num_rigs=int(num_rigs),
        num_frac_crews=int(num_frac),
        num_land_crews=int(num_land),
        num_permit_crews=int(num_permit),
        num_construction_crews=int(num_construct),
        # Capital
        annual_capex_limit_mm=float(annual_capex),
        capex_tolerance_mm=float(capex_tol),
        year_1_capex_limit_mm=float(year_1_override) if year_1_override > 0 else None,
        capex_cagr_pct=float(capex_cagr_pct),
        capex_cagr_years=int(capex_cagr_years),
        # OL/MS sub-budget
        overland_midstream_budget_mm=float(ol_ms_budget),
        ol_ms_cagr_pct=float(ol_ms_cagr_pct),
        ol_ms_cagr_years=int(ol_ms_cagr_years),
        # Production ceiling
        production_ceiling_mcfd=float(prod_ceiling),
        production_cagr_pct=float(production_cagr_pct),
        production_cagr_years=int(production_cagr_years),
        # Pricing & shortfall
        commodity_price_per_mcf=float(price_per_mcf),
        shortfall_tolerance_mcfd=float(shortfall_tol),
        # Sim
        simulation_days=int(sim_days),
        simulation_start_date=sim_start_date,
        # v1.1 capex disbursement
        capex_disbursement=str(capex_disbursement),
        reserve_mandatory_capex=bool(reserve_mandatory_capex),
        mandatory_pad_grace_days=int(mandatory_pad_grace_days),
        # Water — storage & outlets
        water_enabled=bool(water_enabled),
        rainfall_bwpd=float(rainfall_bwpd),
        storage_company_capacity_bbl=float(storage_company_cap),
        storage_thirdparty_capacity_bbl=float(storage_thirdparty_cap),
        water_to_frac_cost_per_bbl=float(water_to_frac_cost),
        storage_company_cost_per_bbl=float(storage_company_cost),
        storage_thirdparty_cost_per_bbl=float(storage_thirdparty_cost),
        water_sharing_capacity_bwpd=float(water_sharing_cap),
        water_sharing_cost_per_bbl=float(water_sharing_cost),
        select_rail_capacity_bwpd=float(select_rail_cap),
        select_rail_cost_per_bbl=float(select_rail_cost),
        pa_swd_capacity_bwpd=float(pa_swd_cap),
        pa_swd_cost_per_bbl=float(pa_swd_cost),
        remainder_capacity_bwpd=float(remainder_cap),
        remainder_cost_per_bbl=float(remainder_cost),
        # Water deferral
        water_deferral_enabled=bool(defer_enabled),
        water_deferral_max_days=int(defer_max_days),
        water_deferral_trigger_bbl=float(defer_trigger),
        # Inputs
        pad_filepath=pad_fp,
        well_filepath=well_fp,
        base_production_filepath=base_fp,
        minimum_volume_filepath=min_fp,
        # Outputs
        schedule_output=os.path.join(out_dir, "pad_schedule_GA.csv"),
        capex_output=os.path.join(out_dir, "capex_timeline_GA.csv"),
        well_prod_output=os.path.join(out_dir, "well_production_output_GA.csv"),
        monthly_output=os.path.join(out_dir, "monthly_results_GA.csv"),
        adjustment_output=os.path.join(out_dir, "adjustment_log_GA.csv"),
        water_output=os.path.join(out_dir, "water_mass_balance_GA.csv"),
        plot_output_folder=os.path.join(out_dir, "plots"),
    )
    os.makedirs(config.plot_output_folder, exist_ok=True)

    ga_config = GAConfig(
        population_size=int(pop_size),
        num_generations=int(n_gens),
        elitism_count=int(elitism),
        tournament_size=int(tourn_size),
        crossover_rate=float(crossover_rate),
        swap_mutation_rate=float(swap_rate),
        insert_mutation_rate=float(insert_rate),
        discount_rate_annual=float(discount_rate),
        gas_replacement_price_per_mcf=float(gas_repl_price),
        enforce_capex_budget=enf_capex,
        enforce_production_ceiling=enf_ceiling,
        enforce_mandatory_pads=enf_mand,
        enforce_minimum_volumes=enf_minvol,
        enforce_water_takeaway=enf_water,
        water_unhandled_tolerance_bbl=float(water_tol),
        pvi_delay_penalty_constant=float(pvi_penalty),
        random_seed=int(random_seed),
        parallel_workers=int(parallel_workers),
        pvi_warm_start_fraction=float(pvi_warm_frac),
        verbose=True,
    )

    # ---- Live console
    st.subheader("Run log")
    log_box = st.empty()
    capture = LiveStreamCapture(log_box)

    progress = st.progress(0.0, text="Starting…")
    t_start = time.time()

    try:
        with contextlib.redirect_stdout(capture):
            print("Loading inputs…")
            pads = load_pads_from_csv(config.pad_filepath, config.simulation_start_date)
            wells = load_wells_from_csv(config.well_filepath)
            assign_wells_to_pads(pads, wells)
            base_prod  = load_base_production(config.base_production_filepath, config.simulation_days, config.simulation_start_date)
            min_vols   = load_minimum_volumes(config.minimum_volume_filepath, config.simulation_days, config.simulation_start_date)
            base_water = load_base_water(config.base_production_filepath, config.simulation_days, config.simulation_start_date)
            print(f"Loaded {len(pads)} pads, {sum(len(p.wells) for p in pads)} wells assigned.")

            progress.progress(0.05, text="Inputs loaded — starting GA")

            optimizer = GeneticAlgorithmOptimizer(
                config=config,
                ga_config=ga_config,
                base_production=base_prod,
                minimum_volumes=min_vols,
                template_pads=pads,
                base_water=base_water,
            )
            best_order, best_fb = optimizer.run()

            progress.progress(0.85, text="GA finished — running PVI-greedy benchmark + final sim")

            # PVI-greedy benchmark (informational + needed for plot baselines)
            bench_order = build_pvi_greedy_order(pads)
            bench_fb = _evaluate_ordering(
                pad_order=bench_order, config=config, ga_config=ga_config,
                base_production=base_prod, minimum_volumes=min_vols,
                template_pads=pads, base_water=base_water,
            )
            bench_result = run_single_simulation(
                config=config, base_production=base_prod, minimum_volumes=min_vols,
                pad_order=bench_order, label="PVI_GREEDY", template_pads=pads,
                base_water=base_water,
            )
            bench_monthly = bench_result["monthly"]

            # Final re-sim for reporting
            final_result = run_single_simulation(
                config=config, base_production=base_prod, minimum_volumes=min_vols,
                pad_order=best_order, label="GA_BEST", template_pads=pads,
                base_water=base_water,
            )
            final_sim     = final_result["sim"]
            final_monthly = final_result["monthly"]

            # Augment monthly DataFrames with FCF NET of water + gas-shortfall costs
            final_monthly = _add_net_fcf_columns(final_monthly, config,
                                                  ga_config.gas_replacement_price_per_mcf)
            bench_monthly = _add_net_fcf_columns(bench_monthly, config,
                                                  ga_config.gas_replacement_price_per_mcf)

            progress.progress(0.95, text="Writing CSVs")

            # CSV outputs (replicate the engine's writer behavior for the GUI)
            final_monthly.to_csv(config.monthly_output, index=False)

            # Pad schedule across every generation's best + final.
            # Also capture each generation-best sim for the "Explore Generations" tab.
            sched_rows = []
            per_gen: dict = {}  # generation -> {order, sim, monthly, score}
            def _row(p, label):
                return {
                    "run": label, "pad": p.name, "is_mandatory": p.is_mandatory,
                    "mandatory_start_day": p.mandatory_start_day,
                    "land_start": p.land_start, "land_end": p.land_end,
                    "permit_start": p.permit_start, "permit_end": p.permit_end,
                    "pad_con_start": p.pad_con_start, "pad_con_end": p.pad_con_end,
                    "midstream_start": p.midstream_start, "midstream_end": p.midstream_end,
                    "overland_start": p.overland_start, "overland_end": p.overland_end,
                    "drill_start": p.drill_start, "drill_end": p.drill_end,
                    "frac_start": p.frac_start, "frac_end": p.frac_end,
                    "first_production_day": p.first_production_day, "pvi": p.pvi,
                }
            for h in optimizer.history:
                gen_order = h.get("gen_best_order")
                if not gen_order:
                    continue
                if not write_per_gen_csv:
                    continue
                gen_idx = int(h["generation"])
                if per_gen_every_n > 1 and (gen_idx % per_gen_every_n) != 0:
                    continue
                gen_res = run_single_simulation(
                    config=config, base_production=base_prod, minimum_volumes=min_vols,
                    pad_order=gen_order, label=f"gen_{h['generation']}_best",
                    template_pads=pads, base_water=base_water,
                )
                gen_monthly = _add_net_fcf_columns(
                    gen_res["monthly"], config, ga_config.gas_replacement_price_per_mcf
                )
                per_gen[int(h["generation"])] = {
                    "order":   list(gen_order),
                    "sim":     gen_res["sim"],
                    "monthly": gen_monthly,
                    "score":   float(h.get("gen_best_score", float("nan"))),
                }
                for p in gen_res["sim"].pads:
                    sched_rows.append(_row(p, f"population {h['generation']}"))
            for p in final_sim.pads:
                sched_rows.append(_row(p, "final solution"))
            sched_df = pd.DataFrame(sched_rows)
            sched_df.to_csv(config.schedule_output, index=False)

            # Also write a dates version of the schedule
            day_cols = [
                "mandatory_start_day",
                "land_start", "land_end",
                "permit_start", "permit_end",
                "pad_con_start", "pad_con_end",
                "midstream_start", "midstream_end",
                "overland_start", "overland_end",
                "drill_start", "drill_end",
                "frac_start", "frac_end",
                "first_production_day",
            ]
            sim_start = pd.to_datetime(config.simulation_start_date)
            dates_df = sched_df.copy()
            for col in day_cols:
                if col in dates_df.columns:
                    dates_df[col] = dates_df[col].apply(
                        lambda d: (sim_start + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
                        if pd.notna(d) and d != "" and d is not None else ""
                    )
            base, ext = os.path.splitext(config.schedule_output)
            dates_path = f"{base}_dates{ext}"
            dates_df.to_csv(dates_path, index=False)

            # Daily water mass balance
            n_days = len(final_sim.daily_water_produced_bbl)
            if n_days > 0:
                start = pd.to_datetime(config.simulation_start_date)
                dates = pd.date_range(start, periods=n_days, freq="D")
                base_water_daily = np.zeros(n_days)
                if base_water is not None:
                    base_n = min(n_days, len(base_water))
                    base_water_daily[:base_n] = np.asarray(base_water[:base_n], dtype=float)
                produced_water_daily = np.asarray(final_sim.daily_water_produced_bbl, dtype=float)
                new_pad_water_daily = produced_water_daily - base_water_daily
                water_df = pd.DataFrame({
                    "date":                       dates,
                    "day":                        np.arange(n_days),
                    "produced_water_bbl":         final_sim.daily_water_produced_bbl,
                    "base_water_production_bbl":  base_water_daily,
                    "new_pad_water_production_bbl": new_pad_water_daily,
                    "rainfall_bbl":               final_sim.daily_water_rainfall_bbl,
                    "to_frac_bbl":                final_sim.daily_water_to_frac_bbl,
                    "to_company_storage_bbl":     final_sim.daily_water_to_company_storage_bbl,
                    "to_thirdparty_storage_bbl":  final_sim.daily_water_to_thirdparty_storage_bbl,
                    "water_sharing_bbl":          final_sim.daily_water_sharing_bbl,
                    "select_rail_bbl":            final_sim.daily_water_select_rail_bbl,
                    "pa_swd_bbl":                 final_sim.daily_water_pa_swd_bbl,
                    "remainder_bbl":              final_sim.daily_water_remainder_bbl,
                    "unhandled_bbl":              final_sim.daily_water_unhandled_bbl,
                    "frac_shortfall_bbl":         final_sim.daily_water_frac_shortfall_bbl,
                    "company_storage_level_bbl":  final_sim.daily_company_storage_bbl,
                    "thirdparty_storage_level_bbl": final_sim.daily_thirdparty_storage_bbl,
                    "water_cost_mm":              final_sim.daily_water_cost_mm,
                })
                water_df["supply_bbl"] = (water_df["produced_water_bbl"]
                                           + water_df["rainfall_bbl"])
                water_df["disposed_bbl"] = (water_df["to_frac_bbl"]
                                             + water_df["to_company_storage_bbl"]
                                             + water_df["to_thirdparty_storage_bbl"]
                                             + water_df["water_sharing_bbl"]
                                             + water_df["select_rail_bbl"]
                                             + water_df["pa_swd_bbl"]
                                             + water_df["remainder_bbl"]
                                             + water_df["unhandled_bbl"])
                water_df.to_csv(config.water_output, index=False)
            else:
                water_df = pd.DataFrame()

            history_df = optimizer.history_df()

            # ---- GA convergence CSV
            if not history_df.empty:
                hist_path = os.path.join(config.plot_output_folder, "ga_convergence.csv")
                history_df.to_csv(hist_path, index=False)

            # ---- Per-pad-by-month FCF & production breakdown + per-pad summary
            try:
                from typing import Dict as _Dict
                n_days_sim = config.simulation_days
                n_months = n_days_sim // 30 + 1
                r = ga_config.discount_rate_annual
                price = config.commodity_price_per_mcf
                days_in_month = 30
                df_month = (1.0 + r) ** (-np.arange(n_months) / 12.0)

                outlet_caps = [
                    (getattr(config, cap), getattr(config, cost))
                    for _key, _lab, cap, cost in WATER_OUTLET_CASCADE
                ]
                tot_cap = sum(c for c, _ in outlet_caps)
                avg_disp_cost = (sum(c * p for c, p in outlet_caps) / tot_cap) if tot_cap > 0 else 0.0

                pad_month_rows = []
                for p in final_sim.pads:
                    reserved = getattr(p, "_reserved_starts", None)
                    def _start(default, ms_key):
                        if reserved is not None and ms_key in reserved:
                            return reserved[ms_key]
                        return default
                    capex_by_month: dict = {}
                    milestones = [
                        (_start(p.land_start,      "land"),             p.capex_land_mm,             p.land_owner_agreement_days),
                        (_start(p.permit_start,    "permit"),           p.capex_permit_mm,           p.pad_permit_days),
                        (_start(p.pad_con_start,   "pad_construction"), p.capex_pad_construction_mm, p.pad_construction_days),
                        (_start(p.midstream_start, "midstream"),        p.capex_midstream_mm,        p.midstream_construction_days),
                        (_start(p.overland_start,  "overland"),         p.capex_overland_mm,         p.overland_construction_days),
                        (_start(p.drill_start,     "drill"),            p.capex_drill_mm,            p.drill_days),
                        (_start(p.frac_start,      "frac"),             p.capex_frac_mm,             p.frac_days),
                    ]
                    mode = getattr(config, "capex_disbursement", "even")
                    for s, amt, dur in milestones:
                        if not amt or amt <= 0 or s is None:
                            continue
                        if mode == "lump_start" or dur is None or dur <= 0:
                            mi = int(s) // 30
                            capex_by_month[mi] = capex_by_month.get(mi, 0.0) + float(amt)
                            continue
                        n_days_ms = max(1, int(np.ceil(float(dur))))
                        per_day = float(amt) / n_days_ms
                        written = 0.0
                        for k in range(n_days_ms):
                            d = int(s) + k
                            if k == n_days_ms - 1:
                                today = float(amt) - written
                            else:
                                today = per_day
                                written += per_day
                            mi = d // 30
                            capex_by_month[mi] = capex_by_month.get(mi, 0.0) + today

                    for mi in range(n_months):
                        sample_day = mi * 30 + 15
                        if sample_day >= n_days_sim:
                            break
                        prod_mcfd = p.production_at_day(sample_day)
                        water_bwpd = p.water_production_at_day(sample_day)
                        revenue_mm = prod_mcfd * days_in_month * price / 1e6
                        is_producing = (p.first_production_day is not None
                                        and sample_day >= p.first_production_day)
                        opex_mm = (p.annual_opex_mm / 12.0) if is_producing else 0.0
                        capex_mm = capex_by_month.get(mi, 0.0)
                        water_cost_mm = water_bwpd * days_in_month * avg_disp_cost / 1e6
                        fcf_mm = revenue_mm - opex_mm - capex_mm - water_cost_mm
                        fcf_pv_mm = fcf_mm * df_month[mi]
                        pad_month_rows.append({
                            "pad": p.name,
                            "drill_position": p.sequence_number,
                            "pvi": p.pvi,
                            "npv_mm_input": p.npv_mm,
                            "is_mandatory": p.is_mandatory,
                            "first_production_day": p.first_production_day,
                            "first_production_month": (p.first_production_day // 30
                                                       if p.first_production_day is not None else None),
                            "month": mi,
                            "year": mi // 12,
                            "production_mcfd": round(prod_mcfd, 1),
                            "water_bwpd": round(water_bwpd, 1),
                            "revenue_mm": round(revenue_mm, 4),
                            "opex_mm": round(opex_mm, 4),
                            "capex_mm": round(capex_mm, 4),
                            "water_cost_mm": round(water_cost_mm, 4),
                            "fcf_mm": round(fcf_mm, 4),
                            "fcf_pv_mm": round(fcf_pv_mm, 4),
                        })

                pad_month_df = pd.DataFrame(pad_month_rows)
                pm_path = os.path.join(config.plot_output_folder, "pad_month_fcf_production.csv")
                pad_month_df.to_csv(pm_path, index=False)

                pad_summary = (pad_month_df.groupby(
                    ["pad", "drill_position", "pvi", "npv_mm_input",
                     "is_mandatory", "first_production_day", "first_production_month"])
                    .agg(
                        total_production_mcf=("production_mcfd",
                                              lambda s: float(s.sum() * days_in_month)),
                        peak_mcfd=("production_mcfd", "max"),
                        total_revenue_mm=("revenue_mm", "sum"),
                        total_opex_mm=("opex_mm", "sum"),
                        total_capex_mm=("capex_mm", "sum"),
                        total_water_cost_mm=("water_cost_mm", "sum"),
                        total_fcf_mm=("fcf_mm", "sum"),
                        pv_fcf_mm=("fcf_pv_mm", "sum"),
                    )
                    .reset_index()
                    .sort_values("drill_position"))
                pad_summary["pv_drag_mm"] = pad_summary["total_fcf_mm"] - pad_summary["pv_fcf_mm"]
                pad_summary["pv_fcf_per_pvi"] = pad_summary["pv_fcf_mm"] / pad_summary["pvi"].replace(0, np.nan)
                ps_path = os.path.join(config.plot_output_folder, "pad_summary_pvi_vs_fcf.csv")
                pad_summary.to_csv(ps_path, index=False)
            except Exception as ex:  # noqa: BLE001
                print(f"  (per-pad FCF CSV error: {ex})")

            # ---- Render the full engine plot dashboard into plot_output_folder
            try:
                make_diagnostic_plots(
                    optimizer=optimizer,
                    best_fb=best_fb,
                    best_order=best_order,
                    final_sim=final_sim,
                    final_monthly=final_monthly,
                    hist=history_df,
                    config=config,
                    ga_config=ga_config,
                    baseline_monthly=bench_monthly,
                    baseline_label="PVI-rank baseline",
                )
            except Exception as ex:  # noqa: BLE001
                print(f"  (plotting error: {ex})")

            try:
                plot_final_results(
                    final_monthly, final_sim, config,
                    optimizer_used="GA",
                    plot_folder=config.plot_output_folder,
                    baseline_results=bench_monthly,
                    baseline_sim=bench_result["sim"],
                    baseline_label="PVI-rank baseline",
                )
            except Exception as ex:  # noqa: BLE001
                print(f"  (skipped final_results plot: {ex})")

            progress.progress(1.0, text="Done")
    except Exception as exc:
        progress.empty()
        st.exception(exc)
        st.stop()

    elapsed = time.time() - t_start

    # Stash everything to session_state so the explorer tab survives Streamlit
    # widget reruns without re-running the GA.
    st.session_state["last_run"] = {
        "out_dir":        out_dir,
        "config":         config,
        "ga_config":      ga_config,
        "best_order":     best_order,
        "best_fb":        best_fb,
        "bench_order":    bench_order,
        "bench_fb":       bench_fb,
        "final_sim":      final_sim,
        "final_monthly":  final_monthly,
        "bench_sim":      bench_result["sim"],
        "bench_monthly":  bench_monthly,
        "history_df":     history_df,
        "per_gen":        per_gen,
        "elapsed":        elapsed,
        "base_water":     base_water,
    }


# =============================================================================
# Results — rendered from session_state so widgets don't re-run the GA
# =============================================================================
def _render_capex_per_year(sim, label_prefix=""):
    cap = getattr(sim, "annual_capex_spent", {}) or {}
    if not cap:
        st.caption("(no capex data)")
        return
    df = (
        pd.DataFrame({"year": list(cap.keys()), "capex_mm": list(cap.values())})
        .sort_values("year")
        .reset_index(drop=True)
    )
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={"capex_mm": st.column_config.NumberColumn(format="%.2f")})
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(df["year"].astype(str), df["capex_mm"], color="#2b6cb0")
    ax.set_xlabel("Year"); ax.set_ylabel("$MM")
    ax.set_title(f"{label_prefix}Capex per year".strip())
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig, clear_figure=True)


def _render_pvi_vs_order(order, sim, label_prefix=""):
    pad_lookup = {p.name: p for p in sim.pads}
    pos = np.arange(1, len(order) + 1)
    pvi = [pad_lookup[n].pvi if n in pad_lookup else np.nan for n in order]
    mand = [pad_lookup[n].is_mandatory if n in pad_lookup else False for n in order]
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#d62728" if m else "#1f77b4" for m in mand]
    ax.scatter(pos, pvi, c=colors, s=40)
    for i, n in enumerate(order):
        ax.annotate(n, (pos[i], pvi[i]), fontsize=6, alpha=0.6,
                    xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Drill order"); ax.set_ylabel("PVI")
    ax.set_title(f"{label_prefix}PVI vs drill order (red = mandatory)".strip())
    ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)


def _build_water_df(sim, sim_start_date, base_water=None):
    n = len(sim.daily_water_produced_bbl)
    if n == 0:
        return pd.DataFrame()
    start = pd.to_datetime(sim_start_date)
    base_water_daily = np.zeros(n)
    if base_water is not None:
        base_n = min(n, len(base_water))
        base_water_daily[:base_n] = np.asarray(base_water[:base_n], dtype=float)
    produced_water_daily = np.asarray(sim.daily_water_produced_bbl, dtype=float)
    new_pad_water_daily = produced_water_daily - base_water_daily
    wdf = pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="D"),
        "day": np.arange(n),
        "produced_water_bbl":         sim.daily_water_produced_bbl,
        "base_water_production_bbl":  base_water_daily,
        "new_pad_water_production_bbl": new_pad_water_daily,
        "rainfall_bbl":               sim.daily_water_rainfall_bbl,
        "to_frac_bbl":                sim.daily_water_to_frac_bbl,
        "to_company_storage_bbl":     sim.daily_water_to_company_storage_bbl,
        "to_thirdparty_storage_bbl":  sim.daily_water_to_thirdparty_storage_bbl,
        "water_sharing_bbl":          sim.daily_water_sharing_bbl,
        "select_rail_bbl":            sim.daily_water_select_rail_bbl,
        "pa_swd_bbl":                 sim.daily_water_pa_swd_bbl,
        "remainder_bbl":              sim.daily_water_remainder_bbl,
        "unhandled_bbl":              sim.daily_water_unhandled_bbl,
        "frac_shortfall_bbl":         sim.daily_water_frac_shortfall_bbl,
        "company_storage_level_bbl":  sim.daily_company_storage_bbl,
        "thirdparty_storage_level_bbl": sim.daily_thirdparty_storage_bbl,
        "water_cost_mm":              sim.daily_water_cost_mm,
    })
    wdf["supply_bbl"] = wdf["produced_water_bbl"] + wdf["rainfall_bbl"]
    wdf["disposed_bbl"] = (wdf["to_frac_bbl"]
                            + wdf["to_company_storage_bbl"]
                            + wdf["to_thirdparty_storage_bbl"]
                            + wdf["water_sharing_bbl"]
                            + wdf["select_rail_bbl"]
                            + wdf["pa_swd_bbl"]
                            + wdf["remainder_bbl"]
                            + wdf["unhandled_bbl"])
    return wdf


def _render_water_balance(sim, sim_start_date, label_prefix="", base_water=None):
    wdf = _build_water_df(sim, sim_start_date, base_water=base_water)
    if wdf.empty:
        st.caption("(no water data)")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wdf["date"], wdf["produced_water_bbl"], lw=1.0, label="Produced")
    ax.plot(wdf["date"], wdf["to_frac_bbl"], lw=1.0, label="To frac")
    ax.plot(wdf["date"], wdf["unhandled_bbl"], lw=1.0, color="red", label="Unhandled")
    ax.set_xlabel("Date"); ax.set_ylabel("bbl/day")
    ax.set_title(f"{label_prefix}Daily water flows".strip())
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(wdf["date"], wdf["company_storage_level_bbl"], label="Company storage")
    ax2.plot(wdf["date"], wdf["thirdparty_storage_level_bbl"], label="3rd-party storage")
    ax2.set_xlabel("Date"); ax2.set_ylabel("bbl on hand")
    ax2.set_title(f"{label_prefix}Storage tank levels".strip())
    ax2.legend(); ax2.grid(alpha=0.3)
    st.pyplot(fig2, clear_figure=True)

    with st.expander("Daily water balance — full table"):
        st.dataframe(wdf, use_container_width=True, hide_index=True)


def _render_monthly_fcf(monthly_df, label_prefix=""):
    if monthly_df is None or monthly_df.empty:
        return
    fcf_col = "monthly_fcf_net_mm" if "monthly_fcf_net_mm" in monthly_df.columns else "monthly_fcf_mm"
    if fcf_col not in monthly_df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    x = monthly_df["month_index"] if "month_index" in monthly_df.columns else np.arange(len(monthly_df))
    ax.bar(x, monthly_df[fcf_col], color="#2ca02c")
    ax.set_xlabel("Month"); ax.set_ylabel("$MM")
    ax.set_title(f"{label_prefix}Monthly FCF ({fcf_col})".strip())
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig, clear_figure=True)


def _render_cumulative_fcf(monthly_df, label_prefix="", baseline_df=None, baseline_label="PVI baseline"):
    """Cumulative FCF line chart — net of capital, opex, water cost, and
    gas-shortfall penalty.  PVI delay penalty is excluded (fitness-only)."""
    if monthly_df is None or monthly_df.empty:
        return
    # Prefer the NET column (includes water + shortfall deductions)
    cum_col = ("cumulative_fcf_net_mm" if "cumulative_fcf_net_mm" in monthly_df.columns
               else ("cumulative_fcf_mm" if "cumulative_fcf_mm" in monthly_df.columns else None))
    if cum_col is None:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    x = monthly_df["month"] if "month" in monthly_df.columns else np.arange(len(monthly_df))
    ax.plot(x, monthly_df[cum_col], "g-", lw=2.5, label="GA best")
    if baseline_df is not None and cum_col in baseline_df.columns:
        xb = baseline_df["month"] if "month" in baseline_df.columns else np.arange(len(baseline_df))
        ax.plot(xb, baseline_df[cum_col], color="purple", ls="--", lw=1.8, label=baseline_label)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Month"); ax.set_ylabel("Cumulative FCF ($MM)")
    net_tag = " (net of capex, opex, water, shortfall)" if cum_col == "cumulative_fcf_net_mm" else ""
    ax.set_title(f"{label_prefix}Cumulative FCF{net_tag}".strip(), fontsize=11)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)


def _render_pad_order_table(order, sim):
    pad_lookup = {p.name: p for p in sim.pads}
    df = pd.DataFrame({
        "drill_position": np.arange(1, len(order) + 1),
        "pad": order,
        "pvi":      [pad_lookup[n].pvi if n in pad_lookup else np.nan for n in order],
        "mandatory":[pad_lookup[n].is_mandatory if n in pad_lookup else False for n in order],
        "first_production_day": [
            pad_lookup[n].first_production_day if n in pad_lookup else None for n in order
        ],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)


if "last_run" in st.session_state:
    state = st.session_state["last_run"]
    config        = state["config"]
    best_order    = state["best_order"]
    best_fb       = state["best_fb"]
    bench_fb      = state["bench_fb"]
    bench_order   = state.get("bench_order", [])
    bench_sim     = state.get("bench_sim")
    bench_monthly = state.get("bench_monthly")
    final_sim     = state["final_sim"]
    final_monthly = state["final_monthly"]
    history_df    = state["history_df"]
    per_gen       = state["per_gen"]
    out_dir       = state["out_dir"]
    elapsed       = state["elapsed"]
    base_water    = state.get("base_water")

    st.success(f"GA finished in {elapsed:,.1f}s — output folder: `{out_dir}`")

    # NOTE: Don't use st.tabs() here — it resets to the first tab on every script
    # rerun (which Streamlit triggers whenever any widget value changes), so any
    # selection on the Explore tab would bounce the user back to the main view.
    # A st.radio with a stable `key` persists in session_state and survives reruns.
    TAB_MAIN    = "📈 Best / Final solution"
    TAB_EXPLORE = "🔬 Explore generations"
    active_tab = st.radio(
        "View",
        [TAB_MAIN, TAB_EXPLORE],
        horizontal=True,
        label_visibility="collapsed",
        key="active_results_tab",
    )

    # -------------------------------------------------------------------------
    # TAB 1 — Best / Final solution (the main view)
    # -------------------------------------------------------------------------
    if active_tab == TAB_MAIN:
        feas_tag = "✅ feasible" if best_fb.feasible else "❌ infeasible"
        bench_tag = "✅ feasible" if bench_fb.feasible else "❌ infeasible"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("GA NPV score ($MM)",
                  f"{best_fb.score:,.1f}" if best_fb.feasible else "−∞",
                  help=feas_tag)
        m2.metric("PV FCF ($MM)",         f"{best_fb.pv_fcf_mm:,.1f}")
        m3.metric("PV Water cost ($MM)",  f"{best_fb.pv_water_cost_mm:,.2f}")
        m4.metric("PV Shortfall ($MM)",   f"{best_fb.pv_shortfall_cost_mm:,.2f}")

        if not best_fb.feasible:
            st.error(f"GA solution infeasible: {best_fb.invalid_reason}")

        st.subheader("GA vs PVI-greedy benchmark")
        cmp_df = pd.DataFrame({
            "Metric": [
                "NPV score ($MM)", "PV FCF ($MM)", "PV Water ($MM)",
                "PV Shortfall ($MM)", "PV PVI delay ($MM)", "Total FCF undisc ($MM)",
                "Capex overage ($MM)", "Mandatory violations", "Water unhandled (Mbbl)",
                "Mandatory capex overage ($MM)", "Pads ending blocked",
                "Feasible",
            ],
            "GA": [
                best_fb.score, best_fb.pv_fcf_mm, best_fb.pv_water_cost_mm,
                best_fb.pv_shortfall_cost_mm, best_fb.pv_pvi_delay_penalty_mm,
                best_fb.total_fcf_mm, best_fb.capex_overage_mm,
                best_fb.mandatory_violations, best_fb.water_unhandled_total_bbl / 1e3,
                best_fb.mandatory_capex_overage_mm, best_fb.pads_capex_blocked,
                feas_tag,
            ],
            "PVI-greedy": [
                bench_fb.score, bench_fb.pv_fcf_mm, bench_fb.pv_water_cost_mm,
                bench_fb.pv_shortfall_cost_mm, bench_fb.pv_pvi_delay_penalty_mm,
                bench_fb.total_fcf_mm, bench_fb.capex_overage_mm,
                bench_fb.mandatory_violations, bench_fb.water_unhandled_total_bbl / 1e3,
                bench_fb.mandatory_capex_overage_mm, bench_fb.pads_capex_blocked,
                bench_tag,
            ],
        })
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        if not history_df.empty:
            st.subheader("GA convergence")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(history_df["generation"], history_df["best_score"],
                    "g-", lw=2.5, label="Best so far")
            ax.plot(history_df["generation"], history_df["gen_best_score"],
                    "b--", lw=1.0, alpha=0.7, label="Gen best")
            if "gen_mean_score_feas" in history_df.columns:
                ax.plot(history_df["generation"], history_df["gen_mean_score_feas"],
                        "k:", lw=1.0, alpha=0.7, label="Gen mean (feasible)")
            ax.set_xlabel("Generation"); ax.set_ylabel("NPV score ($MM)")
            ax.grid(alpha=0.3); ax.legend()
            st.pyplot(fig, clear_figure=True)

        st.subheader("Cumulative FCF")
        _render_cumulative_fcf(final_monthly, baseline_df=bench_monthly,
                               baseline_label="PVI-rank baseline")

        st.subheader("Best pad ordering")
        _render_pad_order_table(best_order, final_sim)

        st.subheader("Downloads")
        sched_base, sched_ext = os.path.splitext(config.schedule_output)
        sched_dates_path = f"{sched_base}_dates{sched_ext}"
        files = [
            ("Pad schedule (all generations)", config.schedule_output),
            ("Pad schedule – dates (all generations)", sched_dates_path),
            ("Monthly results",                config.monthly_output),
            ("Water mass balance",             config.water_output),
            ("GA convergence history",         os.path.join(config.plot_output_folder, "ga_convergence.csv")),
            ("Per-pad monthly FCF/production", os.path.join(config.plot_output_folder, "pad_month_fcf_production.csv")),
            ("Per-pad summary (PVI vs FCF)",   os.path.join(config.plot_output_folder, "pad_summary_pvi_vs_fcf.csv")),
        ]
        for label, path in files:
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    st.download_button(
                        label, f.read(),
                        file_name=os.path.basename(path),
                        mime="text/csv",
                        key=f"dl_{label}",
                    )

        plots_dir = config.plot_output_folder
        if plots_dir and os.path.isdir(plots_dir):
            png_files = sorted(p for p in os.listdir(plots_dir) if p.lower().endswith(".png"))
            if png_files:
                st.subheader("Diagnostic plots")
                for p in png_files:
                    st.image(os.path.join(plots_dir, p), caption=p, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2 — Explore individual generations (best parent of each generation)
    # -------------------------------------------------------------------------
    if active_tab == TAB_EXPLORE:
        st.markdown(
            "Pick a scenario to inspect. **PVI baseline** is the rank-ordered "
            "PVI-greedy schedule used as the GA's benchmark. **Gen N** scenarios "
            "are each generation's best parent (highest-scoring chromosome from "
            "that generation's population). The detailed view re-uses the same "
            "simulator the main run uses."
        )

        # Build the scenario menu: PVI baseline (if available) + every captured generation.
        PVI_LABEL = "PVI baseline (rank-ordered)"
        gens = sorted(per_gen.keys()) if per_gen else []
        scenario_options: list[str] = []
        if bench_sim is not None and bench_order:
            scenario_options.append(PVI_LABEL)
        scenario_options.extend(f"Gen {g}" for g in gens)

        if not scenario_options:
            st.info("No PVI baseline or per-generation data captured for this run.")
        else:
            # Quick summary table of every generation's best (unchanged)
            if gens:
                summary_rows = []
                for g in gens:
                    rec = per_gen[g]
                    fcf_total = float(np.nansum(rec["monthly"]["monthly_fcf_mm"])) \
                        if "monthly_fcf_mm" in rec["monthly"].columns else float("nan")
                    cap = getattr(rec["sim"], "annual_capex_spent", {}) or {}
                    summary_rows.append({
                        "generation": g,
                        "gen_best_score_mm": rec["score"],
                        "total_fcf_mm": fcf_total,
                        "total_capex_mm": float(sum(cap.values())) if cap else 0.0,
                    })
                summary_df = pd.DataFrame(summary_rows)
                with st.expander("All generations — summary", expanded=False):
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

            sel_scenario = st.selectbox(
                "Scenario",
                scenario_options,
                index=len(scenario_options) - 1,
                help="Choose the PVI baseline or any captured generation's best parent.",
                key="explore_scenario_select",
            )

            # Resolve the selected scenario into (label, order, sim, monthly, score)
            if sel_scenario == PVI_LABEL:
                scenario_label = "PVI baseline"
                order_g   = list(bench_order)
                sim_g     = bench_sim
                monthly_g = bench_monthly
                score_g   = float(bench_fb.score) if bench_fb is not None else float("nan")
                key_tag   = "pvi"
            else:
                # "Gen N" -> int N
                sel_gen = int(sel_scenario.split()[-1])
                rec = per_gen[sel_gen]
                scenario_label = f"Gen {sel_gen} best"
                order_g   = rec["order"]
                sim_g     = rec["sim"]
                monthly_g = rec["monthly"]
                score_g   = rec["score"]
                key_tag   = f"gen{sel_gen}"

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Scenario", scenario_label)
            mc2.metric("Score ($MM)",
                       f"{score_g:,.1f}" if np.isfinite(score_g) else "−∞")
            cap = getattr(sim_g, "annual_capex_spent", {}) or {}
            mc3.metric("Total capex ($MM)", f"{sum(cap.values()):,.1f}" if cap else "0.0")

            st.subheader(f"Pad ordering — {scenario_label}")
            _render_pad_order_table(order_g, sim_g)

            st.subheader("PVI vs drill order")
            _render_pvi_vs_order(order_g, sim_g, label_prefix=f"{scenario_label} — ")

            st.subheader("Capex per year")
            _render_capex_per_year(sim_g, label_prefix=f"{scenario_label} — ")

            st.subheader("Monthly FCF")
            _render_monthly_fcf(monthly_g, label_prefix=f"{scenario_label} — ")

            st.subheader("Cumulative FCF")
            _render_cumulative_fcf(monthly_g, label_prefix=f"{scenario_label} — ")

            st.subheader("Water mass balance")
            _render_water_balance(sim_g, config.simulation_start_date,
                                  label_prefix=f"{scenario_label} — ",
                                  base_water=base_water)

            # CSV downloads for this scenario
            st.subheader(f"Downloads ({scenario_label})")
            order_df = pd.DataFrame({
                "drill_position": np.arange(1, len(order_g) + 1), "pad": order_g,
            })
            _df_download_button(
                f"Pad order — {scenario_label}", order_df,
                f"pad_order_{key_tag}.csv", key=f"dl_order_{key_tag}",
            )
            wdf = _build_water_df(sim_g, config.simulation_start_date, base_water=base_water)
            if not wdf.empty:
                _df_download_button(
                    f"Water mass balance — {scenario_label}", wdf,
                    f"water_mass_balance_{key_tag}.csv", key=f"dl_water_{key_tag}",
                )
            if monthly_g is not None and not monthly_g.empty:
                _df_download_button(
                    f"Monthly results — {scenario_label}", monthly_g,
                    f"monthly_results_{key_tag}.csv", key=f"dl_monthly_{key_tag}",
                )

elif not run_clicked:
    st.info("Configure inputs in the sidebar and click **Run Genetic Algorithm**.")
