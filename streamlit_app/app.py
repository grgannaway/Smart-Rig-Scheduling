#!/usr/bin/env python3
"""
Streamlit GUI for Rig Scheduler CP-SAT Optimizer
Run locally:  streamlit run app.py
"""

import os
import io
import sys
import time
import tempfile
import contextlib
import threading
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy


class LiveStreamCapture(io.StringIO):
    """Captures stdout and pushes live updates to a Streamlit container."""

    def __init__(self, st_container, max_lines=200):
        super().__init__()
        self._container = st_container
        self._lines = []
        self._max_lines = max_lines
        self._lock = threading.Lock()

    def write(self, s):
        super().write(s)
        if s.strip():
            with self._lock:
                for line in s.splitlines():
                    if line.strip():
                        self._lines.append(line)
                        if len(self._lines) > self._max_lines:
                            self._lines = self._lines[-self._max_lines:]
                # Show last 40 lines in the live console
                display = "\n".join(self._lines[-40:])
                try:
                    self._container.code(display, language="text")
                except Exception:
                    pass

    def flush(self):
        super().flush()

# ---------------------------------------------------------------------------
# Import the engine (same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rig_scheduler_engine import (
    SimConfig,
    load_pads_from_csv,
    load_wells_from_csv,
    assign_wells_to_pads,
    load_base_production,
    load_minimum_volumes,
    load_global_overwrites,
    run_with_auto_adjustment,
    run_single_simulation,
    plot_final_results,
    plot_strategy_comparison,
    plot_cpsat_diagnostics,
    plot_cpsat_solution_comparison,
    plot_iteration_overlay,
    print_strategy_comparison,
    print_cpsat_diagnostics_report,
    OrderedDrillingSimulator,
    ORTOOLS_AVAILABLE,
)


# =========================================================================
# PAGE CONFIG
# =========================================================================
st.set_page_config(
    page_title="Rig Scheduler — CP-SAT Optimizer",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛢️ Rig Scheduler — CP-SAT Optimizer")
st.caption("Upload CSV inputs, configure parameters, run the optimizer, and explore results.")

if not ORTOOLS_AVAILABLE:
    st.error("OR-Tools is not installed. Run: `pip install ortools`")


# =========================================================================
# SIDEBAR — CONFIGURATION
# =========================================================================
st.sidebar.header("⚙️ Configuration")

# --- Starting Resources ---
with st.sidebar.expander("🔧 Starting Resources", expanded=True):
    num_rigs = st.number_input("Rigs", min_value=1, max_value=20, value=1, step=1, key="s_rigs")
    num_frac_crews = st.number_input("Frac Crews", min_value=1, max_value=10, value=1, step=1, key="s_frac")
    num_land_crews = st.number_input("Land Crews", min_value=1, max_value=20, value=3, step=1, key="s_land")
    num_permit_crews = st.number_input("Permit Crews", min_value=1, max_value=20, value=3, step=1, key="s_perm")
    num_construction_crews = st.number_input("Construction Crews", min_value=1, max_value=20, value=3, step=1, key="s_con")

# --- Max Resources ---
with st.sidebar.expander("📈 Max Resources"):
    max_rigs = st.number_input("Max Rigs", min_value=1, max_value=20, value=1, step=1, key="m_rigs")
    max_frac_crews = st.number_input("Max Frac Crews", min_value=1, max_value=10, value=1, step=1, key="m_frac")
    max_land_crews = st.number_input("Max Land Crews", min_value=1, max_value=20, value=3, step=1, key="m_land")
    max_permit_crews = st.number_input("Max Permit Crews", min_value=1, max_value=20, value=3, step=1, key="m_perm")
    max_construction_crews = st.number_input("Max Construction Crews", min_value=1, max_value=20, value=3, step=1, key="m_con")

# --- Capital ---
with st.sidebar.expander("💰 Capital Budget"):
    annual_capex_limit_mm = st.number_input("Annual Capex Limit ($MM)", min_value=10.0, max_value=5000.0, value=350.0, step=25.0, key="capex_lim")
    max_capex_mm = st.number_input("Max Capex ($MM)", min_value=10.0, max_value=5000.0, value=350.0, step=25.0, key="capex_max")
    capex_increment_mm = st.number_input("Capex Increment ($MM)", min_value=10.0, max_value=500.0, value=100.0, step=25.0, key="capex_inc")
    capex_tolerance_mm = st.number_input("Capex Tolerance ($MM)", min_value=0.0, max_value=200.0, value=10.0, step=5.0, key="capex_tol")
    capex_cagr_pct = st.number_input("Capex CAGR (%/yr)", min_value=0.0, max_value=50.0, value=5.0, step=1.0, key="capex_cagr",
                                      help="Compound annual growth rate applied to capex limit each year.")
    capex_cagr_years = st.number_input("CAGR Years", min_value=0, max_value=30, value=10, step=1, key="capex_cagr_yrs",
                                        help="Number of years to apply CAGR before it drops to 0%.")

# --- OL/MS Sub-Budget ---
with st.sidebar.expander("🔧 Overland / Midstream Budget"):
    overland_midstream_budget_mm = st.number_input("OL/MS Budget ($MM)", min_value=0.0, max_value=500.0, value=25.0, step=5.0, key="olms_budget",
                                                    help="Separate sub-budget for overland & midstream within total capex. 0 = no sub-limit.")
    ol_ms_cagr_pct = st.number_input("OL/MS CAGR (%/yr)", min_value=0.0, max_value=50.0, value=5.0, step=1.0, key="olms_cagr")
    ol_ms_cagr_years = st.number_input("OL/MS CAGR Years", min_value=0, max_value=30, value=10, step=1, key="olms_cagr_yrs")

# --- Production Ceiling ---
with st.sidebar.expander("📈 Production Ceiling"):
    production_ceiling_mcfd = st.number_input("Production Ceiling (MCFD)", min_value=0.0, max_value=10_000_000.0, value=0.0, step=100000.0, key="prod_ceil",
                                               help="Max annual average production. 0 = no ceiling. e.g. 1500000 = 1.5 BCFD.")
    production_cagr_pct = st.number_input("Production CAGR (%/yr)", min_value=0.0, max_value=50.0, value=5.0, step=1.0, key="prod_cagr")
    production_cagr_years = st.number_input("Production CAGR Years", min_value=0, max_value=30, value=10, step=1, key="prod_cagr_yrs")

# --- Free Cashflow ---
with st.sidebar.expander("💵 Free Cashflow"):
    commodity_price_per_mcf = st.number_input("Commodity Price ($/MCF)", min_value=0.50, max_value=20.0, value=3.00, step=0.25, key="gas_price",
                                               help="Realized gas price per MCF for revenue calculation.")

# --- Shortfall ---
with st.sidebar.expander("📊 Shortfall Tolerance"):
    shortfall_tolerance_mcfd = st.number_input("Shortfall Tolerance (MCFD)", min_value=0.0, max_value=200000.0, value=10000.0, step=1000.0, key="sf_tol")

# --- CP-SAT Settings ---
with st.sidebar.expander("🧮 CP-SAT Optimizer"):
    cpsat_enabled = st.checkbox("Enable CP-SAT", value=True, key="cpsat_on")
    cpsat_pvi_weight = st.number_input("PVI Weight Multiplier", min_value=1, max_value=10000, value=500, step=50, key="cpsat_pvi")
    cpsat_shortfall_weight = st.number_input("Shortfall Weight", min_value=1, max_value=100000, value=1000, step=100, key="cpsat_sf")
    cpsat_fcf_weight = st.number_input("FCF Weight", min_value=0, max_value=10000, value=100, step=10, key="cpsat_fcf",
                                        help="Weight for FCF maximization in CP-SAT objective.")
    cpsat_time_limit = st.number_input("Time Limit (seconds)", min_value=10, max_value=600, value=120, step=10, key="cpsat_time")
    cpsat_num_workers = st.number_input("Workers (threads)", min_value=1, max_value=16, value=8, step=1, key="cpsat_wrk")
    cpsat_max_pads_text = st.text_input("Max Pads Override (blank = auto)", value="", key="cpsat_pads")
    cpsat_max_pads_override = None
    if cpsat_max_pads_text.strip():
        try:
            cpsat_max_pads_override = int(cpsat_max_pads_text)
        except ValueError:
            st.warning("Max Pads must be integer or blank.")

# --- PVI Reshuffle ---
with st.sidebar.expander("🔄 PVI Reshuffle (Fallback)"):
    pvi_reshuffle_enabled = st.checkbox("Enable Reshuffle", value=True, key="reshuf_on")
    pvi_reshuffle_tolerance = st.number_input("PVI Tolerance", min_value=0.0, max_value=2.0, value=0.3, step=0.05, key="reshuf_tol")
    pvi_reshuffle_min_pvi = st.number_input("Min PVI", min_value=0.0, max_value=5.0, value=1.3, step=0.1, key="reshuf_min")
    pvi_reshuffle_window = st.number_input("Swap Window", min_value=1, max_value=20, value=5, step=1, key="reshuf_win")

# --- Simulation ---
with st.sidebar.expander("📅 Simulation"):
    simulation_days = st.number_input("Simulation Days", min_value=365, max_value=7300, value=3650, step=365, key="sim_days")
    simulation_start_date = st.text_input("Start Date", value="2026-01-01", key="sim_start")


# =========================================================================
# FILE UPLOADS
# =========================================================================
st.header("📁 Input Files")
st.markdown("Upload the **4 required** CSV files below (plus optional global overwrites), then click **Run Optimizer**.")

col1, col2 = st.columns(2)
with col1:
    pad_file = st.file_uploader("📄 Pad Schedule CSV", type=["csv"], key="pad_csv",
                                help="Pad definitions with durations, capex, PVI, etc.")
    well_file = st.file_uploader("📄 Well Decline Parameters CSV", type=["csv"], key="well_csv",
                                 help="Individual well decline curve parameters.")
with col2:
    base_prod_file = st.file_uploader("📄 Base Production CSV", type=["csv"], key="base_csv",
                                      help="Existing production profile (date, rate_mcfd).")
    min_vol_file = st.file_uploader("📄 Minimum Volumes CSV", type=["csv"], key="minvol_csv",
                                    help="Target minimum production volumes.")

overwrite_file = st.file_uploader("📄 Global Overwrites CSV (optional)", type=["csv"], key="overwrite_csv",
                                   help="Per-year overwrite of capital, production, or FCF limits. "
                                        "Columns: year, capital_limit_mm, production_limit_mcfd, fcf_limit_mm, ol_ms_limit_mm.")

# --- Preview uploaded files ---
previews = [
    ("Preview: Pad Schedule", pad_file),
    ("Preview: Well Decline Parameters", well_file),
    ("Preview: Base Production", base_prod_file),
    ("Preview: Minimum Volumes", min_vol_file),
    ("Preview: Global Overwrites", overwrite_file),
]
for label, f in previews:
    if f is not None:
        with st.expander(label):
            try:
                df_preview = pd.read_csv(f, encoding="utf-8-sig")
                st.dataframe(df_preview, use_container_width=True, height=200)
            except Exception as e:
                st.error(f"Cannot read file: {e}")
            f.seek(0)


# =========================================================================
# HELPER — save uploaded file to temp and return path
# =========================================================================
def save_upload(uploaded, filename):
    """Save uploaded file to a temp dir and return its path."""
    tmp_dir = os.path.join(tempfile.gettempdir(), "rig_scheduler_uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, filename)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


# =========================================================================
# RUN BUTTON
# =========================================================================
all_files_uploaded = all([pad_file, well_file, base_prod_file, min_vol_file])

if not all_files_uploaded:
    missing = []
    if not pad_file: missing.append("Pad Schedule")
    if not well_file: missing.append("Well Decline Parameters")
    if not base_prod_file: missing.append("Base Production")
    if not min_vol_file: missing.append("Minimum Volumes")
    st.warning(f"**Missing files:** {', '.join(missing)}")
else:
    st.success("All 4 files uploaded. Ready to run!")

st.markdown("---")
run_button = st.button("🚀 Run Optimizer", disabled=not all_files_uploaded, type="primary",
                       use_container_width=True)

if run_button and all_files_uploaded:
    # Save uploads to temp files
    pad_path = save_upload(pad_file, "pad_schedule.csv")
    well_path = save_upload(well_file, "well_decline.csv")
    base_path = save_upload(base_prod_file, "base_production.csv")
    minvol_path = save_upload(min_vol_file, "minimum_volumes.csv")
    overwrite_path = ""
    if overwrite_file is not None:
        overwrite_path = save_upload(overwrite_file, "global_overwrites.csv")

    # Build output folder in temp
    out_dir = os.path.join(tempfile.gettempdir(), "rig_scheduler_output")
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    config = SimConfig(
        num_rigs=num_rigs,
        num_frac_crews=num_frac_crews,
        num_land_crews=num_land_crews,
        num_permit_crews=num_permit_crews,
        num_construction_crews=num_construction_crews,
        max_rigs=max_rigs,
        max_frac_crews=max_frac_crews,
        max_land_crews=max_land_crews,
        max_permit_crews=max_permit_crews,
        max_construction_crews=max_construction_crews,
        annual_capex_limit_mm=annual_capex_limit_mm,
        max_capex_mm=max_capex_mm,
        capex_increment_mm=capex_increment_mm,
        capex_tolerance_mm=capex_tolerance_mm,
        capex_cagr_pct=capex_cagr_pct,
        capex_cagr_years=capex_cagr_years,
        overland_midstream_budget_mm=overland_midstream_budget_mm,
        ol_ms_cagr_pct=ol_ms_cagr_pct,
        ol_ms_cagr_years=ol_ms_cagr_years,
        production_ceiling_mcfd=production_ceiling_mcfd,
        production_cagr_pct=production_cagr_pct,
        production_cagr_years=production_cagr_years,
        commodity_price_per_mcf=commodity_price_per_mcf,
        shortfall_tolerance_mcfd=shortfall_tolerance_mcfd,
        cpsat_enabled=cpsat_enabled,
        cpsat_pvi_weight_multiplier=cpsat_pvi_weight,
        cpsat_shortfall_weight=cpsat_shortfall_weight,
        cpsat_fcf_weight=cpsat_fcf_weight,
        cpsat_time_limit_seconds=cpsat_time_limit,
        cpsat_num_workers=cpsat_num_workers,
        cpsat_max_pads_override=cpsat_max_pads_override,
        pvi_reshuffle_enabled=pvi_reshuffle_enabled,
        pvi_reshuffle_tolerance=pvi_reshuffle_tolerance,
        pvi_reshuffle_min_pvi=pvi_reshuffle_min_pvi,
        pvi_reshuffle_window=pvi_reshuffle_window,
        simulation_days=simulation_days,
        simulation_start_date=simulation_start_date,
        pad_filepath=pad_path,
        well_filepath=well_path,
        base_production_filepath=base_path,
        minimum_volume_filepath=minvol_path,
        global_overwrite_filepath=overwrite_path,
        schedule_output=os.path.join(out_dir, "pad_schedule.csv"),
        capex_output=os.path.join(out_dir, "capex_timeline.csv"),
        well_prod_output=os.path.join(out_dir, "well_production_output.csv"),
        monthly_output=os.path.join(out_dir, "monthly_results.csv"),
        adjustment_output=os.path.join(out_dir, "adjustment_log.csv"),
        plot_output_folder=plot_dir,
    )

    # =====================================================================
    # EXECUTE — with live console output
    # =====================================================================
    st.subheader("🔄 Running Optimizer...")
    status_container = st.status("**Optimizer running...**", expanded=True, state="running")

    with status_container:
        phase_text = st.empty()
        live_console = st.empty()
        elapsed_text = st.empty()

    log_capture = LiveStreamCapture(live_console)
    run_success = False
    start_time = time.time()

    old_stdout = sys.stdout
    sys.stdout = log_capture
    try:
        phase_text.markdown("**Phase 1/5:** Loading input data...")
        base_production = load_base_production(base_path, simulation_days)
        minimum_volumes = load_minimum_volumes(minvol_path, simulation_days)
        global_overwrites = load_global_overwrites(overwrite_path) if overwrite_path else {}
        elapsed_text.caption(f"⏱️ Elapsed: {time.time() - start_time:.0f}s")

        phase_text.markdown("**Phase 2/5:** Running auto-adjustment engine (CP-SAT optimization + simulation)...")
        sim, monthly, adj_log, all_iters, cpsat_diagnostics, all_strategy_results = \
            run_with_auto_adjustment(config, base_production, minimum_volumes, global_overwrites=global_overwrites)
        elapsed_text.caption(f"⏱️ Elapsed: {time.time() - start_time:.0f}s")

        phase_text.markdown("**Phase 3/5:** Generating plots...")
        final_optimizer = adj_log[-1].get("optimizer", "unknown") if adj_log else "unknown"
        plot_final_results(monthly, sim, sim.config, final_optimizer, plot_folder=plot_dir)
        plt.close("all")

        plot_strategy_comparison(all_strategy_results, config, plot_folder=plot_dir)
        plt.close("all")

        plot_iteration_overlay(all_iters, sim.config, plot_folder=plot_dir)
        plt.close("all")

        if cpsat_diagnostics and cpsat_diagnostics.get("callback"):
            plot_cpsat_diagnostics(cpsat_diagnostics, config, plot_folder=plot_dir)
            plt.close("all")

            final_baselines = [k for k in all_strategy_results if k.startswith("baseline_pvi")]
            baseline_for_plot = all_strategy_results[final_baselines[-1]] if final_baselines else None
            plot_cpsat_solution_comparison(
                cpsat_diagnostics, sim.config,
                base_production, minimum_volumes,
                top_k=5, baseline_result=baseline_for_plot,
                plot_folder=plot_dir,
                global_overwrites=global_overwrites)
            plt.close("all")
        elapsed_text.caption(f"⏱️ Elapsed: {time.time() - start_time:.0f}s")

        phase_text.markdown("**Phase 4/5:** Saving CSV outputs...")
        sim.get_schedule().to_csv(config.schedule_output, index=False)
        ct = sim.get_capex_timeline()
        if not ct.empty:
            ct.to_csv(config.capex_output, index=False)
        wp = sim.get_well_production()
        if not wp.empty:
            wp.to_csv(config.well_prod_output, index=False)
        monthly.to_csv(config.monthly_output, index=False)
        pd.DataFrame(adj_log).to_csv(config.adjustment_output, index=False)

        comp_df = print_strategy_comparison(all_strategy_results, config)
        if comp_df is not None:
            comp_df.to_csv(os.path.join(plot_dir, "strategy_comparison.csv"), index=False)

        total_time = time.time() - start_time
        phase_text.markdown(f"**Phase 5/5:** ✅ Complete in {total_time:.0f} seconds!")
        elapsed_text.caption(f"⏱️ Total time: {total_time:.0f}s")
        status_container.update(label=f"**✅ Optimizer complete** ({total_time:.0f}s)", state="complete", expanded=False)
        run_success = True

    except Exception as e:
        total_time = time.time() - start_time
        status_container.update(label=f"**❌ Error after {total_time:.0f}s**", state="error", expanded=True)
        st.error(f"**Error:** {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        sys.stdout = old_stdout

    # =====================================================================
    # STORE RESULTS IN SESSION STATE
    # =====================================================================
    if run_success:
        st.session_state["results"] = {
            "sim": sim,
            "monthly": monthly,
            "adj_log": adj_log,
            "all_iters": all_iters,
            "cpsat_diagnostics": cpsat_diagnostics,
            "all_strategy_results": all_strategy_results,
            "config": config,
            "plot_dir": plot_dir,
            "out_dir": out_dir,
            "log": log_capture.getvalue(),
        }
        st.rerun()


# =========================================================================
# DISPLAY RESULTS (persisted in session state across reruns)
# =========================================================================
if "results" in st.session_state:
    R = st.session_state["results"]
    sim = R["sim"]
    monthly = R["monthly"]
    adj_log = R["adj_log"]
    all_iters = R["all_iters"]
    cpsat_diagnostics = R["cpsat_diagnostics"]
    all_strategy_results = R["all_strategy_results"]
    config = R["config"]
    plot_dir = R["plot_dir"]
    out_dir = R["out_dir"]

    st.markdown("---")

    # =================================================================
    # SUMMARY METRICS
    # =================================================================
    st.header("📊 Results Summary")

    passed, shortfall = sim.check_minimum_volumes(monthly, tolerance=config.shortfall_tolerance_mcfd)
    raw_sf = monthly[monthly["avg_min_vol_mcfd"] > 0]
    raw_max = raw_sf["avg_shortfall_mcfd"].max() if not raw_sf.empty else 0
    total_prod = monthly["monthly_volume_mcf"].sum()
    peak_prod = monthly["avg_total_mcfd"].max()
    producing = sum(1 for p in sim.pads if p.first_production_day is not None)
    final_opt = adj_log[-1].get("optimizer", "unknown") if adj_log else "unknown"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Status", "✅ PASSED" if passed else "❌ SHORTFALL")
    c2.metric("Total Production", f"{total_prod / 1e6:,.1f} MMCF")
    c3.metric("Peak Rate", f"{peak_prod:,.0f} MCFD")
    c4.metric("Max Shortfall", f"{raw_max:,.0f} MCFD")
    c5.metric("Pads Producing", f"{producing}/{len(sim.pads)}")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Optimizer", final_opt)
    c7.metric("Iterations", str(len(all_iters)))
    c8.metric("Resources",
              f"{config.num_rigs}R {config.num_frac_crews}F {config.num_land_crews}L")
    c9.metric("Capex Ceiling",
              f"${config.effective_annual_capex_mm:,.0f}MM")
    c10.metric("CP-SAT", "✅ Available" if ORTOOLS_AVAILABLE else "❌ Missing")

    # --- FCF summary metrics row ---
    if config.commodity_price_per_mcf > 0 and "monthly_fcf_mm" in monthly.columns:
        st.markdown("---")
        fcf_summary = sim.get_fcf_summary()
        total_rev = sum(v["revenue_mm"] for v in fcf_summary.values())
        total_opex = sum(v["opex_mm"] for v in fcf_summary.values())
        total_capex_mm = sum(v["capex_mm"] for v in fcf_summary.values())
        total_fcf = sum(v["fcf_mm"] for v in fcf_summary.values())
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        fc1.metric("💰 Total Revenue", f"${total_rev:,.1f}MM")
        fc2.metric("Total OpEx", f"${total_opex:,.1f}MM")
        fc3.metric("Total CapEx", f"${total_capex_mm:,.1f}MM")
        fc4.metric("Free Cash Flow", f"${total_fcf:,.1f}MM")
        fc5.metric("Cumulative FCF", f"${monthly['cumulative_fcf_mm'].iloc[-1]:,.1f}MM")

    # --- Production ceiling check ---
    if config.production_ceiling_mcfd > 0:
        ceiling_ok, ceiling_violations = sim.check_production_ceiling(monthly)
        if ceiling_ok:
            st.success("✅ Production ceiling not exceeded.")
        else:
            st.warning("⚠️ Production ceiling exceeded in some years:")
            st.dataframe(ceiling_violations, use_container_width=True)

    # =================================================================
    # TABS
    # =================================================================
    tab_charts, tab_schedule, tab_monthly, tab_capex, tab_strategy, tab_cpsat, tab_log, tab_downloads = \
        st.tabs(["📈 Charts", "📋 Pad Schedule", "📊 Monthly Results",
                 "💰 Capex", "🏆 Strategy Comparison",
                 "🧮 CP-SAT Diagnostics", "📝 Console Log", "⬇️ Downloads"])

    # -----------------------------------------------------------------
    # TAB: Charts
    # -----------------------------------------------------------------
    with tab_charts:
        st.subheader("Final Results Dashboard")
        final_plot = os.path.join(plot_dir, "final_result.png")
        if os.path.exists(final_plot):
            st.image(final_plot, use_container_width=True)
        else:
            st.warning("Final results plot not generated.")

        st.subheader("Strategy Comparison")
        strat_plot = os.path.join(plot_dir, "strategy_comparison.png")
        if os.path.exists(strat_plot):
            st.image(strat_plot, use_container_width=True)

        st.subheader("Iteration Overlay")
        iter_plot = os.path.join(plot_dir, "iteration_overlay.png")
        if os.path.exists(iter_plot):
            st.image(iter_plot, use_container_width=True)

    # -----------------------------------------------------------------
    # TAB: Pad Schedule
    # -----------------------------------------------------------------
    with tab_schedule:
        schedule_df = sim.get_schedule()
        st.subheader(f"Pad Schedule ({len(schedule_df)} pads)")

        fc1, fc2 = st.columns(2)
        with fc1:
            all_statuses = schedule_df["status"].unique().tolist()
            status_filter = st.multiselect(
                "Filter by Status", all_statuses,
                default=all_statuses, key="sched_status")
        with fc2:
            mandatory_filter = st.checkbox("Mandatory only", value=False, key="sched_mand")

        filtered = schedule_df[schedule_df["status"].isin(status_filter)]
        if mandatory_filter:
            filtered = filtered[filtered["mandatory_day"].notna()]

        st.dataframe(filtered, use_container_width=True, height=400)

        csv_buf = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download filtered schedule", csv_buf,
                           "filtered_schedule.csv", "text/csv", key="dl_sched")

    # -----------------------------------------------------------------
    # TAB: Monthly Results
    # -----------------------------------------------------------------
    with tab_monthly:
        st.subheader("Monthly Production vs Targets")

        chart_df = monthly[["month", "avg_new_mcfd", "avg_base_mcfd",
                            "avg_total_mcfd", "avg_min_vol_mcfd",
                            "avg_shortfall_mcfd"]].copy()
        chart_df = chart_df.rename(columns={
            "avg_new_mcfd": "New Production",
            "avg_base_mcfd": "Base Production",
            "avg_total_mcfd": "Total Production",
            "avg_min_vol_mcfd": "Minimum Volume",
            "avg_shortfall_mcfd": "Shortfall",
        })

        st.line_chart(chart_df.set_index("month")[
            ["Total Production", "Minimum Volume", "Base Production"]],
            use_container_width=True, height=400)

        st.subheader("Shortfall Over Time")
        st.bar_chart(chart_df.set_index("month")["Shortfall"],
                     use_container_width=True, height=250,
                     color="#ff4b4b")

        with st.expander("📋 Raw Monthly Data Table"):
            st.dataframe(monthly, use_container_width=True, height=300)

        if not passed:
            st.subheader("⚠️ Months With Shortfall Exceeding Tolerance")
            st.dataframe(
                shortfall[["month", "avg_total_mcfd", "avg_min_vol_mcfd",
                           "avg_shortfall_mcfd"]],
                use_container_width=True)

    # -----------------------------------------------------------------
    # TAB: Capex
    # -----------------------------------------------------------------
    with tab_capex:
        st.subheader("Capital Expenditure Summary")

        capex_summary = sim.get_capex_summary()
        if capex_summary:
            capex_rows = []
            for yr, info in sorted(capex_summary.items()):
                pct = info["spent"] / info["budget"] * 100 if info["budget"] > 0 else 0
                row = {
                    "Year": yr,
                    "Spent ($MM)": round(info["spent"], 1),
                    "Budget ($MM)": round(info["budget"], 0),
                    "Ceiling ($MM)": round(info["ceiling"], 0),
                    "Over Budget ($MM)": round(info["over_budget"], 1),
                    "Over Ceiling ($MM)": round(info["over_ceiling"], 1),
                    "% of Budget": round(pct, 0),
                }
                if info.get("ol_ms_limit", float("inf")) != float("inf"):
                    row["OL/MS Spent ($MM)"] = round(info["ol_ms_spent"], 1)
                    row["OL/MS Limit ($MM)"] = round(info["ol_ms_limit"], 0)
                    row["OL/MS Over ($MM)"] = round(info["ol_ms_over"], 1)
                capex_rows.append(row)
            st.dataframe(pd.DataFrame(capex_rows), use_container_width=True)

        ct_df = sim.get_capex_timeline()
        if not ct_df.empty:
            st.subheader("Capex Timeline by Milestone")
            st.dataframe(ct_df, use_container_width=True, height=300)

    # -----------------------------------------------------------------
    # TAB: Strategy Comparison
    # -----------------------------------------------------------------
    with tab_strategy:
        st.subheader("Strategy Comparison")

        strat_plot2 = os.path.join(plot_dir, "strategy_comparison.png")
        if os.path.exists(strat_plot2):
            st.image(strat_plot2, use_container_width=True)

        if all_strategy_results:
            rows = []
            for label, res in all_strategy_results.items():
                rows.append({
                    "Strategy": label,
                    "Passed": "✅" if res["passed"] else "❌",
                    "Max Shortfall (MCFD)": round(res["raw_max_shortfall"], 0),
                    "Total Prod (MMCF)": round(res["total_prod_mcf"] / 1e6, 1),
                    "Peak (MCFD)": round(res["peak_mcfd"], 0),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # -----------------------------------------------------------------
    # TAB: CP-SAT Diagnostics
    # -----------------------------------------------------------------
    with tab_cpsat:
        if cpsat_diagnostics and cpsat_diagnostics.get("callback"):
            st.subheader("CP-SAT Solver Diagnostics")

            diag_plot = os.path.join(plot_dir, "cpsat_diagnostics.png")
            if os.path.exists(diag_plot):
                st.image(diag_plot, use_container_width=True)

            sol_plot = os.path.join(plot_dir, "cpsat_solution_comparison.png")
            if os.path.exists(sol_plot):
                st.image(sol_plot, use_container_width=True)

            stats = cpsat_diagnostics.get("solver_stats")
            if stats:
                st.subheader("Solver Statistics")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Status", stats.get("status", "N/A"))
                sc2.metric("Solutions Found", stats.get("num_solutions_found", 0))
                gap = stats.get("optimality_gap_pct")
                sc3.metric("Optimality Gap", f"{gap:.2f}%" if gap is not None else "N/A")
                sc4.metric("Wall Time", f"{stats.get('wall_time_sec', 0):.1f}s")

                sc5, sc6, sc7, sc8 = st.columns(4)
                sc5.metric("Branches", f"{stats.get('num_branches', 0):,}")
                sc6.metric("Conflicts", f"{stats.get('num_conflicts', 0):,}")
                sc7.metric("Pads Optimized", cpsat_diagnostics.get("final_subset_size", "N/A"))
                sc8.metric("Tier Attempts", len(cpsat_diagnostics.get("tier_attempts", [])))
        else:
            st.info("CP-SAT diagnostics not available for this run.")

    # -----------------------------------------------------------------
    # TAB: Console Log
    # -----------------------------------------------------------------
    with tab_log:
        st.subheader("Engine Console Output")
        log_text = R.get("log", "")
        if log_text:
            st.code(log_text, language="text")
        else:
            st.info("No console output captured.")

    # -----------------------------------------------------------------
    # TAB: Downloads
    # -----------------------------------------------------------------
    with tab_downloads:
        st.subheader("⬇️ Download Output Files")

        # CSVs
        st.markdown("**CSV Files**")
        dl_cols = st.columns(3)
        csv_files = [
            ("📄 Pad Schedule", config.schedule_output),
            ("📄 Monthly Results", config.monthly_output),
            ("📄 Capex Timeline", config.capex_output),
            ("📄 Well Production", config.well_prod_output),
            ("📄 Adjustment Log", config.adjustment_output),
        ]
        for i, (label, path) in enumerate(csv_files):
            if os.path.exists(path):
                with open(path, "rb") as f:
                    dl_cols[i % 3].download_button(
                        label=label,
                        data=f.read(),
                        file_name=os.path.basename(path),
                        mime="text/csv",
                        key=f"dl_csv_{i}",
                    )

        # Plots
        st.markdown("**Plot Images**")
        plot_files = []
        if os.path.exists(plot_dir):
            for fn in sorted(os.listdir(plot_dir)):
                if fn.endswith(".png"):
                    plot_files.append(fn)

        if plot_files:
            pcols = st.columns(3)
            for i, fn in enumerate(plot_files):
                fpath = os.path.join(plot_dir, fn)
                with open(fpath, "rb") as f:
                    pcols[i % 3].download_button(
                        label=f"🖼️ {fn}",
                        data=f.read(),
                        file_name=fn,
                        mime="image/png",
                        key=f"dl_plot_{i}",
                    )
        else:
            st.info("No plots generated.")

    # -----------------------------------------------------------------
    # FCF & GROWTH ANALYSIS (shown below tabs when enabled)
    # -----------------------------------------------------------------
    if config.commodity_price_per_mcf > 0 and "monthly_fcf_mm" in monthly.columns:
        st.markdown("---")
        st.header("💰 Free Cash Flow Analysis")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Cumulative FCF
        ax1 = axes[0]
        ax1.plot(monthly["month"], monthly["cumulative_fcf_mm"], "g-", linewidth=2, label="Cumulative FCF")
        ax1.axhline(0, color="red", linewidth=0.5, linestyle="--")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("$MM")
        ax1.set_title("Cumulative Free Cash Flow")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Monthly FCF
        ax2 = axes[1]
        ax2.bar(monthly["month"], monthly["monthly_fcf_mm"], label="Monthly FCF", alpha=0.7, color="green")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("$MM")
        ax2.set_title("Monthly Free Cash Flow")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("📋 FCF Data Table"):
            fcf_summary = sim.get_fcf_summary()
            fcf_rows = []
            for yr, info in sorted(fcf_summary.items()):
                fcf_rows.append({
                    "Year": yr,
                    "Revenue ($MM)": round(info["revenue_mm"], 1),
                    "OpEx ($MM)": round(info["opex_mm"], 1),
                    "CapEx ($MM)": round(info["capex_mm"], 1),
                    "FCF ($MM)": round(info["fcf_mm"], 1),
                })
            st.dataframe(pd.DataFrame(fcf_rows), use_container_width=True)

    # =================================================================
    # ADJUSTMENT LOG (always visible)
    # =================================================================
    with st.expander("🔧 Adjustment Log"):
        if adj_log:
            st.dataframe(pd.DataFrame(adj_log), use_container_width=True)
        else:
            st.info("No adjustments were needed.")

    # Clear button
    st.markdown("---")
    if st.button("🗑️ Clear Results & Start Over", key="clear"):
        del st.session_state["results"]
        st.rerun()

    plt.close("all")
