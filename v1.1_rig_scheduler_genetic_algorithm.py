"""
v1_rig_scheduler_genetic_algorithm.py
=====================================

Standalone rig scheduler that uses a *Genetic Algorithm* (GA) to choose the
pad-drill order. No CP-SAT / OR-Tools dependency — the GA evaluates each
candidate ordering with the full daily simulator, so the fitness function is
the TRUE end-of-run free cash flow / water cost / shortfall (no proxies,
no linearization tricks).

Architecture
------------
* All data classes / simulator / CSV loaders are inlined directly into this
  file — there is no runtime dependency on any other rig-scheduler module.
* Chromosome = permutation of pad names (i.e. drill order).
* Fitness    = NPV from a single full daily simulation run:
                  score = PV(FCF)
                          - PV(Water cost)
                          - PV(Shortfall replacement gas @ $/MCF)
               where PV() discounts each monthly cashflow at `discount_rate_annual`.
* Hard constraints (capex budget, production ceiling, mandatory pads): violation
  forces score = -inf so the GA REJECTS those schedules. Toggle each via GAConfig.
* GA maximizes score.
* Operators  : tournament selection, Order-Crossover (OX), swap + insertion mutation,
               elitism (top-N preserved each generation).
* Initial population: a configurable fraction is PVI-seeded; the rest are
  random permutations (mandatory pads stay in the front half).
"""

from __future__ import annotations

import os
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum, auto
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# WATER OUTLET CASCADE
# =============================================================================
# Single source of truth for the water-disposal outlet tiers used after both
# storage tanks are full. Order matters — outlets are filled in this order
# (cheapest → most expensive in practice). Each tuple:
#   (key, label, capacity_attr_on_SimConfig, cost_attr_on_SimConfig)
WATER_OUTLET_CASCADE: List[Tuple[str, str, str, str]] = [
    ("water_sharing", "Water Sharing", "water_sharing_capacity_bwpd", "water_sharing_cost_per_bbl"),
    ("select_rail",   "Select Rail",   "select_rail_capacity_bwpd",   "select_rail_cost_per_bbl"),
    ("pa_swd",        "PA SWD",        "pa_swd_capacity_bwpd",        "pa_swd_cost_per_bbl"),
    ("remainder",     "Remainder",     "remainder_capacity_bwpd",     "remainder_cost_per_bbl"),
]


def _save_fig(fig, folder: str, filename: str, label: str) -> None:
    """Tight-layout, save to {folder}/{filename}, close, and print a confirmation."""
    fig.tight_layout()
    path = os.path.join(folder, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  -> wrote {label}: {path}")


# =============================================================================
# 1. EMBEDDED SIMULATION ENGINE
# =============================================================================

# ----- 1a. Pad lifecycle status enum -----------------------------------------

class PadStatus(Enum):
    WAITING = auto()
    WAITING_LAND = auto()
    LAND_AGREEMENT = auto()
    WAITING_PERMIT = auto()
    PERMITTING = auto()
    WAITING_PAD_CON = auto()
    PAD_CONSTRUCTION = auto()
    PREDRILL = auto()
    WAITING_DRILL = auto()
    DRILLING = auto()
    WAITING_FRAC = auto()
    FRACKING = auto()
    WAITING_PRODUCTION = auto()
    PRODUCING = auto()
    CAPEX_BLOCKED = auto()  # v1.1: pad could not admit any milestone within sim horizon


# ----- 1b. Well decline curve (Arps with switchover) -------------------------

@dataclass
class WellDeclineCurve:
    well_name: str
    pad_name: str
    gas_qi: float
    gas_di: float
    gas_b: float
    gas_final_di: float
    # Water decline curve (Arps, same form as gas) — units: BBL/day for qi
    water_qi: float = 0.0
    water_di: float = 0.0
    water_b: float = 0.0
    water_final_di: float = 0.0
    min_rate: float = 1.0
    min_water_rate: float = 0.0

    def __post_init__(self):
        self._di_nominal: float = self._effective_to_nominal(self.gas_di, self.gas_b)
        self._di_final_nominal: float = self._effective_to_nominal(self.gas_final_di, self.gas_b)
        if self.gas_b != 0 and self._di_nominal > 0 and self._di_final_nominal > 0:
            self._t_sw: Optional[float] = (self._di_nominal - self._di_final_nominal) / (
                self.gas_b * self._di_nominal * self._di_final_nominal)
            self._q_sw: Optional[float] = self.gas_qi * (
                1 + self.gas_b * self._di_nominal * self._t_sw) ** (-1.0 / self.gas_b)
        else:
            self._t_sw = None
            self._q_sw = None
        # Water curve switchover
        self._wdi_nominal: float = self._effective_to_nominal(self.water_di, self.water_b)
        self._wdi_final_nominal: float = self._effective_to_nominal(self.water_final_di, self.water_b)
        if self.water_b != 0 and self._wdi_nominal > 0 and self._wdi_final_nominal > 0:
            self._wt_sw: Optional[float] = (self._wdi_nominal - self._wdi_final_nominal) / (
                self.water_b * self._wdi_nominal * self._wdi_final_nominal)
            self._wq_sw: Optional[float] = self.water_qi * (
                1 + self.water_b * self._wdi_nominal * self._wt_sw) ** (-1.0 / self.water_b)
        else:
            self._wt_sw = None
            self._wq_sw = None

    @staticmethod
    def _effective_to_nominal(d_eff: float, b: float) -> float:
        if d_eff <= 0:
            return 0.0
        if d_eff >= 1.0:
            d_eff = 0.9999
        if b == 0:
            return -np.log(1 - d_eff)
        else:
            return ((1 - d_eff) ** (-b) - 1) / b

    def rate_at_time(self, t_days: float) -> float:
        if t_days < 0:
            return 0.0
        t_years = t_days / 365.25
        if self.gas_b == 0:
            q = self.gas_qi * np.exp(-self._di_nominal * t_years)
        else:
            d_inst = self._di_nominal / (1 + self.gas_b * self._di_nominal * t_years)
            if d_inst > self._di_final_nominal:
                q = self.gas_qi * (
                    1 + self.gas_b * self._di_nominal * t_years
                ) ** (-1.0 / self.gas_b)
            else:
                if self._t_sw is not None:
                    q = self._q_sw * np.exp(-self._di_final_nominal * (t_years - self._t_sw))
                else:
                    t_sw = (self._di_nominal - self._di_final_nominal) / (
                        self.gas_b * self._di_nominal * self._di_final_nominal)
                    q_sw = self.gas_qi * (
                        1 + self.gas_b * self._di_nominal * t_sw) ** (-1.0 / self.gas_b)
                    q = q_sw * np.exp(-self._di_final_nominal * (t_years - t_sw))
        return q if q > self.min_rate else 0.0

    def water_rate_at_time(self, t_days: float) -> float:
        """Water production rate (BBL/day) at t_days since first production. Arps with switchover."""
        if t_days < 0 or self.water_qi <= 0:
            return 0.0
        t_years = t_days / 365.25
        if self.water_b == 0:
            q = self.water_qi * np.exp(-self._wdi_nominal * t_years)
        else:
            d_inst = self._wdi_nominal / (1 + self.water_b * self._wdi_nominal * t_years)
            if d_inst > self._wdi_final_nominal:
                q = self.water_qi * (
                    1 + self.water_b * self._wdi_nominal * t_years
                ) ** (-1.0 / self.water_b)
            else:
                if self._wt_sw is not None:
                    q = self._wq_sw * np.exp(-self._wdi_final_nominal * (t_years - self._wt_sw))
                else:
                    q = self.water_qi * np.exp(-self._wdi_final_nominal * t_years)
        return q if q > self.min_water_rate else 0.0


# ----- 1c. Well pad ----------------------------------------------------------

@dataclass
class WellPad:
    pad_id: str
    name: str
    land_owner_agreement_days: float
    pad_permit_days: float
    pad_construction_days: float
    predrill_days: float = 0.0
    drill_days: float = 0.0
    frac_days: float = 0.0
    midstream_construction_days: float = 0.0
    overland_construction_days: float = 0.0
    midstream_lead_days: float = 0.0
    overland_lead_days: float = 0.0
    capex_land_mm: float = 0.0
    capex_permit_mm: float = 0.0
    capex_pad_construction_mm: float = 0.0
    capex_midstream_mm: float = 0.0
    capex_overland_mm: float = 0.0
    capex_drill_mm: float = 0.0
    capex_frac_mm: float = 0.0
    pvi: float = 0.0
    npv_mm: float = 0.0
    annual_opex_mm: float = 0.0
    earliest_start_day: int = 0
    mandatory_start_day: Optional[int] = None
    required_frac_water: float = 0.0   # Total BBL of water needed for completion (consumed during frac)
    wells: List[WellDeclineCurve] = field(default_factory=list)
    status: PadStatus = PadStatus.WAITING
    sequence_number: Optional[int] = None
    land_start: Optional[int] = None
    land_end: Optional[int] = None
    permit_start: Optional[int] = None
    permit_end: Optional[int] = None
    pad_con_start: Optional[int] = None
    pad_con_end: Optional[int] = None
    predrill_start: Optional[int] = None
    predrill_end: Optional[int] = None
    drill_start: Optional[int] = None
    drill_end: Optional[int] = None
    frac_start: Optional[int] = None
    frac_end: Optional[int] = None
    midstream_start: Optional[int] = None
    midstream_end: Optional[int] = None
    overland_start: Optional[int] = None
    overland_end: Optional[int] = None
    first_production_day: Optional[int] = None
    # Cumulative number of days this pad's first-production has been deferred
    # by the water-takeaway deferral mechanism (see SimConfig.water_deferral_*).
    production_deferred_days: int = 0

    @property
    def is_mandatory(self) -> bool:
        return self.mandatory_start_day is not None

    @property
    def num_wells(self) -> int:
        return len(self.wells)

    @property
    def total_qi_mcfd(self) -> float:
        return sum(w.gas_qi for w in self.wells)

    @property
    def total_capex_mm(self) -> float:
        return (self.capex_land_mm + self.capex_permit_mm +
                self.capex_pad_construction_mm + self.capex_midstream_mm +
                self.capex_overland_mm + self.capex_drill_mm + self.capex_frac_mm)

    @property
    def total_cycle_days(self) -> float:
        return (self.land_owner_agreement_days + self.pad_permit_days +
                self.pad_construction_days +
                self.drill_days + self.frac_days)

    def production_at_day(self, sim_day: int) -> float:
        if self.first_production_day is None or sim_day < self.first_production_day:
            return 0.0
        return sum(w.rate_at_time(sim_day - self.first_production_day) for w in self.wells)

    def water_production_at_day(self, sim_day: int) -> float:
        """Total produced water rate (BWPD) for this pad at sim_day."""
        if self.first_production_day is None or sim_day < self.first_production_day:
            return 0.0
        t = sim_day - self.first_production_day
        return sum(w.water_rate_at_time(t) for w in self.wells)

    def frac_water_demand_bwpd(self) -> float:
        """Average BBL/day of water needed during fracking (required_frac_water / frac_days)."""
        if self.frac_days <= 0 or self.required_frac_water <= 0:
            return 0.0
        return self.required_frac_water / self.frac_days

    def well_production_at_day(self, sim_day: int) -> Dict[str, float]:
        if self.first_production_day is None or sim_day < self.first_production_day:
            return {w.well_name: 0.0 for w in self.wells}
        t = sim_day - self.first_production_day
        return {w.well_name: w.rate_at_time(t) for w in self.wells}

    def approx_monthly_production(self, max_months: int = 120) -> List[float]:
        return [sum(w.rate_at_time(m * 30) for w in self.wells) for m in range(max_months)]


# ----- 1d. Simulation configuration -----------------------------------------

@dataclass
class SimConfig:
    """SINGLE CONFIGURATION BLOCK — edit here only."""
    # Resources (explicit fixed counts)
    num_rigs: int = 1
    num_frac_crews: int = 1
    num_land_crews: int = 1
    num_permit_crews: int = 1
    num_construction_crews: int = 1

    # Capital (base year)
    annual_capex_limit_mm: float = 500
    capex_tolerance_mm: float = 25

    # Capex disbursement model (v1.1)
    # - "lump_start": entire milestone cost lands on the milestone START day
    #   (legacy v1.0 behavior).
    # - "even": cost is spread uniformly across `[start, start + ceil(duration))`
    #   so long milestones smooth their impact on annual capex caps and the
    #   daily FCF stream. Multi-year milestones are checked against EACH
    #   year's cap; bleed-over is hard-rejected by `_can_admit`.
    capex_disbursement: str = "even"

    # Mandatory-pad pre-reservation (v1.1)
    # When True (default), mandatory pads' capex is back-chained from
    # `mandatory_start_day` and committed into the daily capex arrays during
    # `OrderedDrillingSimulator.__init__` BEFORE any GA pad ordering runs.
    # Non-mandatory pads then admit against the remaining cap.
    reserve_mandatory_capex: bool = True

    # Mandatory-pad slip tolerance: a mandatory pad whose actual land start
    # exceeds `mandatory_start_day + grace_days` is counted as a violation.
    mandatory_pad_grace_days: int = 0

    # Capital CAGR
    capex_cagr_pct: float = 0.0        # % compound annual growth rate applied to capex limit
    capex_cagr_years: int = 0           # number of years to apply CAGR before it drops to 0%

    # Overland / Midstream separate sub-budget (within total capital)
    overland_midstream_budget_mm: float = 0.0   # 0 = no sub-limit; >0 = separate OL/MS budget
    ol_ms_cagr_pct: float = 0.0                 # CAGR applied to OL/MS sub-budget
    ol_ms_cagr_years: int = 0

    # Production ceiling
    production_ceiling_mcfd: float = 0.0  # 0 = no ceiling; >0 = hard annual avg production limit
    production_cagr_pct: float = 0.0
    production_cagr_years: int = 0

    # Free cashflow
    commodity_price_per_mcf: float = 3.0   # $/MCF for revenue calculation

    # ---------------------------------------------------------------
    # WATER MANAGEMENT (v2.0)
    # ---------------------------------------------------------------
    water_enabled: bool = True
    rainfall_bwpd: float = 400.0               # Field-wide rainfall water disposal load (BWPD)
    storage_company_capacity_bbl: float = 200_000.0
    storage_thirdparty_capacity_bbl: float = 75_000.0
    water_to_frac_cost_per_bbl: float = 2.75   # Cost when water (from wells or storage) is sent to frac
    storage_company_cost_per_bbl: float = 4.50    # Cost to send each bbl into company-owned storage
    storage_thirdparty_cost_per_bbl: float = 13.00  # Cost to send each bbl into 3rd-party storage
    # Outlet tiers (used after both storage tanks are full and frac is not running)
    water_sharing_capacity_bwpd: float = 1500.0
    water_sharing_cost_per_bbl: float = 8.00
    select_rail_capacity_bwpd: float = 3000.0
    select_rail_cost_per_bbl: float = 13.35
    pa_swd_capacity_bwpd: float = 1000.0
    pa_swd_cost_per_bbl: float = 18.09
    remainder_capacity_bwpd: float = 15000.0
    remainder_cost_per_bbl: float = 100.00

    # Shortfall tolerance
    shortfall_tolerance_mcfd: float = 50000

    # ---- Water-takeaway production deferral --------------------------------
    # If enabled, a pad that has finished frac + midstream + overland will NOT
    # be turned in line (status -> PRODUCING) on a day where a SAME-DAY
    # predictive water balance shows the system would overflow by more than
    # `water_deferral_trigger_bbl` bbl. The forecast = (existing producers'
    # water + base water + rainfall + this pad's day-1 water) − frac demand −
    # total outlet takeaway capacity − remaining storage headroom. If the
    # forecast overflow exceeds the trigger, the pad stays in
    # WAITING_PRODUCTION and `production_deferred_days` is incremented. Once a
    # pad has been deferred for `water_deferral_max_days`, it is forced online
    # regardless of water status. Mandatory pads are NOT deferred.
    water_deferral_enabled: bool = False
    water_deferral_max_days: int = 45
    water_deferral_trigger_bbl: float = 0.0   # forecast overflow bbl that triggers deferral

    # Simulation
    simulation_days: int = 3650
    simulation_start_date: str = "2026-01-01"

    # Input files
    pad_filepath: str = ""
    well_filepath: str = ""
    base_production_filepath: str = ""
    minimum_volume_filepath: str = ""

    # Output files
    schedule_output: str = ""
    capex_output: str = ""
    well_prod_output: str = ""
    monthly_output: str = ""
    adjustment_output: str = ""
    water_output: str = ""
    plot_output_folder: str = ""

    def capex_limit_for_year(self, year: int) -> float:
        """Annual capex limit applying CAGR."""
        growth_years = min(year, self.capex_cagr_years) if self.capex_cagr_years > 0 else 0
        return self.annual_capex_limit_mm * (1 + self.capex_cagr_pct / 100.0) ** growth_years

    def ol_ms_limit_for_year(self, year: int) -> float:
        """OL/MS sub-budget for year (0 = unlimited within global budget)."""
        if self.overland_midstream_budget_mm <= 0:
            return float("inf")
        growth = min(year, self.ol_ms_cagr_years) if self.ol_ms_cagr_years > 0 else 0
        return self.overland_midstream_budget_mm * (1 + self.ol_ms_cagr_pct / 100.0) ** growth

    def production_limit_for_year(self, year: int) -> float:
        """Annual average production ceiling (MCFD). Returns inf if no ceiling."""
        if self.production_ceiling_mcfd <= 0:
            return float("inf")
        growth = min(year, self.production_cagr_years) if self.production_cagr_years > 0 else 0
        return self.production_ceiling_mcfd * (1 + self.production_cagr_pct / 100.0) ** growth

    @property
    def effective_annual_capex_mm(self) -> float:
        return self.annual_capex_limit_mm + self.capex_tolerance_mm


# ----- 1e. CSV loaders -------------------------------------------------------

def _load_time_series_csv(filepath: str, simulation_days: int,
                          tail_fill: bool = False,
                          rate_keywords: Optional[List[str]] = None) -> np.ndarray:
    """Load a daily/monthly time series from CSV.

    rate_keywords: list of substrings (lowercased) used to identify the rate column.
        Defaults to ["gas rate", "mcfd", "rate"] (in priority order).
        For water use ["water rate", "bwpd", "bbl"].
    """
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    if rate_keywords is None:
        rate_keywords = ["gas rate", "mcfd", "rate"]
    rate_col = None
    for kw in rate_keywords:
        for c in df.columns:
            if kw in c.lower():
                rate_col = c
                break
        if rate_col is not None:
            break
    if rate_col is None:
        raise ValueError(f"No rate column found in {filepath}. Tried keywords: {rate_keywords}. "
                         f"Columns: {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df["rate"] = pd.to_numeric(df[rate_col], errors="coerce").fillna(0)
    ref_date = df[date_col].iloc[0]
    arr = np.zeros(simulation_days)
    for _, row in df.iterrows():
        ds = max(0, (row[date_col] - ref_date).days)
        de = min(simulation_days, (row[date_col] + pd.offsets.MonthBegin(1) - ref_date).days)
        if ds < simulation_days:
            arr[ds:de] = row["rate"]
    if tail_fill:
        lv = df["rate"].iloc[-1]
        ld = min(simulation_days, (df[date_col].iloc[-1] - ref_date).days + 30)
        if ld < simulation_days:
            arr[ld:] = lv
    return arr


def load_base_production(filepath: str, simulation_days: int) -> np.ndarray:
    """Base GAS production (MCFD) from the base production CSV."""
    arr = _load_time_series_csv(filepath, simulation_days, tail_fill=True,
                                 rate_keywords=["gas rate", "mcfd", "rate"])
    print(f"Base gas production: {arr[0]:,.0f} → {arr[-1]:,.0f} MCFD")
    return arr


def load_base_water(filepath: str, simulation_days: int) -> np.ndarray:
    """Base WATER production (BWPD) from the base production CSV."""
    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        has_water = any(("water rate" in c.lower()) or ("bwpd" in c.lower())
                        for c in df.columns)
        if not has_water:
            print("Base water production: no Water Rate column found — using zeros.")
            return np.zeros(simulation_days)
        arr = _load_time_series_csv(filepath, simulation_days, tail_fill=True,
                                     rate_keywords=["water rate", "bwpd", "bbl"])
        print(f"Base water production: {arr[0]:,.0f} → {arr[-1]:,.0f} BWPD")
        return arr
    except Exception as e:
        print(f"Base water production: failed to load ({e}) — using zeros.")
        return np.zeros(simulation_days)


def load_minimum_volumes(filepath: str, simulation_days: int) -> np.ndarray:
    arr = _load_time_series_csv(filepath, simulation_days, tail_fill=False)
    print(f"Minimum volumes: {arr[0]:,.0f} MCFD, {int(np.sum(arr > 0) / 30)} months")
    return arr


def load_pads_from_csv(filepath: str, sim_start_date: str = "2026-01-01") -> List[WellPad]:
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how="all").dropna(subset=["name"])
    ref_date = pd.Timestamp(sim_start_date)
    pads = []
    for idx, row in df.iterrows():
        dur = {}
        for f in ["land_owner_agreement_days", "pad_permit_days",
                   "pad_construction_days", "midstream_construction_days",
                   "overland_construction_days",
                   "drill_days", "frac_days"]:
            dur[f] = row.get(f, np.nan)
        # Optional durations — default to 0 if missing
        for f in ["predrill_days", "midstream_lead_days", "overland_lead_days"]:
            v = row.get(f, 0.0)
            dur[f] = 0.0 if pd.isna(v) else v
        capex = {}
        for f in ["capex_land_mm", "capex_permit_mm", "capex_pad_construction_mm",
                   "capex_midstream_mm", "capex_overland_mm",
                   "capex_drill_mm", "capex_frac_mm"]:
            capex[f] = row.get(f, 0.0)
            if pd.isna(capex[f]):
                capex[f] = 0.0
        skip = False
        for fn, val in dur.items():
            if fn in ("predrill_days", "midstream_lead_days", "overland_lead_days"):
                continue  # optional — already defaulted to 0
            if pd.isna(val):
                skip = True
                break
        if skip:
            continue
        pvi = row.get("pvi", row.get("PVI", 0.0))
        if pd.isna(pvi):
            pvi = 0.0
        npv = row.get("npv_mm", row.get("NPV_MM", row.get("net_present_value_mm", 0.0)))
        if pd.isna(npv):
            npv = 0.0
        opex = row.get("annual_opex_mm", row.get("opex_mm", 0.0))
        if pd.isna(opex):
            opex = 0.0
        rfw = row.get("required_frac_water", row.get("required_frac_water_bbl", 0.0))
        if pd.isna(rfw):
            rfw = 0.0
        es = row.get("earliest_start", 0)
        if pd.isna(es):
            es = 0
        try:
            es = int(es)
        except (ValueError, TypeError):
            try:
                es = max(0, int((pd.Timestamp(es) - ref_date).days))
            except Exception:
                es = 0
        mandatory_day = None
        md_raw = row.get("mandatory_date", None)
        if md_raw is not None and pd.notna(md_raw) and str(md_raw).strip() != "":
            try:
                mandatory_day = max(0, int((pd.Timestamp(str(md_raw).strip()) - ref_date).days))
            except Exception:
                mandatory_day = None
        pads.append(WellPad(
            pad_id=str(row.get("pad_id", row["name"])),
            name=str(row["name"]),
            land_owner_agreement_days=float(dur["land_owner_agreement_days"]),
            pad_permit_days=float(dur["pad_permit_days"]),
            pad_construction_days=float(dur["pad_construction_days"]),
            predrill_days=float(dur["predrill_days"]),
            drill_days=float(dur["drill_days"]),
            frac_days=float(dur["frac_days"]),
            midstream_construction_days=float(dur["midstream_construction_days"]),
            overland_construction_days=float(dur["overland_construction_days"]),
            midstream_lead_days=float(dur["midstream_lead_days"]),
            overland_lead_days=float(dur["overland_lead_days"]),
            capex_land_mm=float(capex["capex_land_mm"]),
            capex_permit_mm=float(capex["capex_permit_mm"]),
            capex_pad_construction_mm=float(capex["capex_pad_construction_mm"]),
            capex_midstream_mm=float(capex["capex_midstream_mm"]),
            capex_overland_mm=float(capex["capex_overland_mm"]),
            capex_drill_mm=float(capex["capex_drill_mm"]),
            capex_frac_mm=float(capex["capex_frac_mm"]),
            pvi=float(pvi), npv_mm=float(npv), annual_opex_mm=float(opex),
            earliest_start_day=es,
            mandatory_start_day=mandatory_day,
            required_frac_water=float(rfw),
            wells=[],
        ))
    mc = sum(1 for p in pads if p.is_mandatory)
    print(f"Loaded {len(pads)} pads ({mc} mandatory)")
    return pads


def load_wells_from_csv(filepath: str) -> List[WellDeclineCurve]:
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how="all").dropna(subset=["PAD_NAME", "PROPERTY_NAME"])
    wells = []
    for idx, row in df.iterrows():
        try:
            qi = float(str(row["gas_qi"]).replace(",", ""))
            di = float(str(row["gas_di"]).replace("%", "").strip())
            if di > 1.0:
                di /= 100.0
            b = float(row["gas_b"])
            fdi = float(str(row["gas_final_di"]).replace("%", "").strip())
            if fdi > 1.0:
                fdi /= 100.0
        except (ValueError, TypeError):
            continue

        # Water decline parameters (optional — default to 0 = no produced water)
        def _opt_float(col, pct=False):
            if col not in row or pd.isna(row[col]):
                return 0.0
            try:
                v = float(str(row[col]).replace(",", "").replace("%", "").strip())
            except (ValueError, TypeError):
                return 0.0
            if pct and v > 1.0:
                v /= 100.0
            return v

        wqi = _opt_float("water_qi")
        wdi = _opt_float("water_di", pct=True)
        wb = _opt_float("water_b")
        wfdi = _opt_float("water_final_di", pct=True)

        wells.append(WellDeclineCurve(
            well_name=str(row["PROPERTY_NAME"]),
            pad_name=str(row["PAD_NAME"]),
            gas_qi=qi, gas_di=di, gas_b=b, gas_final_di=fdi,
            water_qi=wqi, water_di=wdi, water_b=wb, water_final_di=wfdi))
    print(f"Loaded {len(wells)} wells")
    return wells


def assign_wells_to_pads(pads: List[WellPad], wells: List[WellDeclineCurve]):
    lookup: Dict[str, WellPad] = {p.name.strip().lower(): p for p in pads}
    matched = 0
    for w in wells:
        k = w.pad_name.strip().lower()
        if k in lookup:
            lookup[k].wells.append(w)
            matched += 1
    print(f"Wells matched: {matched}")


def load_global_overwrites(filepath: str) -> Dict[int, Dict[str, Optional[float]]]:
    """DEPRECATED: per-year overwrites have been removed. Always returns {}."""
    return {}


def _load_fresh_sim(config: SimConfig) -> Tuple[List[WellPad], List[WellDeclineCurve]]:
    """Helper: load pads and wells from CSV and assign."""
    pads = load_pads_from_csv(config.pad_filepath, config.simulation_start_date)
    wells = load_wells_from_csv(config.well_filepath)
    assign_wells_to_pads(pads, wells)
    return pads, wells


# ----- 1f. Core simulator ----------------------------------------------------

class OrderedDrillingSimulator:
    def __init__(self, pads: List[WellPad], config: SimConfig,
                 base_production: Optional[np.ndarray] = None,
                 minimum_volumes: Optional[np.ndarray] = None,
                 pad_order: Optional[List[str]] = None,
                 global_overwrites: Optional[Dict] = None,
                 base_water: Optional[np.ndarray] = None):
        self.pads = deepcopy(pads)
        if pad_order is not None:
            om = {name: i for i, name in enumerate(pad_order)}
            self.pads.sort(key=lambda p: om.get(p.name, 9999))
        else:
            self.pads.sort(key=lambda p: p.pvi, reverse=True)
        for i, pad in enumerate(self.pads):
            pad.sequence_number = i + 1
        self.config = config
        self.global_overwrites = global_overwrites or {}
        self.next_pad_index = 0
        self.land_crews_in_use = 0
        self.permit_crews_in_use = 0
        self.construction_crews_in_use = 0
        self.rigs_in_use = 0
        self.frac_crews_in_use = 0

        # ---- Capex bookkeeping (v1.1) -------------------------------------
        # Source of truth = two per-day arrays of disbursed capex ($MM/day).
        # `annual_capex_spent` / `annual_ol_ms_spent` are derived views,
        # rebuilt from the arrays after every commit so legacy reporting code
        # keeps working without changes.
        n = config.simulation_days
        self._daily_capex_planned: np.ndarray = np.zeros(n, dtype=np.float64)
        self._daily_ol_ms_planned: np.ndarray = np.zeros(n, dtype=np.float64)
        self.annual_capex_spent: Dict[int, float] = {}
        self.annual_ol_ms_spent: Dict[int, float] = {}
        # v1.1: pre-reservation overage diagnostics (set by _reserve_mandatory_pads)
        self.mandatory_capex_overage_mm: float = 0.0
        self.mandatory_capex_overage_year: Optional[int] = None
        # v1.1: warnings emitted during init (e.g. clamped mandatory back-chains)
        self._init_warnings: List[str] = []

        self.base_production = base_production if base_production is not None else np.zeros(n)
        self.base_water = base_water if base_water is not None else np.zeros(n)
        self.minimum_volumes = minimum_volumes if minimum_volumes is not None else np.zeros(n)
        self.daily_new_production: List[float] = []
        self.daily_total_production: List[float] = []
        self.daily_rig_use: List[int] = []
        self.daily_frac_use: List[int] = []
        self.daily_land_use: List[int] = []
        self.daily_permit_use: List[int] = []
        self.daily_construction_use: List[int] = []
        self.daily_fcf: List[float] = []    # daily free cashflow tracking

        # ----- WATER MANAGEMENT STATE (v2.0) -----
        self.company_storage_bbl: float = 0.0
        self.thirdparty_storage_bbl: float = 0.0
        self.daily_water_produced_bbl: List[float] = []
        self.daily_water_rainfall_bbl: List[float] = []
        self.daily_water_to_frac_bbl: List[float] = []
        self.daily_water_to_company_storage_bbl: List[float] = []
        self.daily_water_to_thirdparty_storage_bbl: List[float] = []
        self.daily_water_sharing_bbl: List[float] = []
        self.daily_water_select_rail_bbl: List[float] = []
        self.daily_water_pa_swd_bbl: List[float] = []
        self.daily_water_remainder_bbl: List[float] = []
        self.daily_water_unhandled_bbl: List[float] = []
        self.daily_water_frac_shortfall_bbl: List[float] = []
        self.daily_water_cost_mm: List[float] = []
        self.daily_company_storage_bbl: List[float] = []
        self.daily_thirdparty_storage_bbl: List[float] = []

        self.log: List[str] = []

        # ---- Pre-reserve mandatory-pad capex (v1.1) -----------------------
        if getattr(self.config, "reserve_mandatory_capex", True):
            self._reserve_mandatory_pads()

    # =====================================================================
    # CAPEX BOOKKEEPING (v1.1)
    # =====================================================================
    def _get_year(self, day: int) -> int:
        return day // 365

    def _year_capex_limit(self, year: int) -> float:
        base = self.config.capex_limit_for_year(year)
        return base + self.config.capex_tolerance_mm

    def _year_ol_ms_limit(self, year: int) -> float:
        return self.config.ol_ms_limit_for_year(year)

    def _milestone_disbursement_schedule(
        self, start_day: int, duration_days: float, amt: float
    ) -> Dict[int, float]:
        """Return {year: $MM_in_that_year} for a hypothetical disbursement.

        Pure arithmetic, no state mutation. Honors `capex_disbursement` mode.
        Last-day remainder absorbs FP dust so the schedule sums to exactly `amt`.
        """
        if amt <= 0:
            return {}
        n_sim = self.config.simulation_days
        mode = getattr(self.config, "capex_disbursement", "even")
        if mode == "lump_start" or duration_days <= 0:
            day = max(0, min(n_sim - 1, int(start_day)))
            return {self._get_year(day): float(amt)}
        # "even": uniform $/day across [start, start+ceil(duration))
        dur = max(1, int(np.ceil(duration_days)))
        per_day = amt / dur
        sched: Dict[int, float] = {}
        accumulated = 0.0
        for i in range(dur):
            d = int(start_day) + i
            if d < 0 or d >= n_sim:
                continue
            piece = per_day if i < dur - 1 else (amt - accumulated)
            sched[self._get_year(d)] = sched.get(self._get_year(d), 0.0) + piece
            accumulated += piece
        # If start_day was negative or off the end, the partial sum may not
        # equal `amt`. That's fine — the remainder represents disbursement
        # that would have occurred outside the sim horizon.
        return sched

    def _can_admit(
        self, start_day: int, duration_days: float, amt: float, is_ol_ms: bool = False
    ) -> Tuple[bool, Optional[int]]:
        """Check whether a milestone with this start/duration/cost can be
        admitted without breaching ANY year's capex cap (incl. tolerance) or
        the OL/MS sub-budget. Returns (ok, blocking_year_or_None).
        """
        if amt <= 0:
            return True, None
        sched = self._milestone_disbursement_schedule(start_day, duration_days, amt)
        for yr, share in sched.items():
            if (self.annual_capex_spent.get(yr, 0.0) + share
                    > self._year_capex_limit(yr) + 1e-9):
                return False, yr
            if is_ol_ms:
                if (self.annual_ol_ms_spent.get(yr, 0.0) + share
                        > self._year_ol_ms_limit(yr) + 1e-9):
                    return False, yr
        return True, None

    # Back-compat shim (used by legacy code paths that pass `(day, amount, is_ol_ms)`).
    # Forwards to `_can_admit` using duration=0 (lump check).
    def _can_afford(self, day: int, amount: float, is_ol_ms: bool = False) -> bool:
        ok, _ = self._can_admit(day, 0, amount, is_ol_ms=is_ol_ms)
        return ok

    def _disburse(
        self, start_day: int, duration_days: float, amt: float, is_ol_ms: bool
    ) -> None:
        """Commit `amt` into the daily arrays, then refresh the annual rollups."""
        if amt <= 0:
            return
        n_sim = self.config.simulation_days
        mode = getattr(self.config, "capex_disbursement", "even")
        if mode == "lump_start" or duration_days <= 0:
            day = max(0, min(n_sim - 1, int(start_day)))
            self._daily_capex_planned[day] += amt
            if is_ol_ms:
                self._daily_ol_ms_planned[day] += amt
        else:
            dur = max(1, int(np.ceil(duration_days)))
            per_day = amt / dur
            accumulated = 0.0
            for i in range(dur):
                d = int(start_day) + i
                piece = per_day if i < dur - 1 else (amt - accumulated)
                accumulated += piece
                if d < 0 or d >= n_sim:
                    continue
                self._daily_capex_planned[d] += piece
                if is_ol_ms:
                    self._daily_ol_ms_planned[d] += piece
        self._refresh_annual_rollups()

    def _refresh_annual_rollups(self) -> None:
        """Rebuild annual_capex_spent / annual_ol_ms_spent from the daily arrays."""
        n = self.config.simulation_days
        if n == 0:
            return
        days = np.arange(n)
        years = days // 365
        # totals
        totals: Dict[int, float] = {}
        ol_ms: Dict[int, float] = {}
        # Vectorized aggregation via np.bincount.
        max_year = int(years.max()) if n > 0 else 0
        cap_sum = np.bincount(years, weights=self._daily_capex_planned, minlength=max_year + 1)
        olms_sum = np.bincount(years, weights=self._daily_ol_ms_planned, minlength=max_year + 1)
        for y, v in enumerate(cap_sum):
            if v > 0:
                totals[int(y)] = float(v)
        for y, v in enumerate(olms_sum):
            if v > 0:
                ol_ms[int(y)] = float(v)
        self.annual_capex_spent = totals
        self.annual_ol_ms_spent = ol_ms

    def _schedule_disbursement(
        self, pad: WellPad, milestone: str, start_day: int, duration_days: float
    ) -> None:
        """Public-by-convention wrapper used by the daily loop.

        Replaces the legacy `_schedule_midpoint_capex`. Skips disbursement when
        the pad's mandatory capex was pre-reserved at simulator init.
        """
        amt = self._milestone_cost(pad, milestone)
        if amt <= 0:
            return
        if getattr(pad, "_mandatory_reserved", False):
            return  # already booked in _reserve_mandatory_pads
        is_ol_ms = milestone in ("midstream", "overland")
        self._disburse(start_day, duration_days, amt, is_ol_ms=is_ol_ms)

    # Legacy alias kept for any external callers; same behavior as
    # `_schedule_disbursement`.
    def _schedule_midpoint_capex(self, pad: WellPad, milestone: str,
                                  start_day: int, duration_days: float):
        self._schedule_disbursement(pad, milestone, start_day, duration_days)

    # No-op replacement for the old per-day pending-capex processor. Kept so
    # external test code that calls it doesn't crash.
    def _process_pending_capex(self, day: int):
        return None

    def _record_capex(self, pad: WellPad, milestone: str, day: int):
        """Legacy single-day record; routes through _disburse with duration=0."""
        amt = self._milestone_cost(pad, milestone)
        if amt > 0:
            is_ol_ms = milestone in ("midstream", "overland")
            self._disburse(day, 0, amt, is_ol_ms=is_ol_ms)

    # =====================================================================
    # MANDATORY-PAD PRE-RESERVATION (v1.1, Option C)
    # =====================================================================
    def _reserve_mandatory_pads(self) -> None:
        """Back-chain mandatory pads from `mandatory_start_day` and commit
        their capex into the daily arrays before the GA scheduling loop runs.

        This pass commits CAPEX ONLY — crew availability is not modeled here.
        If the GA orders non-mandatory pads onto the same crews/days the
        mandatory pad needs, the mandatory pad's actual milestone start will
        slip in the daily loop, and `_evaluate_ordering` will hard-fail the
        ordering as a mandatory_violation.

        If a back-solved milestone start falls before day 0, we clamp
        `land_start = 0` and chain subsequent milestones forward; the pad's
        achievable first production day will then be later than its
        configured mandatory date. A warning is logged.

        Validation: if any year's pre-reservation total exceeds
        cap + tolerance, `mandatory_capex_overage_mm` is set so that every
        GA ordering is reported infeasible — but the simulator still
        completes for diagnostic purposes.
        """
        n_sim = self.config.simulation_days
        mandatory = [p for p in self.pads
                     if p.is_mandatory and p.mandatory_start_day is not None]
        mandatory.sort(key=lambda p: int(p.mandatory_start_day or 0))

        for pad in mandatory:
            mand = int(pad.mandatory_start_day)
            land_dur   = int(np.ceil(pad.land_owner_agreement_days))
            permit_dur = int(np.ceil(pad.pad_permit_days))
            con_dur    = int(np.ceil(pad.pad_construction_days))
            drill_dur  = int(np.ceil(pad.drill_days))
            frac_dur   = int(np.ceil(pad.frac_days))
            ms_dur     = int(np.ceil(pad.midstream_construction_days))
            ol_dur     = int(np.ceil(pad.overland_construction_days))

            # Back-solve: pad must reach PRODUCING by `mand`.
            # Chain (serial): land -> permit -> pad_con -> drill -> frac
            # Parallel: midstream/overland start after pad_con and must finish before
            # the day production begins (frac_end). Conservative: assume midstream
            # & overland start the day pad_con ends and finish before frac_end.
            frac_end = mand
            frac_start = frac_end - frac_dur
            drill_end = frac_start
            drill_start = drill_end - drill_dur
            con_end = drill_start
            con_start = con_end - con_dur
            permit_end = con_start
            permit_start = permit_end - permit_dur
            land_end = permit_start
            land_start = land_end - land_dur

            # Clamp if the chain runs off the front of the simulation.
            if land_start < 0:
                shift = -land_start
                land_start = 0
                land_end = land_start + land_dur
                permit_start = land_end
                permit_end = permit_start + permit_dur
                con_start = permit_end
                con_end = con_start + con_dur
                drill_start = con_end
                drill_end = drill_start + drill_dur
                frac_start = drill_end
                frac_end = frac_start + frac_dur
                msg = (f"[WARNING] Pad {pad.name}: mandatory_start_day={mand} not "
                       f"achievable from day 0 (back-chain length={shift + frac_dur} "
                       f"days); first production = day {frac_end + 1}.")
                self._init_warnings.append(msg)
                print(msg)

            # Midstream/overland: start at con_end, must finish by frac_end.
            # If their durations exceed (frac_end - con_end), they will run
            # past frac_end and the pad's actual first-production day will be
            # max(frac_end, con_end + max(ms_dur, ol_dur)). For pre-reservation
            # purposes we just commit their disbursement starting at con_end.
            ms_start = con_end
            ol_start = con_end

            # Commit each milestone's disbursement.
            self._disburse(land_start,   pad.land_owner_agreement_days,
                           pad.capex_land_mm,             is_ol_ms=False)
            self._disburse(permit_start, pad.pad_permit_days,
                           pad.capex_permit_mm,           is_ol_ms=False)
            self._disburse(con_start,    pad.pad_construction_days,
                           pad.capex_pad_construction_mm, is_ol_ms=False)
            self._disburse(drill_start,  pad.drill_days,
                           pad.capex_drill_mm,            is_ol_ms=False)
            self._disburse(frac_start,   pad.frac_days,
                           pad.capex_frac_mm,             is_ol_ms=False)
            self._disburse(ms_start,     pad.midstream_construction_days,
                           pad.capex_midstream_mm,        is_ol_ms=True)
            self._disburse(ol_start,     pad.overland_construction_days,
                           pad.capex_overland_mm,         is_ol_ms=True)

            # Stash the back-chained start days so reporting (get_capex_timeline,
            # per-pad-monthly CSV) can attribute capex to the day it was ACTUALLY
            # committed in `_daily_capex_planned`, not the day the daily loop later
            # marks the milestone as starting (which can differ for mandatory pads).
            pad._reserved_starts = {
                "land":             int(land_start),
                "permit":           int(permit_start),
                "pad_construction": int(con_start),
                "midstream":        int(ms_start),
                "overland":         int(ol_start),
                "drill":            int(drill_start),
                "frac":             int(frac_start),
            }
            pad._mandatory_reserved = True

        # Validate: did the pre-reservation alone exceed any year's cap?
        for yr, spent in self.annual_capex_spent.items():
            limit = self._year_capex_limit(yr)
            if spent > limit + 1e-6:
                over = spent - limit
                if over > self.mandatory_capex_overage_mm:
                    self.mandatory_capex_overage_mm = float(over)
                    self.mandatory_capex_overage_year = int(yr)
        if self.mandatory_capex_overage_mm > 0:
            print(f"[WARNING] Mandatory-pad pre-reservation exceeds capex cap by "
                  f"${self.mandatory_capex_overage_mm:,.2f}MM in year "
                  f"{self.mandatory_capex_overage_year}. Every GA ordering will "
                  f"be reported infeasible until either the cap is raised, "
                  f"mandatory dates are pushed, or `enforce_capex_budget=False`.")

    def _should_defer_production(self, pad: WellPad, day: int) -> bool:
        """Return True if this pad's first-production should be held one more day
        because of constrained water takeaway.

        SAME-DAY PREDICTIVE: forecasts today's water balance assuming this pad
        were brought online (i.e. starts contributing day-1 water tomorrow,
        but treated as a same-day signal). Forecast =
            existing producers' water + base water + rainfall
            + this pad's day-1 water
            − frac demand
            − total outlet takeaway capacity
            − remaining storage headroom (co + 3rd-party).
        If the projected overflow exceeds ``water_deferral_trigger_bbl`` the
        pad is held. Cap: a single pad can only be deferred for
        ``water_deferral_max_days`` cumulative days; after that it is forced
        online. Mandatory pads are never deferred.
        """
        cfg = self.config
        if not getattr(cfg, "water_deferral_enabled", False):
            return False
        if pad.is_mandatory:
            return False
        if pad.production_deferred_days >= cfg.water_deferral_max_days:
            return False
        if not getattr(cfg, "water_enabled", False):
            return False

        # ---- Same-day predictive water balance ----------------------------
        base_water_today = float(self.base_water[day]) if day < len(self.base_water) else 0.0
        existing_producers_water = sum(p.water_production_at_day(day) for p in self.pads)
        # Hypothetical day-1 contribution from this pad (peak rate at t=0).
        new_pad_day1_water = sum(w.water_rate_at_time(0) for w in pad.wells)
        rainfall = float(getattr(cfg, "rainfall_bwpd", 0.0))
        supply = (existing_producers_water + base_water_today
                  + new_pad_day1_water + rainfall)

        frac_demand = sum(
            p.frac_water_demand_bwpd()
            for p in self.pads if p.status == PadStatus.FRACKING
        )
        net_supply = max(0.0, supply - frac_demand)

        outlet_capacity = sum(
            getattr(cfg, cap_attr) for _k, _l, cap_attr, _c in WATER_OUTLET_CASCADE
        )
        storage_headroom = (
            max(0.0, cfg.storage_company_capacity_bbl - self.company_storage_bbl)
            + max(0.0, cfg.storage_thirdparty_capacity_bbl - self.thirdparty_storage_bbl)
        )

        projected_overflow = net_supply - outlet_capacity - storage_headroom
        return projected_overflow > cfg.water_deferral_trigger_bbl

    def _milestone_cost(self, pad: WellPad, milestone: str) -> float:
        return {"land": pad.capex_land_mm, "permit": pad.capex_permit_mm,
                "pad_construction": pad.capex_pad_construction_mm,
                "midstream": pad.capex_midstream_mm, "overland": pad.capex_overland_mm,
                "drill": pad.capex_drill_mm, "frac": pad.capex_frac_mm}.get(milestone, 0.0)

    def run(self) -> pd.DataFrame:
        for day in range(self.config.simulation_days):

            for pad in self.pads:
                if (pad.status == PadStatus.LAND_AGREEMENT and
                        pad.land_end is not None and day >= pad.land_end):
                    pad.status = PadStatus.WAITING_PERMIT
                    self.land_crews_in_use -= 1
                if (pad.status == PadStatus.PERMITTING and
                        pad.permit_end is not None and day >= pad.permit_end):
                    pad.status = PadStatus.WAITING_PAD_CON
                    self.permit_crews_in_use -= 1
                if (pad.status == PadStatus.PAD_CONSTRUCTION and
                        pad.pad_con_end is not None and day >= pad.pad_con_end):
                    self.construction_crews_in_use -= 1
                    pad.status = PadStatus.WAITING_DRILL
                if (pad.status == PadStatus.DRILLING and
                        pad.drill_end is not None and day >= pad.drill_end):
                    pad.status = PadStatus.WAITING_FRAC
                    self.rigs_in_use -= 1
                if (pad.status == PadStatus.FRACKING and
                        pad.frac_end is not None and day >= pad.frac_end):
                    self.frac_crews_in_use -= 1
                    all_done = (pad.midstream_end is not None and day >= pad.midstream_end and
                                pad.overland_end is not None and day >= pad.overland_end)
                    if all_done and not self._should_defer_production(pad, day):
                        pad.status = PadStatus.PRODUCING
                        pad.first_production_day = day + 1
                    else:
                        pad.status = PadStatus.WAITING_PRODUCTION
                        if all_done:
                            pad.production_deferred_days += 1
                if (pad.status == PadStatus.WAITING_PRODUCTION and
                        pad.frac_end is not None and day >= pad.frac_end and
                        pad.midstream_end is not None and day >= pad.midstream_end and
                        pad.overland_end is not None and day >= pad.overland_end):
                    if self._should_defer_production(pad, day):
                        pad.production_deferred_days += 1
                    else:
                        pad.status = PadStatus.PRODUCING
                        pad.first_production_day = day + 1

            # Parallel milestones: midstream & overland
            for pad in self.pads:
                if pad.pad_con_end is not None and day >= pad.pad_con_end:
                    if pad.midstream_start is None:
                        c = self._milestone_cost(pad, "midstream")
                        dur_ms = pad.midstream_construction_days
                        if c == 0 or self._can_admit(day, dur_ms, c, is_ol_ms=True)[0]:
                            pad.midstream_start = day
                            pad.midstream_end = day + int(np.ceil(dur_ms))
                            if c > 0:
                                self._schedule_disbursement(pad, "midstream", day, dur_ms)
                    if pad.overland_start is None:
                        c = self._milestone_cost(pad, "overland")
                        dur_ol = pad.overland_construction_days
                        if c == 0 or self._can_admit(day, dur_ol, c, is_ol_ms=True)[0]:
                            pad.overland_start = day
                            pad.overland_end = day + int(np.ceil(dur_ol))
                            if c > 0:
                                self._schedule_disbursement(pad, "overland", day, dur_ol)

            # Serial milestones: land → permit → pad construction
            for status, crew_attr, max_attr, milestone, end_calc, dur_attr in [
                (PadStatus.WAITING_LAND, "land_crews_in_use", "num_land_crews",
                 "land", "land", "land_owner_agreement_days"),
                (PadStatus.WAITING_PERMIT, "permit_crews_in_use", "num_permit_crews",
                 "permit", "permit", "pad_permit_days"),
                (PadStatus.WAITING_PAD_CON, "construction_crews_in_use", "num_construction_crews",
                 "pad_construction", "pad_con", "pad_construction_days"),
            ]:
                waiting = [p for p in self.pads if p.status == status]
                waiting.sort(key=lambda p: p.sequence_number)
                for pad in waiting:
                    if getattr(self, crew_attr) >= getattr(self.config, max_attr):
                        break
                    cost = self._milestone_cost(pad, milestone)
                    dur = getattr(pad, dur_attr)
                    if cost > 0 and not self._can_admit(day, dur, cost)[0]:
                        continue
                    if end_calc == "land":
                        pad.status = PadStatus.LAND_AGREEMENT
                        pad.land_start = day
                        pad.land_end = day + int(np.ceil(dur))
                    elif end_calc == "permit":
                        pad.status = PadStatus.PERMITTING
                        pad.permit_start = day
                        pad.permit_end = day + int(np.ceil(dur))
                    elif end_calc == "pad_con":
                        pad.status = PadStatus.PAD_CONSTRUCTION
                        pad.pad_con_start = day
                        pad.pad_con_end = day + int(np.ceil(dur))
                    setattr(self, crew_attr, getattr(self, crew_attr) + 1)
                    self._schedule_disbursement(pad, milestone, day, dur)

            # Frac assignment
            wf = [p for p in self.pads if p.status == PadStatus.WAITING_FRAC
                  and p.drill_end is not None and day >= p.drill_end
                  and p.overland_end is not None and day >= p.overland_end]
            wf.sort(key=lambda p: p.drill_end)
            for pad in wf:
                if self.frac_crews_in_use >= self.config.num_frac_crews:
                    break
                cost = self._milestone_cost(pad, "frac")
                if cost > 0 and not self._can_admit(day, pad.frac_days, cost)[0]:
                    continue
                pad.status = PadStatus.FRACKING
                pad.frac_start = day
                pad.frac_end = day + int(np.ceil(pad.frac_days))
                self.frac_crews_in_use += 1
                self._schedule_disbursement(pad, "frac", day, pad.frac_days)

            # Drill assignment
            wd = [p for p in self.pads if p.status == PadStatus.WAITING_DRILL]
            wd.sort(key=lambda p: p.sequence_number)
            for pad in wd:
                if self.rigs_in_use >= self.config.num_rigs:
                    break
                cost = self._milestone_cost(pad, "drill")
                if cost > 0 and not self._can_admit(day, pad.drill_days, cost)[0]:
                    continue
                pad.status = PadStatus.DRILLING
                pad.drill_start = day
                pad.drill_end = day + int(np.ceil(pad.drill_days))
                self.rigs_in_use += 1
                self._schedule_disbursement(pad, "drill", day, pad.drill_days)

            # Pad entry
            for pad in self.pads:
                if (pad.status == PadStatus.WAITING and pad.is_mandatory
                        and pad.mandatory_start_day == day):
                    pad.status = PadStatus.WAITING_LAND
            while self.next_pad_index < len(self.pads):
                pad = self.pads[self.next_pad_index]
                if pad.status != PadStatus.WAITING:
                    self.next_pad_index += 1
                    continue
                if pad.earliest_start_day > day:
                    break
                if pad.is_mandatory and pad.mandatory_start_day > day:
                    self.next_pad_index += 1
                    continue
                pad.status = PadStatus.WAITING_LAND
                self.next_pad_index += 1
                break

            # Production & FCF
            nr = sum(p.production_at_day(day) for p in self.pads)
            br = float(self.base_production[day]) if day < len(self.base_production) else 0.0
            total_prod = br + nr
            self.daily_new_production.append(nr)
            self.daily_total_production.append(total_prod)
            self.daily_rig_use.append(self.rigs_in_use)
            self.daily_frac_use.append(self.frac_crews_in_use)
            self.daily_land_use.append(self.land_crews_in_use)
            self.daily_permit_use.append(self.permit_crews_in_use)
            self.daily_construction_use.append(self.construction_crews_in_use)

            # ============================================================
            # WATER MANAGEMENT (v2.0)
            # ============================================================
            water_cost_mm_today = 0.0
            water_produced_today = 0.0
            water_rainfall_today = 0.0
            water_to_frac_today = 0.0
            water_to_company_storage_today = 0.0
            water_to_thirdparty_storage_today = 0.0
            water_sharing_today = 0.0
            water_select_rail_today = 0.0
            water_pa_swd_today = 0.0
            water_remainder_today = 0.0
            water_unhandled_today = 0.0
            water_frac_shortfall_today = 0.0

            if self.config.water_enabled:
                cfg = self.config
                base_water_today = float(self.base_water[day]) if day < len(self.base_water) else 0.0
                water_produced_today = sum(p.water_production_at_day(day) for p in self.pads) + base_water_today
                water_rainfall_today = float(cfg.rainfall_bwpd)
                water_supply = water_produced_today + water_rainfall_today

                frac_demand_today = sum(
                    p.frac_water_demand_bwpd()
                    for p in self.pads if p.status == PadStatus.FRACKING
                )

                if frac_demand_today > 0:
                    use_from_supply = min(water_supply, frac_demand_today)
                    water_to_frac_today += use_from_supply
                    water_supply -= use_from_supply
                    remaining_frac_demand = frac_demand_today - use_from_supply

                    if remaining_frac_demand > 0 and self.company_storage_bbl > 0:
                        draw = min(remaining_frac_demand, self.company_storage_bbl)
                        self.company_storage_bbl -= draw
                        water_to_frac_today += draw
                        remaining_frac_demand -= draw

                    if remaining_frac_demand > 0 and self.thirdparty_storage_bbl > 0:
                        draw = min(remaining_frac_demand, self.thirdparty_storage_bbl)
                        self.thirdparty_storage_bbl -= draw
                        water_to_frac_today += draw
                        remaining_frac_demand -= draw

                    water_frac_shortfall_today = max(0.0, remaining_frac_demand)
                    water_cost_mm_today += water_to_frac_today * cfg.water_to_frac_cost_per_bbl / 1e6

                if water_supply > 0:
                    company_room = max(0.0, cfg.storage_company_capacity_bbl - self.company_storage_bbl)
                    fill_co = min(water_supply, company_room)
                    if fill_co > 0:
                        self.company_storage_bbl += fill_co
                        water_to_company_storage_today += fill_co
                        water_cost_mm_today += fill_co * cfg.storage_company_cost_per_bbl / 1e6
                        water_supply -= fill_co

                if water_supply > 0:
                    thirdparty_room = max(0.0, cfg.storage_thirdparty_capacity_bbl - self.thirdparty_storage_bbl)
                    fill_tp = min(water_supply, thirdparty_room)
                    if fill_tp > 0:
                        self.thirdparty_storage_bbl += fill_tp
                        water_to_thirdparty_storage_today += fill_tp
                        water_cost_mm_today += fill_tp * cfg.storage_thirdparty_cost_per_bbl / 1e6
                        water_supply -= fill_tp

                # ---- Outlet cascade (water_sharing → select_rail → pa_swd → remainder)
                outlet_takes: Dict[str, float] = {key: 0.0 for key, *_ in WATER_OUTLET_CASCADE}
                for key, _label, cap_attr, cost_attr in WATER_OUTLET_CASCADE:
                    if water_supply <= 0:
                        break
                    take = min(water_supply, getattr(cfg, cap_attr))
                    outlet_takes[key] = take
                    water_cost_mm_today += take * getattr(cfg, cost_attr) / 1e6
                    water_supply -= take
                water_sharing_today    = outlet_takes["water_sharing"]
                water_select_rail_today = outlet_takes["select_rail"]
                water_pa_swd_today     = outlet_takes["pa_swd"]
                water_remainder_today  = outlet_takes["remainder"]

                water_unhandled_today = max(0.0, water_supply)

            self.daily_water_produced_bbl.append(water_produced_today)
            self.daily_water_rainfall_bbl.append(water_rainfall_today)
            self.daily_water_to_frac_bbl.append(water_to_frac_today)
            self.daily_water_to_company_storage_bbl.append(water_to_company_storage_today)
            self.daily_water_to_thirdparty_storage_bbl.append(water_to_thirdparty_storage_today)
            self.daily_water_sharing_bbl.append(water_sharing_today)
            self.daily_water_select_rail_bbl.append(water_select_rail_today)
            self.daily_water_pa_swd_bbl.append(water_pa_swd_today)
            self.daily_water_remainder_bbl.append(water_remainder_today)
            self.daily_water_unhandled_bbl.append(water_unhandled_today)
            self.daily_water_frac_shortfall_bbl.append(water_frac_shortfall_today)
            self.daily_water_cost_mm.append(water_cost_mm_today)
            self.daily_company_storage_bbl.append(self.company_storage_bbl)
            self.daily_thirdparty_storage_bbl.append(self.thirdparty_storage_bbl)

            # Daily FCF: revenue - opex - capex
            # NOTE: water cost is intentionally EXCLUDED here. It is tracked
            # separately in `daily_water_cost_mm` and subtracted from the GA
            # objective as `pv_water_cost_mm`. Including it here would double-
            # count water in the fitness score.
            revenue_mm = total_prod * self.config.commodity_price_per_mcf / 1e6
            daily_opex_mm = sum(
                p.annual_opex_mm / 365.0
                for p in self.pads if p.first_production_day is not None and day >= p.first_production_day
            )
            day_capex_mm = float(self._daily_capex_planned[day]) if day < len(self._daily_capex_planned) else 0.0
            self.daily_fcf.append(revenue_mm - daily_opex_mm - day_capex_mm)

        n = self.config.simulation_days
        dd = pd.DataFrame({
            "day": range(n), "month": [d // 30 for d in range(n)],
            "year": [d // 365 for d in range(n)],
            "new_mcfd": self.daily_new_production,
            "base_mcfd": [float(self.base_production[d]) if d < len(self.base_production)
                          else 0.0 for d in range(n)],
            "total_mcfd": self.daily_total_production,
            "min_mcfd": [float(self.minimum_volumes[d]) if d < len(self.minimum_volumes)
                         else 0.0 for d in range(n)],
            "rigs": self.daily_rig_use, "frac": self.daily_frac_use,
            "land": self.daily_land_use, "permit": self.daily_permit_use,
            "con": self.daily_construction_use,
            "daily_fcf_mm": self.daily_fcf,
            "water_produced_bbl": self.daily_water_produced_bbl,
            "water_rainfall_bbl": self.daily_water_rainfall_bbl,
            "water_to_frac_bbl": self.daily_water_to_frac_bbl,
            "water_to_co_storage_bbl": self.daily_water_to_company_storage_bbl,
            "water_to_tp_storage_bbl": self.daily_water_to_thirdparty_storage_bbl,
            "water_sharing_bbl": self.daily_water_sharing_bbl,
            "water_select_rail_bbl": self.daily_water_select_rail_bbl,
            "water_pa_swd_bbl": self.daily_water_pa_swd_bbl,
            "water_remainder_bbl": self.daily_water_remainder_bbl,
            "water_unhandled_bbl": self.daily_water_unhandled_bbl,
            "water_frac_shortfall_bbl": self.daily_water_frac_shortfall_bbl,
            "water_cost_mm": self.daily_water_cost_mm,
            "co_storage_bbl": self.daily_company_storage_bbl,
            "tp_storage_bbl": self.daily_thirdparty_storage_bbl,
        })
        dd["shortfall"] = np.maximum(0, dd["min_mcfd"] - dd["total_mcfd"])
        m = dd.groupby("month").agg(
            avg_new_mcfd=("new_mcfd", "mean"), avg_base_mcfd=("base_mcfd", "mean"),
            avg_total_mcfd=("total_mcfd", "mean"), avg_min_vol_mcfd=("min_mcfd", "mean"),
            avg_shortfall_mcfd=("shortfall", "mean"),
            monthly_volume_mcf=("total_mcfd", "sum"),
            monthly_fcf_mm=("daily_fcf_mm", "sum"),
            avg_rigs=("rigs", "mean"), avg_frac=("frac", "mean"),
            avg_land=("land", "mean"), avg_permit=("permit", "mean"),
            avg_con=("con", "mean"), max_day=("day", "max"),
            water_produced_bbl=("water_produced_bbl", "sum"),
            water_rainfall_bbl=("water_rainfall_bbl", "sum"),
            water_to_frac_bbl=("water_to_frac_bbl", "sum"),
            water_to_co_storage_bbl=("water_to_co_storage_bbl", "sum"),
            water_to_tp_storage_bbl=("water_to_tp_storage_bbl", "sum"),
            water_sharing_bbl=("water_sharing_bbl", "sum"),
            water_select_rail_bbl=("water_select_rail_bbl", "sum"),
            water_pa_swd_bbl=("water_pa_swd_bbl", "sum"),
            water_remainder_bbl=("water_remainder_bbl", "sum"),
            water_unhandled_bbl=("water_unhandled_bbl", "sum"),
            water_frac_shortfall_bbl=("water_frac_shortfall_bbl", "sum"),
            water_cost_mm=("water_cost_mm", "sum"),
            avg_co_storage_bbl=("co_storage_bbl", "mean"),
            avg_tp_storage_bbl=("tp_storage_bbl", "mean"),
        ).reset_index()
        m["year"] = m["max_day"] // 365
        c = self.config
        m["capex_budget_mm"] = m["year"].apply(lambda yr: c.capex_limit_for_year(yr))
        m["prod_ceiling_mcfd"] = m["year"].apply(lambda yr: c.production_limit_for_year(yr))
        m["rig_util"] = m["avg_rigs"] / c.num_rigs
        m["frac_util"] = m["avg_frac"] / c.num_frac_crews if c.num_frac_crews > 0 else 0
        m["land_util"] = m["avg_land"] / c.num_land_crews if c.num_land_crews > 0 else 0
        m["permit_util"] = m["avg_permit"] / c.num_permit_crews if c.num_permit_crews > 0 else 0
        m["con_util"] = m["avg_con"] / c.num_construction_crews if c.num_construction_crews > 0 else 0
        m["cumulative_mcf"] = m["monthly_volume_mcf"].cumsum()
        m["cumulative_fcf_mm"] = m["monthly_fcf_mm"].cumsum()
        m["cumulative_water_cost_mm"] = m["water_cost_mm"].cumsum()
        return m

    def check_minimum_volumes(self, monthly: pd.DataFrame,
                              tolerance: float = 0.0) -> Tuple[bool, pd.DataFrame]:
        active = monthly[monthly["avg_min_vol_mcfd"] > 0].copy()
        if active.empty:
            return True, pd.DataFrame()
        failing = active[active["avg_shortfall_mcfd"] > tolerance].copy()
        return failing.empty, failing

    def check_production_ceiling(self, monthly: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        if self.config.production_ceiling_mcfd <= 0:
            return True, pd.DataFrame()
        yearly = monthly.groupby("year").agg(
            avg_total_mcfd=("avg_total_mcfd", "mean")).reset_index()
        yearly["ceiling_mcfd"] = yearly["year"].apply(
            lambda yr: self.config.production_limit_for_year(yr))
        yearly["over_ceiling"] = yearly["avg_total_mcfd"] - yearly["ceiling_mcfd"]
        violations = yearly[yearly["over_ceiling"] > 0].copy()
        return violations.empty, violations

    def get_pad_order(self) -> List[str]:
        return [p.name for p in self.pads]

    def get_capex_summary(self) -> Dict[int, Dict]:
        s = {}
        for yr, sp in sorted(self.annual_capex_spent.items()):
            budget = self.config.capex_limit_for_year(yr)
            ceiling = budget + self.config.capex_tolerance_mm
            ol_ms_sp = self.annual_ol_ms_spent.get(yr, 0.0)
            ol_ms_lim = self.config.ol_ms_limit_for_year(yr)
            s[yr] = {"spent": sp, "budget": budget, "ceiling": ceiling,
                     "over_budget": max(0, sp - budget),
                     "over_ceiling": max(0, sp - ceiling),
                     "ol_ms_spent": ol_ms_sp, "ol_ms_limit": ol_ms_lim,
                     "ol_ms_over": max(0, ol_ms_sp - ol_ms_lim) if ol_ms_lim != float("inf") else 0}
        return s

    def get_fcf_summary(self) -> Dict[int, Dict]:
        yearly_fcf_rev: Dict[int, float] = {}
        yearly_opex: Dict[int, float] = {}
        yearly_water: Dict[int, float] = {}
        n = self.config.simulation_days
        for day in range(n):
            yr = day // 365
            total_prod = self.daily_total_production[day] if day < len(self.daily_total_production) else 0
            rev = total_prod * self.config.commodity_price_per_mcf / 1e6
            opex = sum(p.annual_opex_mm / 365.0 for p in self.pads
                       if p.first_production_day is not None and day >= p.first_production_day)
            wc = self.daily_water_cost_mm[day] if day < len(self.daily_water_cost_mm) else 0
            yearly_fcf_rev[yr] = yearly_fcf_rev.get(yr, 0.0) + rev
            yearly_opex[yr] = yearly_opex.get(yr, 0.0) + opex
            yearly_water[yr] = yearly_water.get(yr, 0.0) + wc
        s = {}
        all_years = set(list(yearly_fcf_rev.keys()) + list(self.annual_capex_spent.keys()))
        for yr in sorted(all_years):
            rev = yearly_fcf_rev.get(yr, 0.0)
            opex = yearly_opex.get(yr, 0.0)
            capex = self.annual_capex_spent.get(yr, 0.0)
            water = yearly_water.get(yr, 0.0)
            s[yr] = {"revenue_mm": rev, "opex_mm": opex, "capex_mm": capex,
                     "water_cost_mm": water,
                     "fcf_mm": rev - opex - capex - water}
        return s

    def get_water_summary(self) -> Dict[int, Dict]:
        out: Dict[int, Dict] = {}
        n = self.config.simulation_days
        for day in range(n):
            yr = day // 365
            row = out.setdefault(yr, {
                "produced_bbl": 0.0, "rainfall_bbl": 0.0,
                "to_frac_bbl": 0.0,
                "to_co_storage_bbl": 0.0, "to_tp_storage_bbl": 0.0,
                "water_sharing_bbl": 0.0, "select_rail_bbl": 0.0, "pa_swd_bbl": 0.0,
                "remainder_bbl": 0.0,
                "unhandled_bbl": 0.0, "frac_shortfall_bbl": 0.0,
                "cost_mm": 0.0,
            })
            row["produced_bbl"] += self.daily_water_produced_bbl[day] if day < len(self.daily_water_produced_bbl) else 0
            row["rainfall_bbl"] += self.daily_water_rainfall_bbl[day] if day < len(self.daily_water_rainfall_bbl) else 0
            row["to_frac_bbl"] += self.daily_water_to_frac_bbl[day] if day < len(self.daily_water_to_frac_bbl) else 0
            row["to_co_storage_bbl"] += self.daily_water_to_company_storage_bbl[day] if day < len(self.daily_water_to_company_storage_bbl) else 0
            row["to_tp_storage_bbl"] += self.daily_water_to_thirdparty_storage_bbl[day] if day < len(self.daily_water_to_thirdparty_storage_bbl) else 0
            row["water_sharing_bbl"] += self.daily_water_sharing_bbl[day] if day < len(self.daily_water_sharing_bbl) else 0
            row["select_rail_bbl"] += self.daily_water_select_rail_bbl[day] if day < len(self.daily_water_select_rail_bbl) else 0
            row["pa_swd_bbl"] += self.daily_water_pa_swd_bbl[day] if day < len(self.daily_water_pa_swd_bbl) else 0
            row["remainder_bbl"] += self.daily_water_remainder_bbl[day] if day < len(self.daily_water_remainder_bbl) else 0
            row["unhandled_bbl"] += self.daily_water_unhandled_bbl[day] if day < len(self.daily_water_unhandled_bbl) else 0
            row["frac_shortfall_bbl"] += self.daily_water_frac_shortfall_bbl[day] if day < len(self.daily_water_frac_shortfall_bbl) else 0
            row["cost_mm"] += self.daily_water_cost_mm[day] if day < len(self.daily_water_cost_mm) else 0
        return out

    def get_water_timeline(self) -> pd.DataFrame:
        n = self.config.simulation_days
        return pd.DataFrame({
            "day": range(n),
            "month": [d // 30 for d in range(n)],
            "year": [d // 365 for d in range(n)],
            "produced_bbl": self.daily_water_produced_bbl,
            "rainfall_bbl": self.daily_water_rainfall_bbl,
            "to_frac_bbl": self.daily_water_to_frac_bbl,
            "to_co_storage_bbl": self.daily_water_to_company_storage_bbl,
            "to_tp_storage_bbl": self.daily_water_to_thirdparty_storage_bbl,
            "water_sharing_bbl": self.daily_water_sharing_bbl,
            "select_rail_bbl": self.daily_water_select_rail_bbl,
            "pa_swd_bbl": self.daily_water_pa_swd_bbl,
            "remainder_bbl": self.daily_water_remainder_bbl,
            "unhandled_bbl": self.daily_water_unhandled_bbl,
            "frac_shortfall_bbl": self.daily_water_frac_shortfall_bbl,
            "cost_mm": self.daily_water_cost_mm,
            "co_storage_bbl": self.daily_company_storage_bbl,
            "tp_storage_bbl": self.daily_thirdparty_storage_bbl,
        })

    def get_schedule(self) -> pd.DataFrame:
        recs = []
        for p in self.pads:
            recs.append({
                "seq": p.sequence_number, "pad_id": p.pad_id, "name": p.name,
                "pvi": p.pvi, "npv_mm": p.npv_mm, "annual_opex_mm": p.annual_opex_mm,
                "status": p.status.name, "num_wells": p.num_wells,
                "total_capex_mm": p.total_capex_mm, "pad_qi_mcfd": p.total_qi_mcfd,
                "cycle_days": p.total_cycle_days, "mandatory_day": p.mandatory_start_day,
                "land_start": p.land_start, "land_end": p.land_end,
                "permit_start": p.permit_start, "permit_end": p.permit_end,
                "pad_con_start": p.pad_con_start, "pad_con_end": p.pad_con_end,
                "predrill_start": p.predrill_start, "predrill_end": p.predrill_end,
                "midstream_start": p.midstream_start, "midstream_end": p.midstream_end,
                "overland_start": p.overland_start, "overland_end": p.overland_end,
                "drill_start": p.drill_start, "drill_end": p.drill_end,
                "frac_start": p.frac_start, "frac_end": p.frac_end,
                "first_prod": p.first_production_day})
        return pd.DataFrame(recs)

    def get_well_production(self) -> pd.DataFrame:
        recs = []
        for p in self.pads:
            if p.first_production_day is None:
                continue
            for mi in range(self.config.simulation_days // 30 + 1):
                d = mi * 30
                if d >= self.config.simulation_days:
                    break
                for wn, r in p.well_production_at_day(d).items():
                    recs.append({"pad_name": p.name, "well_name": wn,
                                 "month": mi, "day": d, "production_mcfd": r})
        return pd.DataFrame(recs)

    def get_capex_timeline(self) -> pd.DataFrame:
        """Per-milestone capex disbursement records, binned by year/month.

        v1.1: respects ``capex_disbursement`` mode. With ``"even"``, a milestone
        is split into one record per (milestone, year) so a milestone that
        straddles a year boundary contributes to *both* years \u2014 matching what
        the simulator actually charges to ``_daily_capex_planned`` and what
        ``_can_admit`` enforces. With ``"lump_start"`` the full amount is
        booked on the start day (legacy behavior).
        """
        recs = []
        for p in self.pads:
            reserved = getattr(p, "_reserved_starts", None)
            milestones = [
                ("land",             p.capex_land_mm,             p.land_start,      p.land_owner_agreement_days),
                ("permit",           p.capex_permit_mm,           p.permit_start,    p.pad_permit_days),
                ("pad_construction", p.capex_pad_construction_mm, p.pad_con_start,   p.pad_construction_days),
                ("midstream",        p.capex_midstream_mm,        p.midstream_start, p.midstream_construction_days),
                ("overland",         p.capex_overland_mm,         p.overland_start,  p.overland_construction_days),
                ("drill",            p.capex_drill_mm,            p.drill_start,     p.drill_days),
                ("frac",             p.capex_frac_mm,             p.frac_start,      p.frac_days),
            ]
            for ms, amt, sd, dur in milestones:
                if amt <= 0:
                    continue
                # For mandatory pads, prefer the pre-reservation start day so that
                # this report matches `_daily_capex_planned` exactly.
                if reserved is not None and ms in reserved:
                    sd = reserved[ms]
                if sd is None:
                    continue
                schedule = self._milestone_disbursement_schedule(int(sd), float(dur), float(amt))
                # `schedule` is {year: $MM}. Pick a representative day-of-year
                # per record so downstream month-binning still works.
                for yr, yr_amt in sorted(schedule.items()):
                    if yr_amt <= 0:
                        continue
                    rep_day = max(int(sd), yr * 365)
                    recs.append({
                        "pad_name": p.name, "seq": p.sequence_number,
                        "milestone": ms, "capex_mm": float(yr_amt),
                        "day": rep_day, "month": rep_day // 30, "year": yr,
                    })
        return pd.DataFrame(recs)

    def print_summary(self):
        prod = [p for p in self.pads if p.status == PadStatus.PRODUCING]
        mand = [p for p in self.pads if p.is_mandatory]
        c = self.config
        print("=" * 80)
        print("SIMULATION SUMMARY")
        print("=" * 80)
        print(f"  Pads: {len(self.pads)}, Wells: {sum(p.num_wells for p in self.pads)}")
        print(f"  Producing: {len(prod)} ({sum(p.num_wells for p in prod)}w)")
        print(f"  Resources: {c.num_land_crews}L {c.num_permit_crews}P "
              f"{c.num_construction_crews}C {c.num_rigs}R {c.num_frac_crews}F")
        print(f"  Capex (base): ${c.annual_capex_limit_mm}MM + ${c.capex_tolerance_mm}MM tol")
        if c.capex_cagr_pct > 0:
            print(f"  Capex CAGR: {c.capex_cagr_pct}% for {c.capex_cagr_years} years")
        if c.overland_midstream_budget_mm > 0:
            print(f"  OL/MS Sub-budget: ${c.overland_midstream_budget_mm}MM")
        if c.production_ceiling_mcfd > 0:
            print(f"  Production Ceiling: {c.production_ceiling_mcfd:,.0f} MCFD "
                  f"({c.production_cagr_pct}% CAGR, {c.production_cagr_years} yr)")
        print(f"  Commodity Price: ${c.commodity_price_per_mcf}/MCF")
        if mand:
            print(f"\n  Mandatory ({len(mand)}):")
            for p in mand:
                print(f"    {p.name:20s} day {p.mandatory_start_day} (land: {p.land_start})")
        print("\n  Capex (CAGR-adjusted):")
        for yr, info in self.get_capex_summary().items():
            budget = info["budget"]
            pct = info["spent"] / budget * 100 if budget > 0 else 0
            flag = ""
            if info["over_ceiling"] > 0:
                flag = f" ⚠️ +${info['over_ceiling']:.1f}MM"
            elif info["over_budget"] > 0:
                flag = f" (+${info['over_budget']:.1f}MM tol)"
            ol_ms_str = ""
            if c.overland_midstream_budget_mm > 0:
                ol_ms_str = f" | OL/MS: ${info['ol_ms_spent']:.1f}/${info['ol_ms_limit']:.0f}MM"
            print(f"    Yr {yr}: ${info['spent']:>7.1f}MM / "
                  f"${budget:.0f}MM ({pct:.0f}%){flag}{ol_ms_str}")
        fcf = self.get_fcf_summary()
        if fcf:
            print("\n  Free Cashflow (Revenue - Opex - Capex - Water):")
            for yr, info in fcf.items():
                print(f"    Yr {yr}: Rev ${info['revenue_mm']:>7.1f}MM "
                      f"- Opex ${info['opex_mm']:>6.1f}MM "
                      f"- Capex ${info['capex_mm']:>7.1f}MM "
                      f"- Water ${info['water_cost_mm']:>6.1f}MM "
                      f"= FCF ${info['fcf_mm']:>7.1f}MM")
        if self.config.water_enabled:
            ws = self.get_water_summary()
            if ws:
                print("\n  Water Management (BBL totals & cost):")
                for yr, w in ws.items():
                    print(f"    Yr {yr}: Prod {w['produced_bbl']/1e3:>7.1f}kbbl "
                          f"+Rain {w['rainfall_bbl']/1e3:>6.1f}kbbl "
                          f"→ Frac {w['to_frac_bbl']/1e3:>6.1f}kbbl "
                          f"CoStr {w['to_co_storage_bbl']/1e3:>6.1f}k "
                          f"TpStr {w['to_tp_storage_bbl']/1e3:>5.1f}k "
                          f"Out(WS/SR/SWD/R) {w['water_sharing_bbl']/1e3:>5.1f}/"
                          f"{w['select_rail_bbl']/1e3:>5.1f}/{w['pa_swd_bbl']/1e3:>5.1f}/"
                          f"{w['remainder_bbl']/1e3:>5.1f}k "
                          f"| Cost ${w['cost_mm']:>5.2f}MM")
        print("=" * 80)


# ----- 1g. Single-simulation entry point ------------------------------------

def run_single_simulation(
    config: SimConfig, base_production: np.ndarray, minimum_volumes: np.ndarray,
    pad_order: Optional[List[str]] = None, label: str = "run",
    template_pads: Optional[List[WellPad]] = None,
    global_overwrites: Optional[Dict] = None,
    base_water: Optional[np.ndarray] = None,
) -> Dict:
    """Run a single simulation with the given config and pad_order."""
    if template_pads is not None:
        pads = template_pads
    else:
        pads, _ = _load_fresh_sim(config)
    sim = OrderedDrillingSimulator(pads, config, base_production,
                                  minimum_volumes, pad_order=pad_order,
                                  global_overwrites=global_overwrites,
                                  base_water=base_water)
    monthly = sim.run()
    passed, shortfall = sim.check_minimum_volumes(monthly, tolerance=config.shortfall_tolerance_mcfd)

    raw_sf = monthly[monthly["avg_min_vol_mcfd"] > 0]
    raw_max = raw_sf["avg_shortfall_mcfd"].max() if not raw_sf.empty else 0
    over_tol = shortfall["avg_shortfall_mcfd"].max() if not shortfall.empty else 0
    total_prod = monthly["monthly_volume_mcf"].sum()
    peak_prod = monthly["avg_total_mcfd"].max()
    total_fcf = monthly["cumulative_fcf_mm"].iloc[-1] if "cumulative_fcf_mm" in monthly.columns else 0

    prod_ok, prod_violations = sim.check_production_ceiling(monthly)

    return {
        "label": label,
        "sim": sim,
        "monthly": monthly,
        "passed": passed,
        "shortfall": shortfall,
        "raw_max_shortfall": raw_max,
        "over_tol_shortfall": over_tol,
        "total_prod_mcf": total_prod,
        "peak_mcfd": peak_prod,
        "total_fcf_mm": total_fcf,
        "prod_ceiling_ok": prod_ok,
        "prod_ceiling_violations": prod_violations,
        "pad_order": sim.get_pad_order(),
        "annual_capex": dict(sim.annual_capex_spent),
        "capex_timeline": sim.get_capex_timeline(),
        "fcf_summary": sim.get_fcf_summary(),
    }


# ----- 1h. Final-results plot helper (4×2 dashboard) ------------------------

def _add_net_fcf_columns(monthly_df: Optional[pd.DataFrame],
                         config: "SimConfig",
                         gas_replacement_price_per_mcf: float) -> Optional[pd.DataFrame]:
    """Return a copy of `monthly_df` with FCF columns NET of water cost and
    gas-shortfall replacement cost — strictly for PLOTTING.

    Added columns:
        - monthly_shortfall_cost_mm  : (avg_shortfall_mcfd - tol)+ * 30.4375 * $/MCF
        - monthly_fcf_net_mm         : monthly_fcf_mm - water_cost_mm - shortfall_cost
        - cumulative_fcf_net_mm      : cumsum of the above

    Notes:
        * Uses the same 30.4375 day/month convention as the fitness function so
          plotted totals reconcile with `pv_shortfall_cost_mm`.
        * Does NOT modify `monthly_fcf_mm` (fitness consumes that as gross-of-
          water; subtracting here would double-count the same way the old
          daily-FCF code did).
        * PVI delay penalty is intentionally EXCLUDED — that term is fitness-
          only, not a real cash flow.
    """
    if monthly_df is None or monthly_df.empty:
        return monthly_df
    DAYS_PER_MONTH = 30.4375
    out = monthly_df.copy()
    water = out.get("water_cost_mm", pd.Series(0.0, index=out.index)).fillna(0.0).values
    if "avg_shortfall_mcfd" in out.columns:
        sf_above_tol = np.maximum(
            0.0, out["avg_shortfall_mcfd"].values - config.shortfall_tolerance_mcfd
        )
        sf_cost = sf_above_tol * DAYS_PER_MONTH * gas_replacement_price_per_mcf / 1e6
    else:
        sf_cost = np.zeros(len(out))
    out["monthly_shortfall_cost_mm"] = sf_cost
    if "monthly_fcf_mm" in out.columns:
        out["monthly_fcf_net_mm"] = out["monthly_fcf_mm"].values - water - sf_cost
        out["cumulative_fcf_net_mm"] = out["monthly_fcf_net_mm"].cumsum()
    return out


def plot_final_results(results: pd.DataFrame, sim: OrderedDrillingSimulator,
                       config: SimConfig, optimizer_used: str, plot_folder: str = "",
                       baseline_results: Optional[pd.DataFrame] = None,
                       baseline_sim: Optional[OrderedDrillingSimulator] = None,
                       baseline_label: str = "PVI-rank baseline"):
    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.suptitle(f"FINAL — {optimizer_used} | ${config.annual_capex_limit_mm}MM "
                 f"(+${config.capex_tolerance_mm}MM) | SF tol: {config.shortfall_tolerance_mcfd:,.0f}",
                 fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    ax.fill_between(results["month"], results["avg_base_mcfd"], alpha=0.3, color="gray", label="Base")
    ax.fill_between(results["month"], results["avg_base_mcfd"],
                    results["avg_total_mcfd"], alpha=0.3, color="green", label="New")
    ax.plot(results["month"], results["avg_total_mcfd"], "g-", lw=2, label="Total")
    if baseline_results is not None and "avg_total_mcfd" in baseline_results.columns:
        ax.plot(baseline_results["month"], baseline_results["avg_total_mcfd"],
                color="purple", lw=1.5, ls="--", alpha=0.85, label=baseline_label)
    mm = results["avg_min_vol_mcfd"] > 0
    if mm.any():
        ax.plot(results.loc[mm, "month"], results.loc[mm, "avg_min_vol_mcfd"], "r--", lw=2, label="Min")
        ax.fill_between(results.loc[mm, "month"],
                        results.loc[mm, "avg_min_vol_mcfd"] - config.shortfall_tolerance_mcfd,
                        results.loc[mm, "avg_min_vol_mcfd"],
                        alpha=0.15, color="red", label=f"Tol")
    if config.production_ceiling_mcfd > 0 and "prod_ceiling_mcfd" in results.columns:
        pc = results["prod_ceiling_mcfd"]
        if pc.max() < float("inf"):
            ax.plot(results["month"], pc, "m--", lw=2, label="Prod Ceiling")
    ax.set_xlabel("Month")
    ax.set_ylabel("MCFD")
    ax.set_title("Production")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.bar(results["month"], results["avg_shortfall_mcfd"], color="red", alpha=0.6)
    if config.shortfall_tolerance_mcfd > 0:
        ax.axhline(y=config.shortfall_tolerance_mcfd, color="orange", ls="--", lw=2, label="Tol")
    ax.set_xlabel("Month")
    ax.set_ylabel("MCFD")
    ax.set_title("Shortfall")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.step(results["month"], results["rig_util"], "r-", lw=2,
            label=f"Rigs ({config.num_rigs})", where="post")
    ax.step(results["month"], results["frac_util"], "b-", lw=2,
            label=f"Frac ({config.num_frac_crews})", where="post")
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Rig & Frac")

    ax = axes[1, 1]
    ax.step(results["month"], results["land_util"], "-", lw=2, color="orange",
            label=f"Land ({config.num_land_crews})", where="post")
    ax.step(results["month"], results["permit_util"], "-", lw=2, color="goldenrod",
            label=f"Permit ({config.num_permit_crews})", where="post")
    ax.step(results["month"], results["con_util"], "-", lw=2, color="gold",
            label=f"Constr ({config.num_construction_crews})", where="post")
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Pre-Drill")

    ax = axes[2, 0]
    ct = sim.get_capex_timeline()
    if not ct.empty:
        pv = ct.groupby(["year", "milestone"])["capex_mm"].sum().unstack(fill_value=0)
        mc = {"land": "wheat", "permit": "khaki", "pad_construction": "gold",
              "midstream": "mediumpurple", "overland": "plum",
              "drill": "saddlebrown", "frac": "steelblue"}
        cols = [c for c in mc if c in pv.columns]
        pv[cols].plot(kind="bar", stacked=True, ax=ax, color=[mc[c] for c in cols], alpha=0.85)
        years = sorted(pv.index)
        budgets = [config.capex_limit_for_year(yr) for yr in years]
        ceilings = [b + config.capex_tolerance_mm for b in budgets]
        x_pos = range(len(years))
        ax.plot(x_pos, budgets, "r--", lw=2, label="Budget (CAGR)", marker="o", markersize=3)
        ax.plot(x_pos, ceilings, "orange", ls=":", lw=2, label="Ceiling", marker="s", markersize=3)
        if baseline_sim is not None:
            try:
                bct = baseline_sim.get_capex_timeline()
                if not bct.empty:
                    base_yr = bct.groupby("year")["capex_mm"].sum()
                    base_y = [float(base_yr.get(yr, 0.0)) for yr in years]
                    ax.plot(x_pos, base_y, color="purple", lw=2, ls="--",
                            marker="D", markersize=4, label=baseline_label)
            except Exception:
                pass
    ax.set_title("Capex (CAGR-adjusted)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(results["month"], results["cumulative_mcf"] / 1e6, "b-", lw=2, label="Cum. Production")
    # Prefer the NET cumulative FCF (gross − water − gas-shortfall) when present;
    # fall back to the gross cumulative FCF for back-compat.
    cum_fcf_col = ("cumulative_fcf_net_mm" if "cumulative_fcf_net_mm" in results.columns
                   else ("cumulative_fcf_mm" if "cumulative_fcf_mm" in results.columns else None))
    if cum_fcf_col is not None:
        ax2 = ax.twinx()
        fcf_label = ("Cum. FCF net of water+SF ($MM)"
                     if cum_fcf_col == "cumulative_fcf_net_mm"
                     else "Cum. FCF ($MM)")
        ax2.plot(results["month"], results[cum_fcf_col], "g-", lw=2, label=fcf_label)
        ax2.set_ylabel("FCF ($MM)", color="green")
        ax2.legend(loc="lower right", fontsize=7)
    ax.set_title("Cumulative Production & FCF")
    ax.set_ylabel("Production (MMCF)")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[3, 0]
    sch = sim.get_schedule()
    if sch.empty or "land_start" not in sch.columns:
        ax.set_title("Gantt Chart – Pad Phases (no pads scheduled)")
        ax.text(0.5, 0.5, "No pads scheduled", ha="center", va="center", transform=ax.transAxes)
    else:
        sp = sch[sch["land_start"].notna() | sch["drill_start"].notna()].sort_values("seq")
        yl = []
        gc = {"land": ("wheat", "Land"), "permit": ("khaki", "Permit"),
              "pad_con": ("gold", "Pad Const"), "predrill": ("lightyellow", "Predrill"),
              "midstream": ("mediumpurple", "Midstream"), "overland": ("plum", "Overland"),
              "drill": ("saddlebrown", "Drill"), "frac": ("steelblue", "Frac")}
        for i, (_, row) in enumerate(sp.iterrows()):
            for key, (color, lt) in gc.items():
                sc, ec = f"{key}_start", f"{key}_end"
                if sc in row and pd.notna(row[sc]):
                    ax.barh(i, row[ec] - row[sc], left=row[sc], color=color,
                            alpha=0.8, height=0.6, label=lt if i == 0 else "")
            mf = " ★" if pd.notna(row.get("mandatory_day")) else ""
            yl.append(f"[{int(row['seq'])}] {row['name']} ({int(row['num_wells'])}w){mf}")
        ax.set_yticks(range(len(yl)))
        ax.set_yticklabels(yl, fontsize=5)
        ax.set_title("Gantt (★=mandatory)")
        ax.legend(loc="lower right", fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()

    ax = axes[3, 1]
    ax.axis("off")
    cagr_str = f"\n  CAGR: {config.capex_cagr_pct}% x {config.capex_cagr_years}yr" if config.capex_cagr_pct > 0 else ""
    olms_str = f"\n  OL/MS Budget: ${config.overland_midstream_budget_mm}MM" if config.overland_midstream_budget_mm > 0 else ""
    ceil_str = f"\n  Prod Ceil: {config.production_ceiling_mcfd:,.0f}" if config.production_ceiling_mcfd > 0 else ""
    ax.text(0.1, 0.5,
            f"Config:\n  {config.num_land_crews}L {config.num_permit_crews}P "
            f"{config.num_construction_crews}C\n  {config.num_rigs}R {config.num_frac_crews}F\n"
            f"  Budget: ${config.annual_capex_limit_mm}MM\n  Tol: ${config.capex_tolerance_mm}MM\n"
            f"  SF Tol: {config.shortfall_tolerance_mcfd:,.0f}\n"
            f"  Price: ${config.commodity_price_per_mcf}/MCF\n"
            f"  Optimizer: {optimizer_used}\n"
            f"  Start: {config.simulation_start_date}"
            f"{cagr_str}{olms_str}{ceil_str}",
            fontsize=10, family="monospace", va="center", transform=ax.transAxes)
    plt.tight_layout()
    save_path = os.path.join(plot_folder, "final_result.png") if plot_folder else "final_result.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# 2. GA CONFIGURATION
# =============================================================================

@dataclass
class GAConfig:
    """Genetic Algorithm hyper-parameters."""
    # Population
    population_size: int = 60
    num_generations: int = 160
    elitism_count: int = 4              # top-N copied unchanged each gen
    tournament_size: int = 3            # tournament selection k

    # Operators (probabilities applied to each child)
    crossover_rate: float = 0.85
    swap_mutation_rate: float = 0.20    # per-child probability of one swap mutation
    insert_mutation_rate: float = 0.10  # per-child probability of one insertion mutation

    # ---- Economic objective parameters ------------------------------------
    # FCF and water cost are both discounted to PV using `discount_rate_annual`.
    # Shortfall is converted to a $ cost = shortfall_mcfd × days × gas_replacement_price.
    discount_rate_annual: float = 0.10              # 10% annual discount rate
    gas_replacement_price_per_mcf: float = 3.00     # $/MCF cost to buy gas to cover shortfall

    # PVI delay penalty (subtracted from the objective):
    #   penalty_$ = pad.pvi * pvi_delay_penalty_constant * (1 - discount_factor(first_production_day))
    # where first_production_day is the day the pad comes online (post-frac, with
    # midstream/overland complete) and discount_factor uses `discount_rate_annual`.
    # Pads that never reach production are charged the full sim horizon.
    # The penalty is summed across all pads and subtracted (in $MM) from `score`.
    # Set to 0.0 to disable.
    pvi_delay_penalty_constant: float = 0.0

    # ---- HARD CONSTRAINTS (return -inf if any are violated) ---------------
    # Set any of these to False to relax a constraint into a soft penalty.
    enforce_capex_budget: bool = True               # annual capex must stay within (limit + tolerance)
    enforce_production_ceiling: bool = True         # production ceiling must not be exceeded
    enforce_mandatory_pads: bool = True             # mandatory pads must start by their mandated date
    enforce_minimum_volumes: bool = False           # if True, any month below min vol is invalid
    enforce_water_takeaway: bool = True             # if True, ANY unhandled water (> tolerance) -> infeasible
    water_unhandled_tolerance_bbl: float = 0.0      # total bbl of unhandled water allowed across full sim

    # Reproducibility / runtime
    random_seed: int = 42
    parallel_workers: int = 0           # 0 = auto (CPU count - 1); 1 = serial; >1 = explicit
    verbose: bool = True

    # Initial-population seeding
    pvi_warm_start_fraction: float = 0.20   # fraction of initial pop seeded with PVI-sorted ordering


# =============================================================================
# 3. FITNESS EVALUATION
# =============================================================================

@dataclass
class FitnessBreakdown:
    """Detailed fitness components for inspection / logging.

    All $ values are in $MM. PV = present-value (discounted at `discount_rate_annual`).
    """
    score: float                # the value GA maximizes (NPV in $MM)
    feasible: bool              # False = violated a hard constraint -> score = -inf
    invalid_reason: str         # explanation if infeasible ("" if feasible)

    # Cashflow components (PV $MM)
    pv_fcf_mm: float
    pv_water_cost_mm: float
    pv_shortfall_cost_mm: float       # $ cost to buy gas to cover shortfall, discounted

    # Raw (undiscounted) components for reference
    total_fcf_mm: float
    total_water_cost_mm: float
    total_shortfall_mcfd_yr: float
    total_shortfall_cost_mm: float    # undiscounted shortfall cost

    # Constraint check details
    capex_overage_mm: float           # max annual capex over (limit + tolerance), 0 if none
    ceiling_overage_mcfd_yr: float    # production ceiling overage
    mandatory_violations: int         # count of mandatory pads slipped
    raw_passed_min_volumes: bool
    water_unhandled_total_bbl: float  # total unhandled (overflow) water across full sim, bbl
    water_unhandled_peak_month_bbl: float  # worst single month of unhandled water

    # PVI*constant*(1-df(first_production_day)) summed, in $MM (defaulted for back-compat)
    pv_pvi_delay_penalty_mm: float = 0.0

    # v1.1 additions: capex-disbursement diagnostics (defaulted for back-compat)
    pads_capex_blocked: int = 0
    mandatory_capex_overage_mm: float = 0.0


def _evaluate_ordering(
    pad_order: List[str],
    config: SimConfig,
    ga_config: GAConfig,
    base_production: np.ndarray,
    minimum_volumes: np.ndarray,
    template_pads: List[WellPad],
    base_water: Optional[np.ndarray] = None,
    global_overwrites: Optional[Dict] = None,
) -> FitnessBreakdown:
    """Run the full daily simulator on `pad_order` and compute a scalar score.

    Returns higher = better.
    """
    result = run_single_simulation(
        config=config,
        base_production=base_production,
        minimum_volumes=minimum_volumes,
        pad_order=pad_order,
        label="ga_eval",
        template_pads=template_pads,
        global_overwrites=global_overwrites,
        base_water=base_water,
    )
    monthly = result["monthly"]
    sim     = result["sim"]

    # Per-month discount factor (annual rate compounded monthly).
    # PV multiplier for month index m (0-based) = 1 / (1+r)^(m/12)
    r = ga_config.discount_rate_annual
    if r > 0:
        df_month = (1.0 + r) ** (-monthly["month"].values / 12.0)
    else:
        df_month = np.ones(len(monthly))

    # ---- Discounted FCF ($MM PV) -----------------------------------------
    if "monthly_fcf_mm" in monthly.columns:
        monthly_fcf = monthly["monthly_fcf_mm"].values
        pv_fcf_mm = float((monthly_fcf * df_month).sum())
        total_fcf_mm = float(monthly_fcf.sum())
    else:
        pv_fcf_mm = float(result.get("total_fcf_mm", 0.0))
        total_fcf_mm = pv_fcf_mm

    # ---- Discounted Water cost ($MM PV) -----------------------------------
    if "water_cost_mm" in monthly.columns:
        monthly_water = monthly["water_cost_mm"].values
        pv_water_cost_mm = float((monthly_water * df_month).sum())
        total_water_cost_mm = float(monthly_water.sum())
    else:
        pv_water_cost_mm = 0.0
        total_water_cost_mm = 0.0

    # ---- Shortfall as $ replacement-gas cost ------------------------------
    # Convert avg_shortfall_mcfd -> month volume (MCF) -> $ -> $MM, then discount.
    # Use ~30.4375 days/month (365.25/12) for the bookkeeping.
    DAYS_PER_MONTH = 30.4375
    if "avg_shortfall_mcfd" in monthly.columns:
        sf_mcfd      = monthly["avg_shortfall_mcfd"].values
        # Tolerance: ignore shortfall below the configured tolerance
        sf_above_tol = np.maximum(0.0, sf_mcfd - config.shortfall_tolerance_mcfd)
        sf_mcf_month = sf_above_tol * DAYS_PER_MONTH
        sf_cost_mm   = sf_mcf_month * ga_config.gas_replacement_price_per_mcf / 1e6
        total_shortfall_cost_mm = float(sf_cost_mm.sum())
        pv_shortfall_cost_mm    = float((sf_cost_mm * df_month).sum())
        total_shortfall_mcfd_yr = float(sf_above_tol.sum() / 12.0)
    else:
        total_shortfall_cost_mm = 0.0
        pv_shortfall_cost_mm    = 0.0
        total_shortfall_mcfd_yr = 0.0

    # ---- Production ceiling overage ---------------------------------------
    ceiling_overage_mcfd_yr = 0.0
    prod_ceiling_ok = bool(result.get("prod_ceiling_ok", True))
    if not prod_ceiling_ok:
        viol = result.get("prod_ceiling_violations", None)
        if viol is not None and not viol.empty:
            for col in ("overage_mcfd", "over_ceiling"):
                if col in viol.columns:
                    ceiling_overage_mcfd_yr = float(np.maximum(0.0, viol[col]).sum())
                    break

    # ---- Capex budget overage (per-year) ----------------------------------
    capex_overage_mm = 0.0
    cfg = config
    for yr, spent in sim.annual_capex_spent.items():
        limit = cfg.capex_limit_for_year(yr) if hasattr(cfg, "capex_limit_for_year") \
                else cfg.annual_capex_limit_mm
        ceiling = limit + cfg.capex_tolerance_mm
        capex_overage_mm = max(capex_overage_mm, float(spent - ceiling))
    capex_overage_mm = max(0.0, capex_overage_mm)

    # ---- Water takeaway overflow (bbl) ------------------------------------
    water_unhandled_total_bbl = 0.0
    water_unhandled_peak_month_bbl = 0.0
    if "water_unhandled_bbl" in monthly.columns:
        wu = monthly["water_unhandled_bbl"].values
        water_unhandled_total_bbl = float(wu.sum())
        water_unhandled_peak_month_bbl = float(wu.max()) if len(wu) else 0.0

    # ---- Mandatory-pad violations -----------------------------------------
    mandatory_violations = 0
    grace = int(getattr(config, "mandatory_pad_grace_days", 0))
    for p in sim.pads:
        if p.is_mandatory and p.mandatory_start_day is not None:
            actual_start = p.land_start if p.land_start is not None else 10**9
            if actual_start > p.mandatory_start_day + max(1, grace):
                mandatory_violations += 1

    mandatory_capex_overage_mm = float(getattr(sim, "mandatory_capex_overage_mm", 0.0))

    # ---- Pads that ended the sim still waiting on a paid milestone --------
    # (proxy for "capex starvation never cleared")
    _waiting_states = {
        PadStatus.WAITING_LAND, PadStatus.WAITING_PERMIT, PadStatus.WAITING_PAD_CON,
        PadStatus.WAITING_DRILL, PadStatus.WAITING_FRAC,
    }
    pads_capex_blocked = sum(1 for p in sim.pads if p.status in _waiting_states)

    # ---- PVI delay penalty -------------------------------------------------
    # For each pad: penalty_$ = pad.pvi * pvi_delay_penalty_constant * (1 - df)
    # where df = discount_factor at the pad's `first_production_day`.
    # Discount factor: df = (1+r)^(-day/365). Day 0 -> df=1 -> penalty=0.
    # Later days -> df<1 -> larger penalty -> incentivizes high-PVI pads first.
    # Penalty is converted to $MM and SUBTRACTED from the score.
    # Tying it to `first_production_day` (rather than land/permit start) means
    # any delay in bringing a pad online — chromosome position, capex caps,
    # crew bottlenecks, midstream/overland completion — translates directly
    # into a larger penalty for high-PVI pads. Pads that never reach
    # production are charged the full sim horizon.
    pv_pvi_delay_penalty_mm = 0.0
    if ga_config.pvi_delay_penalty_constant != 0.0:
        max_day = config.simulation_days
        for p in sim.pads:
            first_day = p.first_production_day if p.first_production_day is not None else max_day
            if r > 0:
                df_pad = (1.0 + r) ** (-first_day / 365.0)
            else:
                df_pad = 1.0
            pv_pvi_delay_penalty_mm += (p.pvi
                                        * ga_config.pvi_delay_penalty_constant
                                        * (1.0 - df_pad)) / 1e6

    # ---- HARD-CONSTRAINT GATING -------------------------------------------
    # If any enabled hard constraint is violated, this schedule is INVALID
    # and the GA must reject it.
    invalid_reasons: List[str] = []
    if ga_config.enforce_capex_budget and capex_overage_mm > 1e-6:
        invalid_reasons.append(f"capex over by ${capex_overage_mm:,.2f}MM")
    if ga_config.enforce_production_ceiling and ceiling_overage_mcfd_yr > 1e-6:
        invalid_reasons.append(f"prod ceiling over by {ceiling_overage_mcfd_yr:,.0f} MCFD·yr")
    if ga_config.enforce_mandatory_pads and mandatory_violations > 0:
        invalid_reasons.append(f"{mandatory_violations} mandatory pad(s) slipped")
    if ga_config.enforce_capex_budget and mandatory_capex_overage_mm > 1e-6:
        invalid_reasons.append(
            f"mandatory pre-reservation overflows annual cap by ${mandatory_capex_overage_mm:,.2f}MM"
        )
    if ga_config.enforce_minimum_volumes and not bool(result.get("passed", False)):
        invalid_reasons.append("min volumes not met")
    if ga_config.enforce_water_takeaway and \
            water_unhandled_total_bbl > ga_config.water_unhandled_tolerance_bbl:
        invalid_reasons.append(
            f"water takeaway exceeded ({water_unhandled_total_bbl:,.0f} bbl unhandled, "
            f"peak {water_unhandled_peak_month_bbl:,.0f} bbl/mo)"
        )

    feasible = (len(invalid_reasons) == 0)
    invalid_reason = "; ".join(invalid_reasons)

    if not feasible:
        score = float("-inf")
    else:
        # NPV-style objective (maximize):
        #   score = PV(FCF) - PV(Water) - PV(Shortfall replacement gas) - PV(PVI delay penalty)
        score = (pv_fcf_mm - pv_water_cost_mm - pv_shortfall_cost_mm
                 - pv_pvi_delay_penalty_mm)

    return FitnessBreakdown(
        score=score,
        feasible=feasible,
        invalid_reason=invalid_reason,
        pv_fcf_mm=pv_fcf_mm,
        pv_water_cost_mm=pv_water_cost_mm,
        pv_shortfall_cost_mm=pv_shortfall_cost_mm,
        total_fcf_mm=total_fcf_mm,
        total_water_cost_mm=total_water_cost_mm,
        total_shortfall_mcfd_yr=total_shortfall_mcfd_yr,
        total_shortfall_cost_mm=total_shortfall_cost_mm,
        capex_overage_mm=capex_overage_mm,
        ceiling_overage_mcfd_yr=ceiling_overage_mcfd_yr,
        mandatory_violations=mandatory_violations,
        raw_passed_min_volumes=bool(result.get("passed", False)),
        water_unhandled_total_bbl=water_unhandled_total_bbl,
        water_unhandled_peak_month_bbl=water_unhandled_peak_month_bbl,
        pv_pvi_delay_penalty_mm=pv_pvi_delay_penalty_mm,
        pads_capex_blocked=pads_capex_blocked,
        mandatory_capex_overage_mm=mandatory_capex_overage_mm,
    )


# =============================================================================
# 3b. PARALLEL WORKER PLUMBING
# =============================================================================
# Each worker process holds the (heavy, immutable) simulation context in module
# globals so we don't pickle ~hundreds of MB of pad/well/base-production data
# across the IPC boundary on every chromosome evaluation.

_WORKER_CTX: Dict[str, object] = {}


def _worker_init(config, ga_config, base_production, minimum_volumes,
                 template_pads, base_water, global_overwrites):
    """Pool initializer — runs ONCE per worker process at startup."""
    global _WORKER_CTX
    _WORKER_CTX = {
        "config": config,
        "ga_config": ga_config,
        "base_production": base_production,
        "minimum_volumes": minimum_volumes,
        "template_pads": template_pads,
        "base_water": base_water,
        "global_overwrites": global_overwrites,
    }


def _worker_eval(chrom: List[str]) -> FitnessBreakdown:
    """Pool task — evaluates one chromosome using the worker-local context."""
    ctx = _WORKER_CTX
    return _evaluate_ordering(
        pad_order=chrom,
        config=ctx["config"],            # type: ignore[arg-type]
        ga_config=ctx["ga_config"],      # type: ignore[arg-type]
        base_production=ctx["base_production"],   # type: ignore[arg-type]
        minimum_volumes=ctx["minimum_volumes"],   # type: ignore[arg-type]
        template_pads=ctx["template_pads"],       # type: ignore[arg-type]
        base_water=ctx["base_water"],             # type: ignore[arg-type]
        global_overwrites=ctx["global_overwrites"],   # type: ignore[arg-type]
    )


# =============================================================================
# 3c. PVI-GREEDY BENCHMARK
# =============================================================================
# A naive baseline: sort all pads by PVI descending (mandatory pads still go
# first to preserve their deadlines), then run the same simulator. The
# simulator's per-day `_can_afford` / crew-availability gates produce the
# "project start shifts to ensure constraints are met" behavior automatically.
# Used as a reference to quantify the GA's improvement over a heuristic.

def build_pvi_greedy_order(template_pads: List[WellPad]) -> List[str]:
    """Return pad names sorted by PVI descending. Mandatory pads come first
    (also PVI-sorted within the mandatory group) so deadlines are respected."""
    mand = sorted([p for p in template_pads if p.is_mandatory],
                  key=lambda p: -p.pvi)
    opt  = sorted([p for p in template_pads if not p.is_mandatory],
                  key=lambda p: -p.pvi)
    return [p.name for p in mand] + [p.name for p in opt]


# =============================================================================
# 4. GA OPERATORS  (permutation-based)
# =============================================================================

def _order_crossover(parent_a: List[str], parent_b: List[str], rng: random.Random) -> List[str]:
    """Order Crossover (OX) — preserves relative order of genes from parent_b
    while keeping a contiguous slice of parent_a."""
    n = len(parent_a)
    if n < 2:
        return parent_a[:]
    i, j = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[i:j+1] = parent_a[i:j+1]
    fill = [g for g in parent_b if g not in child]
    k = 0
    for idx in range(n):
        if child[idx] is None:
            child[idx] = fill[k]
            k += 1
    return child  # type: ignore[return-value]


def _swap_mutation(chrom: List[str], rng: random.Random) -> List[str]:
    n = len(chrom)
    if n < 2:
        return chrom
    i, j = rng.sample(range(n), 2)
    chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


def _insertion_mutation(chrom: List[str], rng: random.Random) -> List[str]:
    """Pluck a random gene and re-insert it at a random new position."""
    n = len(chrom)
    if n < 2:
        return chrom
    i = rng.randrange(n)
    gene = chrom.pop(i)
    j = rng.randrange(n)  # n now (length minus 1) + 1 valid spots = n
    chrom.insert(j, gene)
    return chrom


def _tournament_select(pop: List[Tuple[List[str], float]],
                       k: int, rng: random.Random) -> List[str]:
    """Pick k random individuals, return chromosome of best (highest fitness)."""
    contenders = rng.sample(pop, k) if len(pop) >= k else pop
    winner = max(contenders, key=lambda x: x[1])
    return winner[0][:]   # copy


# =============================================================================
# 5. GENETIC ALGORITHM OPTIMIZER
# =============================================================================

class GeneticAlgorithmOptimizer:
    """Permutation-based GA over pad drill orders."""

    def __init__(
        self,
        config: SimConfig,
        ga_config: GAConfig,
        base_production: np.ndarray,
        minimum_volumes: np.ndarray,
        template_pads: List[WellPad],
        base_water: Optional[np.ndarray] = None,
        global_overwrites: Optional[Dict] = None,
    ):
        self.config = config
        self.ga_config = ga_config
        self.base_production = base_production
        self.minimum_volumes = minimum_volumes
        self.template_pads = template_pads
        self.base_water = base_water
        self.global_overwrites = global_overwrites
        self.rng = random.Random(ga_config.random_seed)
        # Universe of pad names = drill order alphabet
        self.pad_names: List[str] = [p.name for p in template_pads]
        # Mandatory pads need to land EARLIER in the order (to give them more chance to start
        # by their deadline). We enforce this softly via the fitness penalty + a hard
        # pre-sort at chromosome creation time:
        self._mandatory_names: List[str] = [
            p.name for p in template_pads if p.is_mandatory
        ]
        self._optional_names: List[str] = [
            p.name for p in template_pads if not p.is_mandatory
        ]

        # Cache PVI lookup for warm-start
        self._pvi_lookup: Dict[str, float] = {p.name: p.pvi for p in template_pads}

        # Fitness cache (chromosome tuple -> FitnessBreakdown) to skip duplicate evals
        self._cache: Dict[Tuple[str, ...], FitnessBreakdown] = {}

        # Cumulative infeasibility-reason counts across ALL evaluations
        # Keys: "capex", "ceiling", "mandatory", "min_volumes", "water_takeaway", "feasible"
        self._infeas_counts: Dict[str, int] = {
            "feasible": 0,
            "capex": 0,
            "ceiling": 0,
            "mandatory": 0,
            "min_volumes": 0,
            "water_takeaway": 0,
        }

        # Generation history for plotting
        self.history: List[Dict] = []

    # ---------- Population initialization ----------
    def _make_random_chromosome(self) -> List[str]:
        """Random shuffle, but keep mandatory pads in the front half of the order
        so they get crew priority."""
        mand = self._mandatory_names[:]
        opt  = self._optional_names[:]
        self.rng.shuffle(mand)
        self.rng.shuffle(opt)
        return mand + opt

    def _make_pvi_chromosome(self) -> List[str]:
        """PVI-sorted ordering (mandatory still goes first)."""
        mand = sorted(self._mandatory_names, key=lambda n: -self._pvi_lookup[n])
        opt  = sorted(self._optional_names,  key=lambda n: -self._pvi_lookup[n])
        return mand + opt

    def _initial_population(self) -> List[List[str]]:
        pop: List[List[str]] = []
        n_warm = max(1, int(self.ga_config.pvi_warm_start_fraction
                            * self.ga_config.population_size))
        for _ in range(n_warm):
            chrom = self._make_pvi_chromosome()
            # Slightly perturb after the first one so we don't get duplicates
            if pop:
                _swap_mutation(chrom, self.rng)
                _swap_mutation(chrom, self.rng)
            pop.append(chrom)
        while len(pop) < self.ga_config.population_size:
            pop.append(self._make_random_chromosome())
        return pop

    # ---------- Fitness with cache ----------
    def _fitness(self, chrom: List[str]) -> FitnessBreakdown:
        key = tuple(chrom)
        if key in self._cache:
            return self._cache[key]
        fb = _evaluate_ordering(
            pad_order=chrom,
            config=self.config,
            ga_config=self.ga_config,
            base_production=self.base_production,
            minimum_volumes=self.minimum_volumes,
            template_pads=self.template_pads,
            base_water=self.base_water,
            global_overwrites=self.global_overwrites,
        )
        self._cache[key] = fb
        return fb

    # ---------- Main loop ----------
    def run(self) -> Tuple[List[str], FitnessBreakdown]:
        """Returns (best_chromosome, best_fitness_breakdown)."""
        ga = self.ga_config
        t0 = time.time()

        # ----- Initial population
        pop_chroms = self._initial_population()

        # ----- Decide worker count
        if ga.parallel_workers <= 0:
            n_workers = max(1, (os.cpu_count() or 2) - 1)
        else:
            n_workers = ga.parallel_workers

        executor = None
        if n_workers > 1:
            try:
                from concurrent.futures import ProcessPoolExecutor
                executor = ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_worker_init,
                    initargs=(self.config, self.ga_config,
                              self.base_production, self.minimum_volumes,
                              self.template_pads, self.base_water,
                              self.global_overwrites),
                )
                if ga.verbose:
                    print(f"  Parallel: ProcessPoolExecutor with {n_workers} workers")
            except Exception as ex:  # noqa: BLE001 — fall back to serial
                print(f"  WARNING: parallel pool init failed ({ex}); running serial.")
                executor = None

        def eval_pop(chroms: List[List[str]]) -> List[Tuple[List[str], FitnessBreakdown]]:
            """Evaluate a population, using cache + parallel pool when available.

            Returns scored list in input order.
            """
            # Split into cached vs uncached
            results: List[Optional[FitnessBreakdown]] = [None] * len(chroms)
            uncached_idxs: List[int] = []
            uncached_chroms: List[List[str]] = []
            for i, c in enumerate(chroms):
                key = tuple(c)
                if key in self._cache:
                    results[i] = self._cache[key]
                else:
                    uncached_idxs.append(i)
                    uncached_chroms.append(c)

            if uncached_chroms:
                if executor is not None:
                    # Parallel: dispatch all uncached chromosomes at once
                    futures = list(executor.map(_worker_eval, uncached_chroms))
                    for idx, fb in zip(uncached_idxs, futures):
                        results[idx] = fb
                        self._cache[tuple(chroms[idx])] = fb
                        self._tally_infeas(fb)
                else:
                    # Serial fallback
                    for idx, c in zip(uncached_idxs, uncached_chroms):
                        fb = _evaluate_ordering(
                            pad_order=c,
                            config=self.config,
                            ga_config=self.ga_config,
                            base_production=self.base_production,
                            minimum_volumes=self.minimum_volumes,
                            template_pads=self.template_pads,
                            base_water=self.base_water,
                            global_overwrites=self.global_overwrites,
                        )
                        results[idx] = fb
                        self._cache[tuple(c)] = fb
                        self._tally_infeas(fb)

            return [(c, fb) for c, fb in zip(chroms, results)]  # type: ignore[misc]

        scored = eval_pop(pop_chroms)
        best_overall = max(scored, key=lambda x: x[1].score)
        n_feas_init = sum(1 for _, fb in scored if fb.feasible)

        if ga.verbose:
            print(f"\n=== GENETIC ALGORITHM — initial pop ({ga.population_size}) ===")
            print(f"  feasible / total:  {n_feas_init} / {ga.population_size}")
            if best_overall[1].feasible:
                print(f"  best NPV score:    ${best_overall[1].score:>13,.2f}MM")
                print(f"  best PV(FCF):      ${best_overall[1].pv_fcf_mm:>13,.2f}MM")
                print(f"  best PV(Water):    ${best_overall[1].pv_water_cost_mm:>13,.2f}MM")
                print(f"  best PV(Shortfall):${best_overall[1].pv_shortfall_cost_mm:>13,.2f}MM")
            else:
                print(f"  ⚠️  No feasible solutions in initial pop — GA will keep searching.")
                print(f"      Best (infeasible) reason: {best_overall[1].invalid_reason}")

        # ---- DIAGNOSTIC: if NO feasible AND water-takeaway is to blame, show
        #      a blocking plot of water supply vs takeaway capacity for the
        #      worst-water chromosome so the user can see HOW MUCH it's missing by.
        water_failures_in_init = [
            (chrom, fb) for chrom, fb in scored
            if (not fb.feasible) and fb.water_unhandled_total_bbl > 0
        ]
        if (n_feas_init == 0 and water_failures_in_init
                and ga.enforce_water_takeaway):
            water_failures_in_init.sort(
                key=lambda x: -x[1].water_unhandled_total_bbl)
            worst_chrom, worst_fb = water_failures_in_init[0]
            print(f"\n  >>> Showing water-overage diagnostic plot for worst chromosome "
                  f"({worst_fb.water_unhandled_total_bbl:,.0f} bbl unhandled, "
                  f"peak {worst_fb.water_unhandled_peak_month_bbl:,.0f} bbl/mo).")
            print(f"  >>> Close the plot window to continue the GA run...")
            try:
                self._show_water_overage_plot(list(worst_chrom), worst_fb)
            except Exception as ex:  # noqa: BLE001
                print(f"  (water diagnostic plot failed: {ex})")

        for gen in range(1, ga.num_generations + 1):
            # Sort population by score desc
            scored.sort(key=lambda x: x[1].score, reverse=True)

            # ----- Elitism
            new_pop: List[List[str]] = [scored[i][0][:] for i in range(ga.elitism_count)]

            # ----- Fill rest via selection + crossover + mutation
            while len(new_pop) < ga.population_size:
                pa = _tournament_select(
                    [(c, fb.score) for c, fb in scored], ga.tournament_size, self.rng)
                pb = _tournament_select(
                    [(c, fb.score) for c, fb in scored], ga.tournament_size, self.rng)

                if self.rng.random() < ga.crossover_rate:
                    child = _order_crossover(pa, pb, self.rng)
                else:
                    child = pa[:]

                if self.rng.random() < ga.swap_mutation_rate:
                    _swap_mutation(child, self.rng)
                if self.rng.random() < ga.insert_mutation_rate:
                    _insertion_mutation(child, self.rng)

                new_pop.append(child)

            # ----- Evaluate new population
            scored = eval_pop(new_pop)
            gen_best = max(scored, key=lambda x: x[1].score)

            if gen_best[1].score > best_overall[1].score:
                best_overall = (gen_best[0][:], gen_best[1])

            # ----- Log
            # Filter -inf (infeasible) scores out of mean stats so the chart is meaningful
            feas_scores = [fb.score for _, fb in scored if fb.feasible]
            n_feas = len(feas_scores)
            mean_score = float(np.mean(feas_scores)) if feas_scores else float("nan")
            min_score  = float(np.min(feas_scores))  if feas_scores else float("nan")
            self.history.append({
                "generation": gen,
                "best_score": best_overall[1].score,
                "gen_best_score": gen_best[1].score,
                "gen_best_order": list(gen_best[0]),
                "gen_mean_score_feas": mean_score,
                "gen_min_score_feas":  min_score,
                "n_feasible": n_feas,
                "best_pv_fcf_mm":         best_overall[1].pv_fcf_mm,
                "best_pv_water_mm":       best_overall[1].pv_water_cost_mm,
                "best_pv_shortfall_mm":   best_overall[1].pv_shortfall_cost_mm,
                "best_pv_pvi_penalty_mm": best_overall[1].pv_pvi_delay_penalty_mm,
                "best_total_fcf_mm":      best_overall[1].total_fcf_mm,
                "best_total_water_mm":    best_overall[1].total_water_cost_mm,
                "best_feasible":          best_overall[1].feasible,
                "elapsed_s": time.time() - t0,
            })
            if ga.verbose:
                feas_tag = "✓" if best_overall[1].feasible else "✗"
                print(f"  gen {gen:>3} {feas_tag} | "
                      f"best=${best_overall[1].score:>11,.1f}MM  "
                      f"gen_best=${gen_best[1].score:>11,.1f}MM  "
                      f"feas={n_feas:>3}/{ga.population_size}  "
                      f"PV_FCF=${best_overall[1].pv_fcf_mm:>9,.1f}MM  "
                      f"PV_H2O=${best_overall[1].pv_water_cost_mm:>7,.2f}MM  "
                      f"PV_SF=${best_overall[1].pv_shortfall_cost_mm:>7,.2f}MM  "
                      f"|cache|={len(self._cache):>4}  "
                      f"({time.time()-t0:>5.1f}s)")

        if executor is not None:
            executor.shutdown(wait=True)

        if ga.verbose:
            print(f"\n=== GA DONE in {time.time()-t0:.1f}s "
                  f"({len(self._cache)} unique evals) ===")
            print(f"  Final best NPV score:   ${best_overall[1].score:>13,.2f}MM")
            print(f"  Final PV(FCF):          ${best_overall[1].pv_fcf_mm:>13,.2f}MM")
            print(f"  Final PV(Water cost):   ${best_overall[1].pv_water_cost_mm:>13,.2f}MM")
            print(f"  Final PV(Shortfall):    ${best_overall[1].pv_shortfall_cost_mm:>13,.2f}MM")
            print(f"  Undiscounted FCF:       ${best_overall[1].total_fcf_mm:>13,.2f}MM")
            print(f"  Undiscounted Water:     ${best_overall[1].total_water_cost_mm:>13,.2f}MM")
            print(f"  Undiscounted Shortfall: ${best_overall[1].total_shortfall_cost_mm:>13,.2f}MM")
            print(f"  Feasible:               {best_overall[1].feasible}")
            if not best_overall[1].feasible:
                print(f"    reason: {best_overall[1].invalid_reason}")
            print(f"  Min vol passed:         {best_overall[1].raw_passed_min_volumes}")
            print(f"  Mandatory violated:     {best_overall[1].mandatory_violations}")
            print(f"  Capex overage:          ${best_overall[1].capex_overage_mm:,.2f}MM")
            print(f"  Ceiling overage:        {best_overall[1].ceiling_overage_mcfd_yr:,.0f} MCFD·yr")
            print(f"  Water unhandled (best): {best_overall[1].water_unhandled_total_bbl:,.0f} bbl total, "
                  f"peak {best_overall[1].water_unhandled_peak_month_bbl:,.0f} bbl/mo")

            # ---- Infeasibility-by-reason rollup across ALL unique evals ----
            # Use cache size as the canonical denominator (a single chromosome
            # can fail multiple constraints, so reason counts can overlap).
            ic = self._infeas_counts
            n_unique = len(self._cache)
            print(f"\n=== INFEASIBILITY ROLLUP (unique evaluations: {n_unique}) ===")
            print(f"  feasible              : {ic['feasible']:>6} ({100*ic['feasible']/max(1,n_unique):>5.1f}%)")
            print(f"  failed - capex budget : {ic['capex']:>6} ({100*ic['capex']/max(1,n_unique):>5.1f}%)")
            print(f"  failed - prod ceiling : {ic['ceiling']:>6} ({100*ic['ceiling']/max(1,n_unique):>5.1f}%)")
            print(f"  failed - mandatory    : {ic['mandatory']:>6} ({100*ic['mandatory']/max(1,n_unique):>5.1f}%)")
            print(f"  failed - min volumes  : {ic['min_volumes']:>6} ({100*ic['min_volumes']/max(1,n_unique):>5.1f}%)")
            print(f"  failed - WATER takeaway: {ic['water_takeaway']:>5} ({100*ic['water_takeaway']/max(1,n_unique):>5.1f}%)")
            print(f"  (a single schedule may fail multiple constraints simultaneously)")

        return best_overall[0], best_overall[1]

    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def _tally_infeas(self, fb: "FitnessBreakdown") -> None:
        """Bucket each evaluation by which constraint(s) it violated.
        A single chromosome can violate multiple — we count each independently."""
        if fb.feasible:
            self._infeas_counts["feasible"] += 1
            return
        reason = fb.invalid_reason or ""
        if "capex" in reason:
            self._infeas_counts["capex"] += 1
        if "prod ceiling" in reason:
            self._infeas_counts["ceiling"] += 1
        if "mandatory" in reason:
            self._infeas_counts["mandatory"] += 1
        if "min volumes" in reason:
            self._infeas_counts["min_volumes"] += 1
        if "water takeaway" in reason:
            self._infeas_counts["water_takeaway"] += 1

    def _show_water_overage_plot(self, chrom: List[str],
                                 fb: "FitnessBreakdown") -> None:
        """Re-simulate `chrom` and pop a BLOCKING matplotlib window showing
        monthly water supply vs total disposal capacity vs unhandled overflow.

        Execution resumes when the user closes the window.
        """
        import matplotlib
        # Switch to an interactive backend if we're on a non-interactive one
        try:
            matplotlib.use("TkAgg", force=False)
        except Exception:  # noqa: BLE001
            pass
        import matplotlib.pyplot as plt

        result = run_single_simulation(
            config=self.config,
            base_production=self.base_production,
            minimum_volumes=self.minimum_volumes,
            pad_order=chrom,
            label="water_diagnostic",
            template_pads=self.template_pads,
            global_overwrites=self.global_overwrites,
            base_water=self.base_water,
        )
        monthly = result["monthly"]
        if "water_unhandled_bbl" not in monthly.columns:
            print("  (no water columns in monthly result; skipping plot)")
            return

        cfg = self.config
        DAYS_PER_MONTH = 30.4375

        # Outlet takeaway capacity (bbl/day), straight from the cascade.
        # (Storage tanks DELAY overflow but don't add steady-state takeaway.)
        outlet_bwpd = sum(getattr(cfg, cap) for _k, _l, cap, _c in WATER_OUTLET_CASCADE)

        month_idx = monthly["month"].values
        produced  = monthly["water_produced_bbl"].values.astype(float)
        if "water_rainfall_bbl" in monthly.columns:
            produced = produced + monthly["water_rainfall_bbl"].values
        unhandled = monthly["water_unhandled_bbl"].values.astype(float)
        to_frac   = monthly.get("water_to_frac_bbl",
                                pd.Series(np.zeros(len(monthly)))).values
        # Convert monthly bbl totals to bbl/day (monthly average)
        produced_pd  = produced / DAYS_PER_MONTH
        unhandled_pd = unhandled / DAYS_PER_MONTH
        to_frac_pd   = to_frac / DAYS_PER_MONTH
        net_disposal_demand_pd = np.maximum(0.0, produced_pd - to_frac_pd)

        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

        # Top panel — supply vs capacity (bbl/day)
        ax = axes[0]
        ax.plot(month_idx, produced_pd, "b-", lw=2, label="Water supply (produced + rainfall)")
        ax.plot(month_idx, net_disposal_demand_pd, "c--", lw=1.5,
                label="Net disposal demand (supply − frac consumption)")
        ax.axhline(outlet_bwpd, color="red", lw=2, ls="--",
                   label=f"Outlet takeaway cap = {outlet_bwpd:,.0f} BWPD")
        # Shade overage region
        over_pd = np.maximum(0.0, net_disposal_demand_pd - outlet_bwpd)
        ax.fill_between(month_idx, outlet_bwpd,
                        outlet_bwpd + over_pd,
                        color="red", alpha=0.25, label="Demand above outlets (storage absorbs some)")
        ax.set_ylabel("BBL / day (monthly avg)")
        ax.set_title(
            f"Water Takeaway Diagnostic — Worst Chromosome of Initial Population\n"
            f"Total unhandled = {fb.water_unhandled_total_bbl:,.0f} bbl  |  "
            f"Peak month unhandled = {fb.water_unhandled_peak_month_bbl:,.0f} bbl  |  "
            f"Tolerance = {self.ga_config.water_unhandled_tolerance_bbl:,.0f} bbl",
            fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)

        # Bottom panel — unhandled overflow per month (bbl/day, monthly avg)
        ax = axes[1]
        ax.bar(month_idx, unhandled_pd, color="black", alpha=0.85,
               label="Unhandled (overflowed all storage + outlets)")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("Month index (sim start = month 0)")
        ax.set_ylabel("Unhandled water (BBL/day, monthly avg)")
        ax.set_title(
            "Unhandled water per month — these months breach the constraint",
            fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="upper right", fontsize=9)

        # First-3 pads in this chromosome (so user sees what sequence broke things)
        head = " → ".join(chrom[:5])
        fig.text(0.01, 0.005, f"First 5 pads in this order: {head} …",
                 fontsize=8, color="gray")

        fig.tight_layout()
        # BLOCKING show — execution resumes when user closes the window
        plt.show(block=True)


# =============================================================================
# 5b. DIAGNOSTIC PLOTS (post-GA visualization dashboard)
# =============================================================================

# Disposition columns used by water-disposition stack plot AND mass-balance.
# Defined here once so both consumers stay in sync.
_WATER_DISPOSITION_COLS: List[Tuple[str, str, str]] = [
    ("water_to_frac_bbl",        "→ Frac",                 "#1f77b4"),
    ("water_to_co_storage_bbl",  "→ Co. Storage",          "#2ca02c"),
    ("water_to_tp_storage_bbl",  "→ 3rd-Party Storage",    "#17becf"),
    ("water_sharing_bbl",        "→ Water Sharing",        "#ffbb78"),
    ("water_select_rail_bbl",    "→ Select Rail",          "#ff7f0e"),
    ("water_pa_swd_bbl",         "→ PA SWD",               "#d62728"),
    ("water_remainder_bbl",      "→ Remainder",            "#7f7f7f"),
    ("water_unhandled_bbl",      "✗ Unhandled (overflow)", "black"),
]


def make_diagnostic_plots(
    optimizer: "GeneticAlgorithmOptimizer",
    best_fb: "FitnessBreakdown",
    best_order: List[str],
    final_sim: OrderedDrillingSimulator,
    final_monthly: pd.DataFrame,
    hist: pd.DataFrame,
    config: SimConfig,
    ga_config: "GAConfig",
    baseline_monthly: Optional[pd.DataFrame] = None,
    baseline_label: str = "PVI-rank baseline",
) -> None:
    """Render the full post-run diagnostic plot dashboard to `config.plot_output_folder`.

    All figures are saved via `_save_fig` (which tight-layouts, writes at dpi=120,
    closes, and prints the destination path).
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize

    folder = config.plot_output_folder

    # -----------------------------------------------------------------------
    # Plot 1: GA Convergence (4-panel)
    # -----------------------------------------------------------------------
    if not hist.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("GA Convergence Diagnostics", fontsize=14, fontweight="bold")

        ax = axes[0, 0]
        ax.plot(hist["generation"], hist["best_score"], "g-", lw=2.5, label="Best so far")
        ax.plot(hist["generation"], hist["gen_best_score"], "b--", lw=1, alpha=0.7, label="Gen best")
        m = hist["gen_mean_score_feas"].notna()
        if m.any():
            ax.plot(hist.loc[m, "generation"], hist.loc[m, "gen_mean_score_feas"],
                    "k:", lw=1, alpha=0.6, label="Gen mean (feasible)")
        ax.set_xlabel("Generation"); ax.set_ylabel("NPV Score ($MM)")
        ax.set_title("Score progression — flattening = converged")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[0, 1]
        ax.plot(hist["generation"], hist["n_feasible"], "m-", lw=2)
        ax.axhline(y=ga_config.population_size, color="gray", ls="--", alpha=0.5,
                   label=f"Pop size ({ga_config.population_size})")
        ax.set_xlabel("Generation"); ax.set_ylabel("# Feasible Individuals")
        ax.set_title("Population feasibility — should rise toward pop size")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.plot(hist["generation"], hist["best_pv_fcf_mm"], "g-", lw=2, label="PV(FCF)")
        ax.plot(hist["generation"], hist["best_pv_water_mm"], "b-", lw=2, label="PV(Water)")
        ax.plot(hist["generation"], hist["best_pv_shortfall_mm"], "r-", lw=2, label="PV(Shortfall)")
        if "best_pv_pvi_penalty_mm" in hist.columns:
            ax.plot(hist["generation"], hist["best_pv_pvi_penalty_mm"], color="purple", lw=2, label="PV(PVI delay)")
        # Net NPV line = PV(FCF) - PV(Water) - PV(Shortfall) - PV(PVI delay)
        net = (hist["best_pv_fcf_mm"] - hist["best_pv_water_mm"]
               - hist["best_pv_shortfall_mm"]
               - hist.get("best_pv_pvi_penalty_mm", 0.0))
        ax.plot(hist["generation"], net, "k-", lw=2.5, alpha=0.8, label="Net NPV")
        ax.set_xlabel("Generation"); ax.set_ylabel("$MM")
        ax.set_title("Best solution — PV components over time")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.plot(hist["generation"], hist["best_total_fcf_mm"], "g--", lw=1.5, label="Undiscounted FCF")
        ax.plot(hist["generation"], hist["best_pv_fcf_mm"], "g-", lw=2.5, label="PV(FCF) @10%")
        ax.set_xlabel("Generation"); ax.set_ylabel("$MM")
        ax.set_title("Discount impact on best solution's FCF")
        ax.legend(); ax.grid(alpha=0.3)

        _save_fig(fig, folder, "ga_convergence.png", "convergence plot")

    # -----------------------------------------------------------------------
    # Plot 2 & 3: PVI ordering + PVI-vs-position scatter
    # -----------------------------------------------------------------------
    # `position` here = rank by FIRST-MILESTONE start day (the day the simulator
    # actually started ANY work on the pad: land/permit/pad_con/midstream/overland/
    # drill/frac, whichever is earliest). This reflects what the engine actually
    # did, not chromosome order — capex caps and crew bottlenecks can push a pad's
    # real start far past its chromosome slot.
    def _first_ms_start(pad) -> Optional[int]:
        days = [d for d in (pad.land_start, pad.permit_start, pad.pad_con_start,
                            pad.midstream_start, pad.overland_start,
                            pad.drill_start, pad.frac_start)
                if d is not None]
        return min(days) if days else None

    order_data = []
    for name in best_order:
        pad = next((p for p in final_sim.pads if p.name == name), None)
        if pad is None:
            continue
        order_data.append({
            "name": name,
            "pvi": pad.pvi,
            "is_mandatory": pad.is_mandatory,
            "first_prod_day": pad.first_production_day,
            "first_ms_day": _first_ms_start(pad),
        })
    od = pd.DataFrame(order_data)
    if not od.empty:
        # Pads that never started any milestone go to the end (sorted by name for stability).
        od["_sort_key"] = od["first_ms_day"].fillna(10**9)
        od = od.sort_values(["_sort_key", "name"], kind="stable").reset_index(drop=True)
        od["position"] = np.arange(1, len(od) + 1)
        od = od.drop(columns=["_sort_key"])

    if not od.empty:
        # ----- Plot 2: bar chart colored by PVI
        fig, ax = plt.subplots(figsize=(max(12, len(od) * 0.25), 8))
        norm = Normalize(vmin=od["pvi"].min(), vmax=od["pvi"].max())
        cmap = cm.get_cmap("RdYlGn")
        bar_colors = [cmap(norm(v)) for v in od["pvi"]]
        ax.bar(od["position"], od["pvi"], color=bar_colors, edgecolor="black", linewidth=0.5)
        for _, row in od.iterrows():
            if row["is_mandatory"]:
                ax.text(row["position"], row["pvi"] + 0.05, "★",
                        ha="center", va="bottom", fontsize=10, color="black")
        ax.set_xticks(od["position"])
        ax.set_xticklabels(od["name"], rotation=90, fontsize=7)
        ax.set_xlabel("First-Milestone Position (1 = earliest activity)")
        ax.set_ylabel("PVI")
        ax.set_title("BEST ORDERING — colored by PVI (★ = mandatory)\n"
                     "Ranked by first milestone start day. Higher PVI bars at LEFT = "
                     "GA got economic pads moving first.",
                     fontsize=12, fontweight="bold")
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label("PVI")
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=od["pvi"].mean(), color="black", ls=":", alpha=0.5,
                   label=f"Mean PVI = {od['pvi'].mean():.2f}")
        ax.legend(loc="upper right")
        _save_fig(fig, folder, "best_order_by_pvi.png", "PVI-ordering plot")

        # ----- Plot 3: scatter with trend line
        fig, ax = plt.subplots(figsize=(10, 6))
        sc_colors = ["red" if m else "steelblue" for m in od["is_mandatory"]]
        ax.scatter(od["position"], od["pvi"], c=sc_colors, s=70, edgecolor="black", alpha=0.85)
        z = np.polyfit(od["position"], od["pvi"], 1)
        xline = np.array([od["position"].min(), od["position"].max()])
        ax.plot(xline, z[0] * xline + z[1], "g--", lw=2,
                label=f"Trend (slope = {z[0]:+.4f})")
        for _, row in od.nlargest(3, "pvi").iterrows():
            ax.annotate(row["name"], (row["position"], row["pvi"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xlabel("First-Milestone Position (1 = earliest activity)")
        ax.set_ylabel("PVI")
        ax.set_title("PVI by First-Milestone Position (red = mandatory)\n"
                     "Negative slope = high-PVI pads got moving first (GOOD)",
                     fontsize=12, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        _save_fig(fig, folder, "pvi_vs_position.png", "PVI-vs-position plot")

        # ----- Plot 3b: PVI vs First-Production Position --------------------
        # Same idea as Plot 3, but ranks pads by when they ACTUALLY come
        # online (first_production_day) rather than chromosome order.
        # Captures the effect of crew/capex/water delays the GA can't avoid.
        od_fp = od.dropna(subset=["first_prod_day"]).copy()
        if not od_fp.empty:
            od_fp = od_fp.sort_values("first_prod_day").reset_index(drop=True)
            od_fp["fp_position"] = np.arange(1, len(od_fp) + 1)

            fig, ax = plt.subplots(figsize=(10, 6))
            sc_colors = ["red" if m else "steelblue" for m in od_fp["is_mandatory"]]
            ax.scatter(od_fp["fp_position"], od_fp["pvi"], c=sc_colors,
                       s=70, edgecolor="black", alpha=0.85)
            z = np.polyfit(od_fp["fp_position"], od_fp["pvi"], 1)
            xline = np.array([od_fp["fp_position"].min(), od_fp["fp_position"].max()])
            ax.plot(xline, z[0] * xline + z[1], "g--", lw=2,
                    label=f"Trend (slope = {z[0]:+.4f})")
            for _, row in od_fp.nlargest(3, "pvi").iterrows():
                ax.annotate(row["name"], (row["fp_position"], row["pvi"]),
                            xytext=(5, 5), textcoords="offset points", fontsize=8)
            ax.set_xlabel("First-Production Position (1 = online first)")
            ax.set_ylabel("PVI")
            ax.set_title("PVI by First-Production Position (red = mandatory)\n"
                         "Negative slope = high-PVI pads come online early (GOOD)",
                         fontsize=12, fontweight="bold")
            ax.legend(); ax.grid(alpha=0.3)
            _save_fig(fig, folder, "pvi_vs_first_prod_position.png",
                      "PVI-vs-first-prod-position plot")

    # -----------------------------------------------------------------------
    # Plot 4: Cumulative discounted vs undiscounted FCF (NET of water + shortfall)
    # PVI delay penalty is intentionally excluded — that's a fitness-only term.
    # -----------------------------------------------------------------------
    fcf_col = ("monthly_fcf_net_mm" if "monthly_fcf_net_mm" in final_monthly.columns
               else "monthly_fcf_mm")
    if fcf_col in final_monthly.columns:
        r = ga_config.discount_rate_annual
        df_m = (1.0 + r) ** (-final_monthly["month"].values / 12.0)
        cum_und = np.cumsum(final_monthly[fcf_col].values)
        cum_pv  = np.cumsum(final_monthly[fcf_col].values * df_m)
        net_tag = " (net of water + gas shortfall)" if fcf_col == "monthly_fcf_net_mm" else ""
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(final_monthly["month"], cum_und, "g--", lw=1.5, alpha=0.7,
                label=f"Undiscounted: ${cum_und[-1]:,.1f}MM")
        ax.plot(final_monthly["month"], cum_pv, "g-", lw=2.5,
                label=f"PV @ {r*100:.0f}%: ${cum_pv[-1]:,.1f}MM")
        ax.fill_between(final_monthly["month"], cum_pv, cum_und, alpha=0.15, color="gray",
                        label="Discount loss")
        if (baseline_monthly is not None
                and fcf_col in baseline_monthly.columns):
            df_mb = (1.0 + r) ** (-baseline_monthly["month"].values / 12.0)
            cum_und_b = np.cumsum(baseline_monthly[fcf_col].values)
            cum_pv_b  = np.cumsum(baseline_monthly[fcf_col].values * df_mb)
            ax.plot(baseline_monthly["month"], cum_und_b, color="purple", lw=1.2, ls=":",
                    alpha=0.8, label=f"{baseline_label} undisc: ${cum_und_b[-1]:,.1f}MM")
            ax.plot(baseline_monthly["month"], cum_pv_b, color="purple", lw=2.0, ls="--",
                    alpha=0.9, label=f"{baseline_label} PV: ${cum_pv_b[-1]:,.1f}MM")
        ax.set_xlabel("Month"); ax.set_ylabel("Cumulative FCF ($MM)")
        ax.set_title(f"Best Solution — Cumulative FCF{net_tag}: Undiscounted vs Present Value",
                     fontsize=12, fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        _save_fig(fig, folder, "cumulative_fcf_pv.png", "cumulative FCF plot")

    # -----------------------------------------------------------------------
    # Plot 5: Score distribution histogram (feasible candidates only)
    # -----------------------------------------------------------------------
    if not hist.empty:
        cache_scores = [fb.score for fb in optimizer._cache.values() if fb.feasible]
        if cache_scores:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(cache_scores, bins=40, color="steelblue", edgecolor="black", alpha=0.75)
            ax.axvline(best_fb.score, color="red", lw=2.5,
                       label=f"BEST = ${best_fb.score:,.1f}MM")
            pct95 = float(np.percentile(cache_scores, 95))
            ax.axvline(pct95, color="orange", ls="--", lw=1.5,
                       label=f"95th pct = ${pct95:,.1f}MM")
            pct50 = float(np.percentile(cache_scores, 50))
            ax.axvline(pct50, color="gray", ls="--", lw=1.5,
                       label=f"Median = ${pct50:,.1f}MM")
            ax.set_xlabel("NPV Score ($MM)")
            ax.set_ylabel("Count")
            top_pct = (np.sum(np.array(cache_scores) >= best_fb.score) / len(cache_scores)) * 100
            ax.set_title(f"All Evaluated Feasible Schedules ({len(cache_scores)})\n"
                         f"Best is in the top {top_pct:.2f}% of all candidates explored",
                         fontsize=12, fontweight="bold")
            ax.legend(); ax.grid(alpha=0.3, axis="y")
            _save_fig(fig, folder, "score_distribution.png", "score-distribution plot")

    # -----------------------------------------------------------------------
    # Plot 6: NPV breakdown waterfall
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = ["PV(FCF)", "− PV(Water)", "− PV(Shortfall)", "− PV(PVI delay)", "= NPV Score"]
    values = [
        best_fb.pv_fcf_mm,
        -best_fb.pv_water_cost_mm,
        -best_fb.pv_shortfall_cost_mm,
        -best_fb.pv_pvi_delay_penalty_mm,
        best_fb.score,
    ]
    bar_colors = ["green", "blue", "red", "purple", "black"]
    bottom = [
        0,
        values[0],
        values[0] + values[1],
        values[0] + values[1] + values[2],
        0,
    ]
    heights = [values[0], values[1], values[2], values[3], best_fb.score]
    bars = ax.bar(labels, heights, bottom=bottom, color=bar_colors, edgecolor="black", alpha=0.8)
    for bar, val in zip(bars, values):
        y = bar.get_y() + bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"${val:,.1f}MM",
                ha="center", va="bottom" if val >= 0 else "top", fontsize=10, fontweight="bold")
    ax.axhline(y=0, color="black", lw=0.8)
    ax.set_ylabel("$MM")
    ax.set_title(f"Best Solution — NPV Breakdown @ {ga_config.discount_rate_annual*100:.0f}% Discount",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    _save_fig(fig, folder, "npv_waterfall.png", "NPV waterfall plot")

    # -----------------------------------------------------------------------
    # Plot 6b: Failure-mode bar chart
    # -----------------------------------------------------------------------
    ic = optimizer._infeas_counts
    n_unique = len(optimizer._cache)
    reason_labels = [
        ("Feasible",           ic["feasible"],       "#2ca02c"),
        ("Capex budget",       ic["capex"],          "#ff7f0e"),
        ("Production ceiling", ic["ceiling"],        "#1f77b4"),
        ("Mandatory pad slip", ic["mandatory"],      "#9467bd"),
        ("Min volumes",        ic["min_volumes"],    "#8c564b"),
        ("Water takeaway",     ic["water_takeaway"], "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = [lab for lab, _, _ in reason_labels]
    ys = [v for _, v, _ in reason_labels]
    cs = [c for _, _, c in reason_labels]
    bars = ax.bar(xs, ys, color=cs, edgecolor="black", alpha=0.85)
    for b, v in zip(bars, ys):
        pct = 100 * v / max(1, n_unique)
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("# unique chromosomes")
    ax.set_title(f"GA Failure Modes — {n_unique:,} unique schedules evaluated\n"
                 f"(a single chromosome can fail multiple constraints)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    _save_fig(fig, folder, "failure_modes.png", "failure-modes plot")

    # -----------------------------------------------------------------------
    # Console: top-10 worst water-takeaway failures
    # -----------------------------------------------------------------------
    water_failures = [(chrom, fb) for chrom, fb in optimizer._cache.items()
                      if (not fb.feasible) and fb.water_unhandled_total_bbl > 0]
    water_failures.sort(key=lambda x: -x[1].water_unhandled_total_bbl)
    if water_failures:
        print(f"\n=== TOP 10 WATER-TAKEAWAY FAILURES (of {len(water_failures)}) ===")
        print(f"{'rank':>4}  {'unhandled MMbbl':>16}  {'peak mo bbl':>14}  first-3 pads in order")
        for i, (chrom, fb) in enumerate(water_failures[:10], 1):
            head = ", ".join(chrom[:3])
            print(f"  {i:>2}.  {fb.water_unhandled_total_bbl/1e6:>15,.3f}  "
                  f"{fb.water_unhandled_peak_month_bbl:>14,.0f}  {head} …")
    else:
        print("\n=== No water-takeaway failures recorded — schedule has plenty of disposal capacity. ===")

    # -----------------------------------------------------------------------
    # Plots 7–10: Water profile through time
    # -----------------------------------------------------------------------
    required_water_cols = [c for c, _, _ in _WATER_DISPOSITION_COLS] + [
        "water_cost_mm", "avg_co_storage_bbl", "avg_tp_storage_bbl", "water_produced_bbl",
    ]
    water_cols_present = all(c in final_monthly.columns for c in required_water_cols)

    if not water_cols_present:
        print("  (skipping water plots: required water columns missing in monthly results)")
        return
    if not getattr(config, "water_enabled", True):
        print("  (skipping water plots: water tracking disabled in config)")
        return

    month_idx = final_monthly["month"]
    DAYS_PER_MONTH = 30  # simulator buckets months into 30-day windows

    # ----- Plot 7: stacked-area dispositions (BBL/day, monthly-average)
    fig, ax = plt.subplots(figsize=(14, 6))
    vals   = [final_monthly[c].values / DAYS_PER_MONTH for c, _, _ in _WATER_DISPOSITION_COLS]
    labs   = [lab for _, lab, _ in _WATER_DISPOSITION_COLS]
    cols   = [c for _, _, c in _WATER_DISPOSITION_COLS]
    ax.stackplot(month_idx, vals, labels=labs, colors=cols, alpha=0.85)
    produced = final_monthly["water_produced_bbl"].values
    if "water_rainfall_bbl" in final_monthly.columns:
        produced = produced + final_monthly["water_rainfall_bbl"].values
    ax.plot(month_idx, produced / DAYS_PER_MONTH, "k--", lw=2,
            label="Total Supply (produced + rainfall)")
    ax.set_xlabel("Month"); ax.set_ylabel("BBL / day (monthly avg)")
    ax.set_title("Water Disposition Over Time — where every drop goes\n"
                 "(stacked outflows must reconcile to total supply)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    _save_fig(fig, folder, "water_dispositions.png", "water disposition plot")

    # ----- Plot 8: storage tank levels
    fig, ax = plt.subplots(figsize=(14, 5))
    co_cap = float(getattr(config, "company_storage_capacity_bbl", 200_000))
    tp_cap = float(getattr(config, "thirdparty_storage_capacity_bbl", 75_000))
    ax.plot(month_idx, final_monthly["avg_co_storage_bbl"], "g-", lw=2,
            label=f"Company Storage (cap {co_cap:,.0f} bbl)")
    ax.plot(month_idx, final_monthly["avg_tp_storage_bbl"], "b-", lw=2,
            label=f"3rd-Party Storage (cap {tp_cap:,.0f} bbl)")
    ax.axhline(y=co_cap, color="g", ls="--", alpha=0.4)
    ax.axhline(y=tp_cap, color="b", ls="--", alpha=0.4)
    ax.fill_between(month_idx, 0, final_monthly["avg_co_storage_bbl"], color="green", alpha=0.10)
    ax.fill_between(month_idx, 0, final_monthly["avg_tp_storage_bbl"], color="blue",  alpha=0.10)
    ax.set_xlabel("Month"); ax.set_ylabel("Average BBL in tank")
    ax.set_title("Storage Tank Levels — confirms cascade fills tanks first, spills to outlets when full",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    _save_fig(fig, folder, "water_storage_levels.png", "storage-levels plot")

    # ----- Plot 9: water cost (monthly + cumulative PV vs undiscounted)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax = axes[0]
    ax.bar(month_idx, final_monthly["water_cost_mm"], color="steelblue", alpha=0.8)
    ax.set_ylabel("Monthly Water Cost ($MM)")
    ax.set_title("Water Cost Over Time — proves $ flows match disposal volumes",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    r = ga_config.discount_rate_annual
    df_m = (1.0 + r) ** (-month_idx.values / 12.0)
    cum_und = np.cumsum(final_monthly["water_cost_mm"].values)
    cum_pv  = np.cumsum(final_monthly["water_cost_mm"].values * df_m)
    ax.plot(month_idx, cum_und, "b--", lw=1.5, alpha=0.8,
            label=f"Undiscounted: ${cum_und[-1]:,.2f}MM")
    ax.plot(month_idx, cum_pv, "b-", lw=2.5,
            label=f"PV @ {r*100:.0f}%: ${cum_pv[-1]:,.2f}MM")
    ax.fill_between(month_idx, cum_pv, cum_und, color="gray", alpha=0.15, label="Discount savings")
    ax.set_xlabel("Month"); ax.set_ylabel("Cumulative Water Cost ($MM)")
    ax.legend(); ax.grid(alpha=0.3)
    _save_fig(fig, folder, "water_cost_timeline.png", "water cost timeline plot")

    # ----- Plot 10: mass-balance reconciliation (BBL/day, monthly-average)
    supply_in = (final_monthly["water_produced_bbl"].values +
                 final_monthly.get("water_rainfall_bbl",
                                   pd.Series(np.zeros(len(final_monthly)))).values)
    sink_out = sum(final_monthly[c].values for c, _, _ in _WATER_DISPOSITION_COLS)
    co = final_monthly["avg_co_storage_bbl"].values
    tp = final_monthly["avg_tp_storage_bbl"].values
    d_co = np.concatenate(([co[0]], np.diff(co)))
    d_tp = np.concatenate(([tp[0]], np.diff(tp)))
    d_storage = d_co + d_tp
    residual = supply_in - sink_out - d_storage

    supply_in_pd  = supply_in / DAYS_PER_MONTH
    sink_out_pd   = sink_out / DAYS_PER_MONTH
    d_storage_pd  = d_storage / DAYS_PER_MONTH
    residual_pd   = residual / DAYS_PER_MONTH

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    ax = axes[0]
    ax.plot(month_idx, supply_in_pd, "g-", lw=2.5, label="Supply IN (produced + rainfall)")
    ax.plot(month_idx, sink_out_pd, "r-", lw=2, alpha=0.85,
            label="Sink OUT (frac + storage + disposal + overflow)")
    ax.plot(month_idx, sink_out_pd + d_storage_pd, "b--", lw=1.5, alpha=0.8,
            label="Sink OUT + Δstorage (should overlay supply)")
    ax.set_ylabel("BBL / day (monthly avg)")
    ax.set_title("Water Mass Balance — Supply IN must equal Sink OUT + Δ(storage)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.bar(month_idx, residual_pd, color="purple", alpha=0.7)
    ax.axhline(y=0, color="black", lw=0.8)
    ax.set_xlabel("Month"); ax.set_ylabel("Residual (BBL/day)")
    tot_supply = supply_in.sum()
    tot_resid  = abs(residual).sum()
    pct_err    = 100 * tot_resid / max(1.0, tot_supply)
    ax.set_title(f"Mass-Balance Residual per Month "
                 f"(should be ≈ 0; total |residual| = {tot_resid:,.0f} bbl, "
                 f"{pct_err:.3f}% of total supply)", fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    _save_fig(fig, folder, "water_mass_balance.png", "mass-balance plot")

    # ----- Numeric summary
    print("\n=== WATER ACCOUNTING SUMMARY (best schedule) ===")
    tot_produced = float(final_monthly["water_produced_bbl"].sum())
    tot_rain     = float(final_monthly.get("water_rainfall_bbl",
                                           pd.Series(np.zeros(len(final_monthly)))).sum())
    print(f"  Total produced water         : {tot_produced/1e6:>10,.3f} MMbbl")
    print(f"  Total rainfall load          : {tot_rain/1e6:>10,.3f} MMbbl")
    print(f"  TOTAL SUPPLY                 : {(tot_produced+tot_rain)/1e6:>10,.3f} MMbbl")
    print(f"  ----")
    for col, lab, _ in _WATER_DISPOSITION_COLS:
        print(f"  {lab:<28} : {float(final_monthly[col].sum())/1e6:>10,.3f} MMbbl")
    print(f"  Δ Co. storage (end - start)  : {float(co[-1]-co[0])/1e6:>10,.3f} MMbbl")
    print(f"  Δ 3rd-party storage          : {float(tp[-1]-tp[0])/1e6:>10,.3f} MMbbl")
    print(f"  ----")
    print(f"  Mass-balance residual        : {float(residual.sum())/1e6:>10,.3f} MMbbl  "
          f"(|sum|={tot_resid/1e6:.3f}, {pct_err:.4f}%)")
    print(f"  Total water cost (undisc)    : ${float(final_monthly['water_cost_mm'].sum()):>10,.2f}MM")


# =============================================================================
# 6. MAIN ORCHESTRATION
# =============================================================================

def main():

    # =========================================================================
    # ALL INPUTS — EDIT THIS ONE BLOCK ONLY
    # =========================================================================
    config = SimConfig(
        # Resources
        num_rigs=2,
        num_frac_crews=1,
        num_land_crews=3,
        num_permit_crews=3,
        num_construction_crews=3,

        # Capital (base year)
        annual_capex_limit_mm=350,
        capex_tolerance_mm=10,

        # Capital CAGR (defaulted to 0)
        capex_cagr_pct=0.0,
        capex_cagr_years=0,

        # Overland / Midstream separate sub-budget
        overland_midstream_budget_mm=25.0,
        ol_ms_cagr_pct=0.0,
        ol_ms_cagr_years=0,

        # Production ceiling
        production_ceiling_mcfd=5_000_000,
        production_cagr_pct=0.0,
        production_cagr_years=0,

        # Free cashflow
        commodity_price_per_mcf=3.0,

        # Shortfall tolerance
        shortfall_tolerance_mcfd=10000,

        # Simulation
        simulation_days=3650,
        simulation_start_date="2026-06-01",

        # Input files — EDIT TO YOUR PATHS
        pad_filepath=r"\\coterra.com\data\Legacy\Tulsa\Departments\ProductionOperations\GRG\Optimization Engineering\Digital Innovation\Operations Research\MBU\v2_Rig Scheduler\v2_Schedule.csv",
        well_filepath=r"\\coterra.com\data\Legacy\Tulsa\Departments\ProductionOperations\GRG\Optimization Engineering\Digital Innovation\Operations Research\MBU\v2_Rig Scheduler\v2_Schedule_decline_curve_parameters_with_water.csv",
        base_production_filepath=r"\\coterra.com\data\Legacy\Tulsa\Departments\ProductionOperations\GRG\Optimization Engineering\Digital Innovation\Operations Research\MBU\v2_Rig Scheduler\v2_base_production.csv",
        minimum_volume_filepath=r"\\coterra.com\data\Legacy\Tulsa\Departments\ProductionOperations\GRG\Optimization Engineering\Digital Innovation\Operations Research\MBU\v2_Rig Scheduler\v1_minimum_volumes.csv",

        # Output files
        schedule_output=r"C:\Users\GGannaway\Downloads\pad_schedule_GA.csv",
        capex_output=r"C:\Users\GGannaway\Downloads\capex_timeline_GA.csv",
        well_prod_output=r"C:\Users\GGannaway\Downloads\well_production_output_GA.csv",
        monthly_output=r"C:\Users\GGannaway\Downloads\monthly_results_GA.csv",
        adjustment_output=r"C:\Users\GGannaway\Downloads\adjustment_log_GA.csv",
        water_output=r"C:\Users\GGannaway\Downloads\water_mass_balance_GA.csv",
        plot_output_folder=r"C:\Users\GGannaway\Downloads\plots_GA",
    )

    ga_config = GAConfig(
        population_size=60,
        num_generations=10,
        elitism_count=4,
        tournament_size=3,
        crossover_rate=0.85,
        swap_mutation_rate=0.20,
        insert_mutation_rate=0.10,

        # Economic objective
        discount_rate_annual=0.10,                # 10% annual discount
        gas_replacement_price_per_mcf=3.00,       # $/MCF replacement cost for shortfall

        # Hard constraints (set False to relax to soft penalty)
        enforce_capex_budget=True,
        enforce_production_ceiling=True,
        enforce_mandatory_pads=True,
        enforce_minimum_volumes=False,            # min volumes priced via gas replacement, not hard
        enforce_water_takeaway=True,              # NEW: any unhandled water -> infeasible
        water_unhandled_tolerance_bbl=0.0,        # zero tolerance; raise to allow tiny overflow

        pvi_warm_start_fraction=0.20,
        pvi_delay_penalty_constant=100_000_000,   # PVI*constant*df(first_milestone) penalty ($, before /1e6)
        random_seed=42,
        verbose=True,
    )
    # =========================================================================

    if config.plot_output_folder:
        os.makedirs(config.plot_output_folder, exist_ok=True)

    # ---- Load all CSVs (same loaders as v2)
    pads  = load_pads_from_csv(config.pad_filepath, config.simulation_start_date)
    wells = load_wells_from_csv(config.well_filepath)
    assign_wells_to_pads(pads, wells)

    base_production = load_base_production(config.base_production_filepath, config.simulation_days)
    minimum_volumes = load_minimum_volumes(config.minimum_volume_filepath, config.simulation_days)
    base_water      = load_base_water(config.base_production_filepath, config.simulation_days)  # optional

    print(f"\nLoaded: {len(pads)} pads, {sum(len(p.wells) for p in pads)} wells matched.\n")

    # ---- Run GA
    optimizer = GeneticAlgorithmOptimizer(
        config=config,
        ga_config=ga_config,
        base_production=base_production,
        minimum_volumes=minimum_volumes,
        template_pads=pads,
        base_water=base_water,
    )
    best_order, best_fb = optimizer.run()

    # ---- PVI-greedy benchmark (rank by PVI, let simulator handle deferrals)
    print("\n=== PVI-GREEDY BENCHMARK (rank-by-PVI, no GA) ===")
    bench_order = build_pvi_greedy_order(pads)
    bench_fb = _evaluate_ordering(
        pad_order=bench_order,
        config=config,
        ga_config=ga_config,
        base_production=base_production,
        minimum_volumes=minimum_volumes,
        template_pads=pads,
        base_water=base_water,
    )
    bench_result = run_single_simulation(
        config=config,
        base_production=base_production,
        minimum_volumes=minimum_volumes,
        pad_order=bench_order,
        label="PVI_GREEDY",
        template_pads=pads,
        base_water=base_water,
    )
    bench_monthly = bench_result["monthly"]

    # ---- Re-simulate the best ordering one more time for full reporting
    final_result = run_single_simulation(
        config=config,
        base_production=base_production,
        minimum_volumes=minimum_volumes,
        pad_order=best_order,
        label="GA_BEST",
        template_pads=pads,
        base_water=base_water,
    )
    final_sim     = final_result["sim"]
    final_monthly = final_result["monthly"]

    # Augment monthly DataFrames with FCF NET of water + gas-shortfall costs
    # for plotting (fitness still uses the gross monthly_fcf_mm column).
    final_monthly = _add_net_fcf_columns(final_monthly, config,
                                          ga_config.gas_replacement_price_per_mcf)
    bench_monthly = _add_net_fcf_columns(bench_monthly, config,
                                          ga_config.gas_replacement_price_per_mcf)

    final_sim.print_summary()

    # ---- GA vs PVI-Greedy side-by-side comparison
    def _fmt(v, w=12, prec=3):
        try:
            return f"{v:>{w},.{prec}f}"
        except Exception:
            return f"{str(v):>{w}}"

    ga_score = best_fb.score if best_fb.feasible else float("-inf")
    bg_score = bench_fb.score if bench_fb.feasible else float("-inf")
    if bench_fb.feasible and best_fb.feasible and abs(bench_fb.score) > 1e-9:
        improvement_pct = (ga_score - bg_score) / abs(bg_score) * 100.0
        improvement_str = f"{ga_score - bg_score:+,.3f} MM ({improvement_pct:+.2f}%)"
    else:
        improvement_str = "n/a (one or both infeasible)"

    print("\n" + "=" * 78)
    print("GA vs PVI-GREEDY BENCHMARK")
    print("=" * 78)
    header = f"{'Metric':<34}{'GA':>14}{'Greedy':>14}{'Δ (GA-Gr)':>16}"
    print(header)
    print("-" * len(header))
    rows = [
        ("NPV score ($MM)",            best_fb.score,                    bench_fb.score),
        ("PV FCF ($MM)",               best_fb.pv_fcf_mm,                bench_fb.pv_fcf_mm),
        ("PV Water cost ($MM)",        best_fb.pv_water_cost_mm,         bench_fb.pv_water_cost_mm),
        ("PV Shortfall cost ($MM)",    best_fb.pv_shortfall_cost_mm,     bench_fb.pv_shortfall_cost_mm),
        ("PV PVI delay penalty ($MM)", best_fb.pv_pvi_delay_penalty_mm,  bench_fb.pv_pvi_delay_penalty_mm),
        ("Total FCF undisc ($MM)",     best_fb.total_fcf_mm,             bench_fb.total_fcf_mm),
        ("Capex overage ($MM)",        best_fb.capex_overage_mm,         bench_fb.capex_overage_mm),
        ("Mandatory violations",       float(best_fb.mandatory_violations), float(bench_fb.mandatory_violations)),
        ("Water unhandled (Mbbl)",     best_fb.water_unhandled_total_bbl / 1e3,
                                       bench_fb.water_unhandled_total_bbl / 1e3),
    ]
    for label, ga_v, bg_v in rows:
        delta = ga_v - bg_v
        print(f"{label:<34}{_fmt(ga_v,14)}{_fmt(bg_v,14)}{_fmt(delta,16)}")
    print("-" * len(header))
    print(f"{'Feasible':<34}{str(best_fb.feasible):>14}{str(bench_fb.feasible):>14}")
    print(f"\nGA improvement over greedy: {improvement_str}")
    print("GA order   :", " > ".join(best_order))
    print("Greedy ord.:", " > ".join(bench_order))
    print("=" * 78 + "\n")

    # ---- Save outputs (re-using v2 helpers where possible)
    if config.monthly_output:
        final_monthly.to_csv(config.monthly_output, index=False)
        print(f"  -> wrote monthly results: {config.monthly_output}")
    if config.schedule_output:
        # Helper: convert a simulator's pads into schedule rows.
        def _pad_rows(pads_iter, run_label: str) -> List[Dict]:
            rows = []
            for p in pads_iter:
                rows.append({
                    "run": run_label,
                    "pad": p.name,
                    "is_mandatory": p.is_mandatory,
                    "mandatory_start_day": p.mandatory_start_day,
                    "land_start": p.land_start, "land_end": p.land_end,
                    "permit_start": p.permit_start, "permit_end": p.permit_end,
                    "pad_con_start": p.pad_con_start, "pad_con_end": p.pad_con_end,
                    "midstream_start": p.midstream_start, "midstream_end": p.midstream_end,
                    "overland_start": p.overland_start, "overland_end": p.overland_end,
                    "drill_start": p.drill_start, "drill_end": p.drill_end,
                    "frac_start": p.frac_start, "frac_end": p.frac_end,
                    "first_production_day": p.first_production_day,
                    "pvi": p.pvi,
                })
            return rows

        all_rows: List[Dict] = []

        # ---- Per-generation best orderings (re-run simulator on each)
        for h in optimizer.history:
            gen_order = h.get("gen_best_order")
            if not gen_order:
                continue
            gen_result = run_single_simulation(
                config=config,
                base_production=base_production,
                minimum_volumes=minimum_volumes,
                pad_order=gen_order,
                label=f"gen_{h['generation']}_best",
                template_pads=pads,
                base_water=base_water,
            )
            all_rows.extend(_pad_rows(gen_result["sim"].pads,
                                       f"population {h['generation']}"))

        # ---- Final solution
        all_rows.extend(_pad_rows(final_sim.pads, "final solution"))

        pd.DataFrame(all_rows).to_csv(config.schedule_output, index=False)
        print(f"  -> wrote pad schedule:  {config.schedule_output}")

    # ---- Daily water mass-balance CSV (final solution only) ---------------
    if getattr(config, "water_output", ""):
        n_days = len(final_sim.daily_water_produced_bbl)
        if n_days > 0:
            start = pd.to_datetime(config.simulation_start_date)
            dates = pd.date_range(start, periods=n_days, freq="D")
            water_df = pd.DataFrame({
                "date":                       dates,
                "day":                        np.arange(n_days),
                "produced_water_bbl":         final_sim.daily_water_produced_bbl,
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
            # Mass-balance check column: supply - dispositions - storage delta
            # Should be ~0 modulo storage tank changes already booked above.
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
            print(f"  -> wrote water mass balance: {config.water_output}")

    # ---- Per-pad-by-month FCF & production breakdown -----------------------
    # This is the diagnostic CSV for understanding *why* certain pads got
    # scheduled where they did. For each (pad × month):
    #   - production_mcfd  : average daily gas rate for that pad that month
    #   - revenue_mm       : production_mcfd * days_in_month * price / 1e6
    #   - opex_mm          : monthly opex if producing that month, else 0
    #   - capex_mm         : capex disbursed that month at any milestone midpoint
    #   - water_cost_mm    : approx pad-attributable water cost (produced bbl × marginal $)
    #   - fcf_mm           : revenue - opex - capex - water
    #   - fcf_pv_mm        : fcf_mm discounted at GA discount rate
    if config.plot_output_folder:
        try:
            n_days = config.simulation_days
            n_months = n_days // 30 + 1
            r = ga_config.discount_rate_annual
            price = config.commodity_price_per_mcf
            days_in_month = 30  # consistent with simulator's month bucketing
            df_month = (1.0 + r) ** (-np.arange(n_months) / 12.0)

            # Capacity-weighted disposal cost ($/bbl) for water cost approximation
            outlet_caps = [
                (getattr(config, cap), getattr(config, cost))
                for _key, _lab, cap, cost in WATER_OUTLET_CASCADE
            ]
            tot_cap = sum(c for c, _ in outlet_caps)
            avg_disp_cost = (sum(c * p for c, p in outlet_caps) / tot_cap) if tot_cap > 0 else 0.0

            pad_month_rows: List[dict] = []
            for p in final_sim.pads:
                # Per-pad capex disbursement by month, honoring the simulator's
                # `capex_disbursement` mode ("even" spreads over duration, "lump_start"
                # charges on the start day). Matches `_daily_capex_planned` semantics
                # so the chart cannot disagree with the engine's enforcement.
                # For mandatory pads, prefer pre-reservation start days (the days
                # the simulator actually committed the capex), not the daily-loop
                # scheduled start days.
                reserved = getattr(p, "_reserved_starts", None)
                def _start(default, ms_key):
                    if reserved is not None and ms_key in reserved:
                        return reserved[ms_key]
                    return default
                capex_by_month: Dict[int, float] = {}
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
                    # Even daily distribution; last day absorbs FP remainder.
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

                # Sum of capex disbursed (used to confirm == p.total_capex_mm)
                pad_total_capex_disbursed = sum(capex_by_month.values())

                # Walk each month
                for mi in range(n_months):
                    sample_day = mi * 30 + 15  # mid-month day
                    if sample_day >= n_days:
                        break
                    # Production: pad's avg rate at mid-month
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
            print(f"  -> wrote per-pad monthly FCF/prod: {pm_path}")

            # Per-pad summary roll-up: total/PV FCF, capex, prod over the whole horizon
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
            print(f"  -> wrote per-pad summary:        {ps_path}")

            # Print top diagnostic lines: pads where high PVI didn't translate to high PV(FCF)
            print("\n=== PVI vs PV(FCF) DIAGNOSTIC — pads sorted by PVI desc ===")
            print(f"{'#':>3} {'pad':<35} {'pvi':>6} {'pos':>4}  {'PV_FCF$MM':>10}  "
                  f"{'capex$MM':>9}  {'first_prod_mo':>13}  mand")
            for _, row in pad_summary.sort_values("pvi", ascending=False).iterrows():
                fp = row["first_production_month"]
                fp_str = f"{int(fp)}" if pd.notna(fp) else "—"
                print(f"{int(row['drill_position']):>3} "
                      f"{str(row['pad'])[:35]:<35} "
                      f"{row['pvi']:>6.2f} "
                      f"{int(row['drill_position']):>4}  "
                      f"${row['pv_fcf_mm']:>9,.1f}  "
                      f"${row['total_capex_mm']:>8,.1f}  "
                      f"{fp_str:>13}  "
                      f"{'★' if row['is_mandatory'] else ' '}")

        except Exception as ex:  # noqa: BLE001
            import traceback
            print(f"  (per-pad FCF CSV error: {ex})")
            traceback.print_exc()

    # ---- GA convergence history
    hist = optimizer.history_df()
    if not hist.empty and config.plot_output_folder:
        hist_path = os.path.join(config.plot_output_folder, "ga_convergence.csv")
        hist.to_csv(hist_path, index=False)
        print(f"  -> wrote GA history:    {hist_path}")

    # =====================================================================
    # VISUALIZATIONS — show the user that the final answer is the best
    # =====================================================================
    if config.plot_output_folder:
        try:
            make_diagnostic_plots(
                optimizer=optimizer,
                best_fb=best_fb,
                best_order=best_order,
                final_sim=final_sim,
                final_monthly=final_monthly,
                hist=hist,
                config=config,
                ga_config=ga_config,
                baseline_monthly=bench_monthly,
                baseline_label="PVI-rank baseline",
            )
        except Exception as ex:  # noqa: BLE001 — plotting is best-effort
            import traceback
            print(f"  (plotting error: {ex})")
            traceback.print_exc()

    # ---- Final results plot (re-uses v2 plotting)
    try:
        plot_final_results(final_monthly, final_sim, config,
                           optimizer_used="GA",
                           plot_folder=config.plot_output_folder,
                           baseline_results=bench_monthly,
                           baseline_sim=bench_result["sim"],
                           baseline_label="PVI-rank baseline")
        print(f"  -> wrote final_result.png to {config.plot_output_folder}")
    except Exception as ex:  # noqa: BLE001
        print(f"  (skipped final_results plot: {ex})")

    # ---- Print best ordering
    print("\n=== BEST PAD ORDER (GA) ===")
    for i, name in enumerate(best_order, 1):
        pad = next((p for p in final_sim.pads if p.name == name), None)
        if pad is None:
            continue
        flag = "★" if pad.is_mandatory else " "
        fp   = pad.first_production_day if pad.first_production_day is not None else "—"
        print(f"  {i:>3}. {flag} {name:<35} PVI={pad.pvi:>5.2f}  first_prod_day={fp}")

    print(f"\nFitness breakdown of best:")
    print(f"  NPV score (PV FCF − PV Water − PV Shortfall − PV PVI-delay) = ${best_fb.score:>13,.2f}MM")
    print(f"  PV(FCF)            @ {ga_config.discount_rate_annual*100:>4.1f}% = ${best_fb.pv_fcf_mm:>13,.2f}MM")
    print(f"  PV(Water cost)     @ {ga_config.discount_rate_annual*100:>4.1f}% = ${best_fb.pv_water_cost_mm:>13,.2f}MM")
    print(f"  PV(Shortfall cost) @ {ga_config.discount_rate_annual*100:>4.1f}% = ${best_fb.pv_shortfall_cost_mm:>13,.2f}MM")
    print(f"     (shortfall priced at ${ga_config.gas_replacement_price_per_mcf:.2f}/MCF replacement)")
    print(f"  PV(PVI delay pen.) @ {ga_config.discount_rate_annual*100:>4.1f}% = ${best_fb.pv_pvi_delay_penalty_mm:>13,.2f}MM  "
          f"(constant = {ga_config.pvi_delay_penalty_constant:,.0f} $/PVI)")
    print(f"  -- Undiscounted reference --")
    print(f"  Total FCF                  = ${best_fb.total_fcf_mm:>13,.2f}MM")
    print(f"  Total water cost           = ${best_fb.total_water_cost_mm:>13,.2f}MM")
    print(f"  Total shortfall cost       = ${best_fb.total_shortfall_cost_mm:>13,.2f}MM")
    print(f"  Total shortfall (volume)   = {best_fb.total_shortfall_mcfd_yr:>13,.2f} MCFD·yr")
    print(f"  -- Constraints --")
    print(f"  Capex overage              = ${best_fb.capex_overage_mm:>13,.2f}MM")
    print(f"  Production ceiling overage = {best_fb.ceiling_overage_mcfd_yr:>13,.0f} MCFD·yr")
    print(f"  Mandatory pad violations   = {best_fb.mandatory_violations}")
    print(f"  Min volumes met            = {best_fb.raw_passed_min_volumes}")
    print(f"  Feasible (all hard cons)   = {best_fb.feasible}")


if __name__ == "__main__":
    main()
