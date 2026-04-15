# =============================================================================
# v1.3.4 Rig-line Optimizer — CP-SAT with CAGR Capital/Production Ceilings,
# Overland/Midstream Sub-Budget, Midpoint Cost Incurrence, FCF Objective,
# Global Overwrite CSV, and Full Prompt Compliance
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use("Agg")  # Uncomment only if running headless
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum, auto
from copy import deepcopy

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("WARNING: OR-Tools not installed. Run: pip install ortools")

# =============================================================================
# 1. DATA STRUCTURES
# =============================================================================

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


@dataclass
class WellDeclineCurve:
    well_name: str
    pad_name: str
    gas_qi: float
    gas_di: float
    gas_b: float
    gas_final_di: float
    min_rate: float = 1.0

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

    def well_production_at_day(self, sim_day: int) -> Dict[str, float]:
        if self.first_production_day is None or sim_day < self.first_production_day:
            return {w.well_name: 0.0 for w in self.wells}
        t = sim_day - self.first_production_day
        return {w.well_name: w.rate_at_time(t) for w in self.wells}

    def approx_monthly_production(self, max_months: int = 120) -> List[float]:
        return [sum(w.rate_at_time(m * 30) for w in self.wells) for m in range(max_months)]


@dataclass
class SimConfig:
    """SINGLE CONFIGURATION BLOCK — edit here only."""
    # Starting resources
    num_rigs: int = 1
    num_frac_crews: int = 1
    num_land_crews: int = 1
    num_permit_crews: int = 1
    num_construction_crews: int = 1

    # Max resources
    max_rigs: int = 10
    max_frac_crews: int = 5
    max_land_crews: int = 10
    max_permit_crews: int = 10
    max_construction_crews: int = 10

    # Capital (base year)
    annual_capex_limit_mm: float = 500
    max_capex_mm: float = 2000
    capex_increment_mm: float = 50
    capex_tolerance_mm: float = 25

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

    # CP-SAT objective weights
    cpsat_pvi_weight_multiplier: float = 100    # PVI scaling factor (default: 100)
    cpsat_shortfall_weight: int = 10000          # SF_W penalty weight (default: 10000)
    cpsat_fcf_weight: int = 100                  # FCF maximization weight

    # Shortfall tolerance
    shortfall_tolerance_mcfd: float = 50000

    # PVI reshuffling (greedy fallback)
    pvi_reshuffle_enabled: bool = True
    pvi_reshuffle_tolerance: float = 0.3
    pvi_reshuffle_min_pvi: float = 1.3
    pvi_reshuffle_window: int = 5

    # CP-SAT — PRIMARY optimizer
    cpsat_enabled: bool = True
    cpsat_time_limit_seconds: int = 120
    cpsat_num_workers: int = 8
    cpsat_max_pads_override: Optional[int] = None

    # Simulation
    simulation_days: int = 3650
    simulation_start_date: str = "2026-01-01"

    # Input files (5 CSVs)
    pad_filepath: str = ""
    well_filepath: str = ""
    base_production_filepath: str = ""
    minimum_volume_filepath: str = ""
    global_overwrite_filepath: str = ""   # 5th CSV: per-year overwrite of capex/production/FCF limits

    # Output files
    schedule_output: str = ""
    capex_output: str = ""
    well_prod_output: str = ""
    monthly_output: str = ""
    adjustment_output: str = ""
    plot_output_folder: str = ""

    def capex_limit_for_year(self, year: int, overwrites: Optional[Dict] = None) -> float:
        """Annual capex limit applying CAGR and optional per-year overwrites."""
        if overwrites and year in overwrites:
            o = overwrites[year].get("capital_limit_mm")
            if o is not None:
                return o
        growth_years = min(year, self.capex_cagr_years) if self.capex_cagr_years > 0 else 0
        return self.annual_capex_limit_mm * (1 + self.capex_cagr_pct / 100.0) ** growth_years

    def ol_ms_limit_for_year(self, year: int, overwrites: Optional[Dict] = None) -> float:
        """OL/MS sub-budget for year (0 = unlimited within global budget)."""
        if self.overland_midstream_budget_mm <= 0:
            return float("inf")
        if overwrites and year in overwrites:
            o = overwrites[year].get("ol_ms_limit_mm")
            if o is not None:
                return o
        growth = min(year, self.ol_ms_cagr_years) if self.ol_ms_cagr_years > 0 else 0
        return self.overland_midstream_budget_mm * (1 + self.ol_ms_cagr_pct / 100.0) ** growth

    def production_limit_for_year(self, year: int, overwrites: Optional[Dict] = None) -> float:
        """Annual average production ceiling (MCFD). Returns inf if no ceiling."""
        if self.production_ceiling_mcfd <= 0:
            return float("inf")
        if overwrites and year in overwrites:
            o = overwrites[year].get("production_limit_mcfd")
            if o is not None:
                return o
        growth = min(year, self.production_cagr_years) if self.production_cagr_years > 0 else 0
        return self.production_ceiling_mcfd * (1 + self.production_cagr_pct / 100.0) ** growth

    @property
    def effective_annual_capex_mm(self) -> float:
        return self.annual_capex_limit_mm + self.capex_tolerance_mm


# =============================================================================
# 2. DATA LOADERS
# =============================================================================

def _load_time_series_csv(filepath: str, simulation_days: int,
                          tail_fill: bool = False) -> np.ndarray:
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    rate_col = [c for c in df.columns if "rate" in c.lower() or "mcfd" in c.lower()][0]
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
    arr = _load_time_series_csv(filepath, simulation_days, tail_fill=True)
    print(f"Base production: {arr[0]:,.0f} → {arr[-1]:,.0f} MCFD")
    return arr


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
            mandatory_start_day=mandatory_day, wells=[],
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
        wells.append(WellDeclineCurve(
            well_name=str(row["PROPERTY_NAME"]),
            pad_name=str(row["PAD_NAME"]),
            gas_qi=qi, gas_di=di, gas_b=b, gas_final_di=fdi))
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
    """Load the 5th CSV: per-year overwrite of capex, production, and FCF limits.

    Expected columns: year, capital_limit_mm, production_limit_mcfd, fcf_limit_mm,
                      ol_ms_limit_mm  (all optional except year).
    If a value is present for a given year, it overrides the CAGR-computed limit.
    """
    if not filepath or not os.path.exists(filepath):
        return {}
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    # Map alternate column names to canonical names
    col_map = {
        "Annual Total Net Capital, $MM": "capital_limit_mm",
        "annual_total_net_capital_mm": "capital_limit_mm",
        "Gross Production Rate, MCFD": "production_limit_mcfd",
        "gross_production_rate_mcfd": "production_limit_mcfd",
        "Free Cashflow, $MM": "fcf_limit_mm",
        "free_cashflow_mm": "fcf_limit_mm",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    overwrites: Dict[int, Dict[str, Optional[float]]] = {}
    for _, row in df.iterrows():
        yr_raw = row.get("year", row.get("Year", None))
        if yr_raw is None or pd.isna(yr_raw):
            continue
        yr = int(yr_raw)
        entry: Dict[str, Optional[float]] = {}
        for col in ["capital_limit_mm", "production_limit_mcfd", "fcf_limit_mm", "ol_ms_limit_mm"]:
            val = row.get(col, None)
            entry[col] = float(val) if val is not None and pd.notna(val) else None
        overwrites[yr] = entry
    print(f"Global overwrites loaded for years: {sorted(overwrites.keys())}")
    return overwrites


def _load_fresh_sim(config: SimConfig) -> Tuple[List[WellPad], List[WellDeclineCurve]]:
    """Helper: load pads and wells from CSV and assign."""
    pads = load_pads_from_csv(config.pad_filepath, config.simulation_start_date)
    wells = load_wells_from_csv(config.well_filepath)
    assign_wells_to_pads(pads, wells)
    return pads, wells


# =============================================================================
# 3. CORE SIMULATOR
# =============================================================================

class OrderedDrillingSimulator:
    def __init__(self, pads: List[WellPad], config: SimConfig,
                 base_production: Optional[np.ndarray] = None,
                 minimum_volumes: Optional[np.ndarray] = None,
                 pad_order: Optional[List[str]] = None,
                 global_overwrites: Optional[Dict] = None):
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
        self.annual_capex_spent: Dict[int, float] = {}
        self.annual_ol_ms_spent: Dict[int, float] = {}   # OL/MS sub-budget tracking
        self._pending_capex: List[Tuple[int, float, bool]] = []  # (day, amount, is_ol_ms)
        n = config.simulation_days
        self.base_production = base_production if base_production is not None else np.zeros(n)
        self.minimum_volumes = minimum_volumes if minimum_volumes is not None else np.zeros(n)
        self.daily_new_production: List[float] = []
        self.daily_total_production: List[float] = []
        self.daily_rig_use: List[int] = []
        self.daily_frac_use: List[int] = []
        self.daily_land_use: List[int] = []
        self.daily_permit_use: List[int] = []
        self.daily_construction_use: List[int] = []
        self.daily_fcf: List[float] = []    # daily free cashflow tracking
        self._daily_capex_mm: float = 0.0   # capex recorded on the current day
        self.log: List[str] = []

    def _get_year(self, day: int) -> int:
        return day // 365

    def _year_capex_limit(self, year: int) -> float:
        """Capex ceiling for year = CAGR-adjusted limit + tolerance, with overwrites."""
        base = self.config.capex_limit_for_year(year, self.global_overwrites)
        return base + self.config.capex_tolerance_mm

    def _year_ol_ms_limit(self, year: int) -> float:
        """OL/MS sub-budget for year (inf if not configured)."""
        return self.config.ol_ms_limit_for_year(year, self.global_overwrites)

    def _year_spent(self, day: int) -> float:
        return self.annual_capex_spent.get(self._get_year(day), 0.0)

    def _year_ol_ms_spent(self, day: int) -> float:
        return self.annual_ol_ms_spent.get(self._get_year(day), 0.0)

    def _can_afford(self, day: int, amount: float, is_ol_ms: bool = False) -> bool:
        yr = self._get_year(day)
        total_ok = (self._year_spent(day) + amount) <= self._year_capex_limit(yr)
        if is_ol_ms:
            ol_ms_ok = (self._year_ol_ms_spent(day) + amount) <= self._year_ol_ms_limit(yr)
            return total_ok and ol_ms_ok
        return total_ok

    def _schedule_midpoint_capex(self, pad: WellPad, milestone: str,
                                  start_day: int, duration_days: float):
        """Schedule capex to be incurred at the midpoint of the milestone."""
        amt = self._milestone_cost(pad, milestone)
        if amt > 0:
            midpoint = start_day + int(np.ceil(duration_days / 2.0))
            is_ol_ms = milestone in ("midstream", "overland")
            self._pending_capex.append((midpoint, amt, is_ol_ms))

    def _process_pending_capex(self, day: int):
        """Incur any capex whose midpoint day has been reached."""
        remaining = []
        for mid_day, amt, is_ol_ms in self._pending_capex:
            if day >= mid_day:
                yr = self._get_year(day)
                self.annual_capex_spent[yr] = self.annual_capex_spent.get(yr, 0.0) + amt
                self._daily_capex_mm += amt
                if is_ol_ms:
                    self.annual_ol_ms_spent[yr] = self.annual_ol_ms_spent.get(yr, 0.0) + amt
            else:
                remaining.append((mid_day, amt, is_ol_ms))
        self._pending_capex = remaining

    def _record_capex(self, pad: WellPad, milestone: str, day: int):
        """Legacy immediate recording — kept for compatibility but now called via midpoint."""
        amt = self._milestone_cost(pad, milestone)
        if amt > 0:
            yr = self._get_year(day)
            self.annual_capex_spent[yr] = self.annual_capex_spent.get(yr, 0.0) + amt
            if milestone in ("midstream", "overland"):
                self.annual_ol_ms_spent[yr] = self.annual_ol_ms_spent.get(yr, 0.0) + amt

    def _milestone_cost(self, pad: WellPad, milestone: str) -> float:
        return {"land": pad.capex_land_mm, "permit": pad.capex_permit_mm,
                "pad_construction": pad.capex_pad_construction_mm,
                "midstream": pad.capex_midstream_mm, "overland": pad.capex_overland_mm,
                "drill": pad.capex_drill_mm, "frac": pad.capex_frac_mm}.get(milestone, 0.0)

    def run(self) -> pd.DataFrame:
        for day in range(self.config.simulation_days):
            # --- Reset daily capex accumulator & process pending midpoint capex ---
            self._daily_capex_mm = 0.0
            self._process_pending_capex(day)

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
                    # Skip predrill — go straight to WAITING_DRILL
                    pad.status = PadStatus.WAITING_DRILL
                if (pad.status == PadStatus.DRILLING and
                        pad.drill_end is not None and day >= pad.drill_end):
                    pad.status = PadStatus.WAITING_FRAC
                    self.rigs_in_use -= 1
                if (pad.status == PadStatus.FRACKING and
                        pad.frac_end is not None and day >= pad.frac_end):
                    self.frac_crews_in_use -= 1
                    # Production begins day after frac, if all milestones done
                    all_done = (pad.midstream_end is not None and day >= pad.midstream_end and
                                pad.overland_end is not None and day >= pad.overland_end)
                    if all_done:
                        pad.status = PadStatus.PRODUCING
                        pad.first_production_day = day + 1  # day AFTER frac
                    else:
                        pad.status = PadStatus.WAITING_PRODUCTION
                if (pad.status == PadStatus.WAITING_PRODUCTION and
                        pad.frac_end is not None and day >= pad.frac_end and
                        pad.midstream_end is not None and day >= pad.midstream_end and
                        pad.overland_end is not None and day >= pad.overland_end):
                    pad.status = PadStatus.PRODUCING
                    pad.first_production_day = day + 1

            # --- Parallel milestones: midstream & overland (start after pad construction, gated by OL/MS budget) ---
            for pad in self.pads:
                if pad.pad_con_end is not None and day >= pad.pad_con_end:
                    if pad.midstream_start is None:
                        c = self._milestone_cost(pad, "midstream")
                        if c == 0 or self._can_afford(day, c, is_ol_ms=True):
                            pad.midstream_start = day
                            pad.midstream_end = day + int(np.ceil(pad.midstream_construction_days))
                            if c > 0:
                                self._schedule_midpoint_capex(pad, "midstream", day,
                                                              pad.midstream_construction_days)
                    if pad.overland_start is None:
                        c = self._milestone_cost(pad, "overland")
                        if c == 0 or self._can_afford(day, c, is_ol_ms=True):
                            pad.overland_start = day
                            pad.overland_end = day + int(np.ceil(pad.overland_construction_days))
                            if c > 0:
                                self._schedule_midpoint_capex(pad, "overland", day,
                                                              pad.overland_construction_days)

            # --- Serial milestones: land → permit → pad construction ---
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
                    if not self._can_afford(day, cost):
                        break
                    dur = getattr(pad, dur_attr)
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
                    self._schedule_midpoint_capex(pad, milestone, day, dur)

            # --- Frac assignment ---
            wf = [p for p in self.pads if p.status == PadStatus.WAITING_FRAC
                  and p.drill_end is not None and day >= p.drill_end
                  and p.overland_end is not None and day >= p.overland_end]
            wf.sort(key=lambda p: p.drill_end)
            for pad in wf:
                if self.frac_crews_in_use >= self.config.num_frac_crews:
                    break
                if not self._can_afford(day, self._milestone_cost(pad, "frac")):
                    break
                pad.status = PadStatus.FRACKING
                pad.frac_start = day
                pad.frac_end = day + int(np.ceil(pad.frac_days))
                self.frac_crews_in_use += 1
                self._schedule_midpoint_capex(pad, "frac", day, pad.frac_days)

            # --- Drill assignment ---
            wd = [p for p in self.pads if p.status == PadStatus.WAITING_DRILL]
            wd.sort(key=lambda p: p.sequence_number)
            for pad in wd:
                if self.rigs_in_use >= self.config.num_rigs:
                    break
                if not self._can_afford(day, self._milestone_cost(pad, "drill")):
                    break
                pad.status = PadStatus.DRILLING
                pad.drill_start = day
                pad.drill_end = day + int(np.ceil(pad.drill_days))
                self.rigs_in_use += 1
                self._schedule_midpoint_capex(pad, "drill", day, pad.drill_days)

            # --- Pad entry ---
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

            # --- Production & FCF ---
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

            # Daily FCF: revenue - opex - capex
            revenue_mm = total_prod * self.config.commodity_price_per_mcf / 1e6
            daily_opex_mm = sum(
                p.annual_opex_mm / 365.0
                for p in self.pads if p.first_production_day is not None and day >= p.first_production_day
            )
            self.daily_fcf.append(revenue_mm - daily_opex_mm - self._daily_capex_mm)

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
        ).reset_index()
        m["year"] = m["max_day"] // 365
        c = self.config

        # Add per-year capex budget and production ceiling (CAGR-adjusted)
        m["capex_budget_mm"] = m["year"].apply(lambda yr: c.capex_limit_for_year(yr, self.global_overwrites))
        m["prod_ceiling_mcfd"] = m["year"].apply(lambda yr: c.production_limit_for_year(yr, self.global_overwrites))

        m["rig_util"] = m["avg_rigs"] / c.num_rigs
        m["frac_util"] = m["avg_frac"] / c.num_frac_crews if c.num_frac_crews > 0 else 0
        m["land_util"] = m["avg_land"] / c.num_land_crews if c.num_land_crews > 0 else 0
        m["permit_util"] = m["avg_permit"] / c.num_permit_crews if c.num_permit_crews > 0 else 0
        m["con_util"] = m["avg_con"] / c.num_construction_crews if c.num_construction_crews > 0 else 0
        m["cumulative_mcf"] = m["monthly_volume_mcf"].cumsum()
        m["cumulative_fcf_mm"] = m["monthly_fcf_mm"].cumsum()
        return m

    def check_minimum_volumes(self, monthly: pd.DataFrame,
                              tolerance: float = 0.0) -> Tuple[bool, pd.DataFrame]:
        active = monthly[monthly["avg_min_vol_mcfd"] > 0].copy()
        if active.empty:
            return True, pd.DataFrame()
        failing = active[active["avg_shortfall_mcfd"] > tolerance].copy()
        return failing.empty, failing

    def check_production_ceiling(self, monthly: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        """Check if annual average production exceeds CAGR-adjusted ceiling."""
        if self.config.production_ceiling_mcfd <= 0:
            return True, pd.DataFrame()
        yearly = monthly.groupby("year").agg(
            avg_total_mcfd=("avg_total_mcfd", "mean")).reset_index()
        yearly["ceiling_mcfd"] = yearly["year"].apply(
            lambda yr: self.config.production_limit_for_year(yr, self.global_overwrites))
        yearly["over_ceiling"] = yearly["avg_total_mcfd"] - yearly["ceiling_mcfd"]
        violations = yearly[yearly["over_ceiling"] > 0].copy()
        return violations.empty, violations

    def get_pad_order(self) -> List[str]:
        return [p.name for p in self.pads]

    def get_capex_summary(self) -> Dict[int, Dict]:
        s = {}
        for yr, sp in sorted(self.annual_capex_spent.items()):
            budget = self.config.capex_limit_for_year(yr, self.global_overwrites)
            ceiling = budget + self.config.capex_tolerance_mm
            ol_ms_sp = self.annual_ol_ms_spent.get(yr, 0.0)
            ol_ms_lim = self.config.ol_ms_limit_for_year(yr, self.global_overwrites)
            s[yr] = {"spent": sp, "budget": budget, "ceiling": ceiling,
                     "over_budget": max(0, sp - budget),
                     "over_ceiling": max(0, sp - ceiling),
                     "ol_ms_spent": ol_ms_sp, "ol_ms_limit": ol_ms_lim,
                     "ol_ms_over": max(0, ol_ms_sp - ol_ms_lim) if ol_ms_lim != float("inf") else 0}
        return s

    def get_fcf_summary(self) -> Dict[int, Dict]:
        """Annual free cashflow summary: revenue - opex - capex."""
        yearly_fcf_rev: Dict[int, float] = {}
        yearly_opex: Dict[int, float] = {}
        n = self.config.simulation_days
        for day in range(n):
            yr = day // 365
            total_prod = self.daily_total_production[day] if day < len(self.daily_total_production) else 0
            rev = total_prod * self.config.commodity_price_per_mcf / 1e6
            opex = sum(p.annual_opex_mm / 365.0 for p in self.pads
                       if p.first_production_day is not None and day >= p.first_production_day)
            yearly_fcf_rev[yr] = yearly_fcf_rev.get(yr, 0.0) + rev
            yearly_opex[yr] = yearly_opex.get(yr, 0.0) + opex
        s = {}
        all_years = set(list(yearly_fcf_rev.keys()) + list(self.annual_capex_spent.keys()))
        for yr in sorted(all_years):
            rev = yearly_fcf_rev.get(yr, 0.0)
            opex = yearly_opex.get(yr, 0.0)
            capex = self.annual_capex_spent.get(yr, 0.0)
            s[yr] = {"revenue_mm": rev, "opex_mm": opex, "capex_mm": capex,
                     "fcf_mm": rev - opex - capex}
        return s

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
        recs = []
        for p in self.pads:
            for ms, amt, sd in [
                ("land", p.capex_land_mm, p.land_start),
                ("permit", p.capex_permit_mm, p.permit_start),
                ("pad_construction", p.capex_pad_construction_mm, p.pad_con_start),
                ("midstream", p.capex_midstream_mm, p.midstream_start),
                ("overland", p.capex_overland_mm, p.overland_start),
                ("drill", p.capex_drill_mm, p.drill_start),
                ("frac", p.capex_frac_mm, p.frac_start)]:
                if amt > 0 and sd is not None:
                    recs.append({"pad_name": p.name, "seq": p.sequence_number,
                                 "milestone": ms, "capex_mm": amt, "day": sd,
                                 "month": sd // 30, "year": sd // 365})
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
            print("\n  Free Cashflow (Revenue - Opex - Capex):")
            for yr, info in fcf.items():
                print(f"    Yr {yr}: Rev ${info['revenue_mm']:>7.1f}MM "
                      f"- Opex ${info['opex_mm']:>6.1f}MM "
                      f"- Capex ${info['capex_mm']:>7.1f}MM "
                      f"= FCF ${info['fcf_mm']:>7.1f}MM")
        print("=" * 80)


# =============================================================================
# 4. CP-SAT SOLUTION CALLBACK & DIAGNOSTICS
# =============================================================================

class SolutionLogger(cp_model.CpSolverSolutionCallback):
    """Captures every improving solution CP-SAT finds during search."""

    def __init__(self, pad_vars, pads_subset):
        super().__init__()
        self.pad_vars = pad_vars
        self.pads_subset = pads_subset
        self.solutions = []
        self._solution_count = 0

    def on_solution_callback(self):
        self._solution_count += 1
        obj_val = self.ObjectiveValue()
        best_bound = self.BestObjectiveBound()
        wall_time = self.WallTime()

        order = []
        for pad in self.pads_subset:
            nm = pad.name
            drill_s = self.Value(self.pad_vars[nm]["drill_s"])
            prod_s = self.Value(self.pad_vars[nm]["prod_s"])
            land_s = self.Value(self.pad_vars[nm]["land_s"])
            order.append({
                "pad": nm, "pvi": pad.pvi,
                "drill_start": drill_s, "prod_start": prod_s,
                "land_start": land_s,
                "qi_mcfd": pad.total_qi_mcfd,
                "capex_mm": pad.total_capex_mm,
            })
        order.sort(key=lambda x: x["drill_start"])

        self.solutions.append({
            "solution_num": self._solution_count,
            "objective": obj_val,
            "best_bound": best_bound,
            "gap_pct": (obj_val - best_bound) / max(abs(obj_val), 1e-9) * 100
                       if obj_val != 0 else 0.0,
            "wall_time": wall_time,
            "pad_order": [s["pad"] for s in order],
            "pad_details": order,
        })


def extract_solver_stats(solver, status, callback=None):
    """Extract comprehensive solver diagnostics after a solve."""
    stats = {
        "status": solver.StatusName(status),
        "objective_value": solver.ObjectiveValue() if status in (
            cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "best_objective_bound": solver.BestObjectiveBound(),
        "wall_time_sec": solver.WallTime(),
        "num_branches": solver.NumBranches(),
        "num_conflicts": solver.NumConflicts(),
        "num_boolean_propagations": solver.NumBooleans(),
        "optimality_gap_pct": None,
    }
    if stats["objective_value"] is not None:
        obj = stats["objective_value"]
        bound = stats["best_objective_bound"]
        stats["optimality_gap_pct"] = (
            (obj - bound) / max(abs(obj), 1e-9) * 100
        )
    if callback:
        stats["num_solutions_found"] = len(callback.solutions)
        stats["convergence_trace"] = [
            {"solution": s["solution_num"],
             "objective": s["objective"],
             "bound": s["best_bound"],
             "gap_pct": s["gap_pct"],
             "time": s["wall_time"]}
            for s in callback.solutions
        ]
    else:
        stats["num_solutions_found"] = 0
        stats["convergence_trace"] = []
    return stats


# =============================================================================
# 5. TIERED CP-SAT OPTIMIZER WITH PROGRESSIVE FALLBACK
# =============================================================================

def estimate_max_pads(pads: List[WellPad], config: SimConfig) -> int:
    sorted_pads = sorted(pads, key=lambda p: p.pvi, reverse=True)
    num_years = config.simulation_days // 365
    total_budget = config.effective_annual_capex_mm * num_years
    cum_capex, capex_count = 0, 0
    for pad in sorted_pads:
        cum_capex += pad.total_capex_mm
        if cum_capex > total_budget:
            break
        capex_count += 1
    total_rig_days = config.num_rigs * config.simulation_days
    cum_drill, rig_count = 0, 0
    for pad in sorted_pads:
        cum_drill += pad.drill_days + pad.frac_days
        if cum_drill > total_rig_days:
            break
        rig_count += 1
    min_res = min(config.num_land_crews, config.num_permit_crews,
                  config.num_construction_crews)
    total_chain_cap = min_res * config.simulation_days
    cum_chain, chain_count = 0, 0
    for pad in sorted_pads:
        cum_chain += pad.total_cycle_days
        if cum_chain > total_chain_cap:
            break
        chain_count += 1
    max_pads = min(capex_count, rig_count, chain_count)
    max_pads = max(max_pads, 3)
    return max_pads


def run_cpsat_on_subset(
    config: SimConfig, pads_subset: List[WellPad],
    base_production: np.ndarray, minimum_volumes: np.ndarray,
    global_overwrites: Optional[Dict] = None,
) -> Tuple[Optional[List[str]], Optional[SolutionLogger], Optional[Dict]]:
    if not ORTOOLS_AVAILABLE:
        return None, None, None

    model = cp_model.CpModel()
    horizon = config.simulation_days
    num_pads = len(pads_subset)
    num_years = horizon // 365 + 1
    num_months = horizon // 30 + 1

    monthly_base = np.zeros(num_months)
    monthly_target = np.zeros(num_months)
    for mi in range(num_months):
        ds, de = mi * 30, min((mi + 1) * 30, horizon)
        if ds < len(base_production):
            monthly_base[mi] = np.mean(base_production[ds:de])
        if ds < len(minimum_volumes):
            monthly_target[mi] = np.mean(minimum_volumes[ds:de])

    COST_SCALE = 100

    pad_vars = {}
    all_land, all_perm, all_con, all_drl, all_frc = [], [], [], [], []
    ms_starts = []       # (pad, milestone_name, start_var, capex_scaled, is_ol_ms)

    for pad in pads_subset:
        nm = pad.name
        ld = max(1, int(np.ceil(pad.land_owner_agreement_days)))
        pd_ = max(1, int(np.ceil(pad.pad_permit_days)))
        cd = max(1, int(np.ceil(pad.pad_construction_days)))
        dd = max(1, int(np.ceil(pad.drill_days)))
        fd = max(1, int(np.ceil(pad.frac_days)))
        md = max(1, int(np.ceil(pad.midstream_construction_days)))
        od = max(1, int(np.ceil(pad.overland_construction_days)))

        ls = model.NewIntVar(pad.earliest_start_day, horizon, f"{nm}_ls")
        le = model.NewIntVar(0, horizon, f"{nm}_le")
        li = model.NewIntervalVar(ls, ld, le, f"{nm}_li")
        ps = model.NewIntVar(0, horizon, f"{nm}_ps")
        pe = model.NewIntVar(0, horizon, f"{nm}_pe")
        pi = model.NewIntervalVar(ps, pd_, pe, f"{nm}_pi")
        cs = model.NewIntVar(0, horizon, f"{nm}_cs")
        ce = model.NewIntVar(0, horizon, f"{nm}_ce")
        ci = model.NewIntervalVar(cs, cd, ce, f"{nm}_ci")
        ds = model.NewIntVar(0, horizon, f"{nm}_ds")
        de = model.NewIntVar(0, horizon, f"{nm}_de")
        di = model.NewIntervalVar(ds, dd, de, f"{nm}_di")
        fs = model.NewIntVar(0, horizon, f"{nm}_fs")
        fe = model.NewIntVar(0, horizon, f"{nm}_fe")
        fi = model.NewIntervalVar(fs, fd, fe, f"{nm}_fi")
        ms_ = model.NewIntVar(0, horizon, f"{nm}_ms")
        me = model.NewIntVar(0, horizon, f"{nm}_me")
        model.Add(me == ms_ + md)
        os_ = model.NewIntVar(0, horizon, f"{nm}_os")
        oe = model.NewIntVar(0, horizon, f"{nm}_oe")
        model.Add(oe == os_ + od)
        prds = model.NewIntVar(0, horizon, f"{nm}_prds")

        # Serial: land → permit → pad construction → drill → frac
        model.Add(ps >= le)
        model.Add(cs >= pe)
        model.Add(ds >= ce)   # drill starts after pad construction (no predrill)
        model.Add(fs >= de)
        if pad.mandatory_start_day is not None:
            model.Add(ls == pad.mandatory_start_day)
        # Parallel: midstream & overland start after pad construction (gated by OL/MS budget)
        model.Add(ms_ >= ce)
        model.Add(os_ >= ce)
        # Frac must wait for overland
        model.Add(fs >= oe)
        # Production after frac AND midstream AND overland complete
        model.Add(prds >= fe)
        model.Add(prds >= me)
        model.Add(prds >= oe)

        all_land.append(li)
        all_perm.append(pi)
        all_con.append(ci)
        all_drl.append(di)
        all_frc.append(fi)
        pad_vars[nm] = {"land_s": ls, "drill_s": ds, "prod_s": prds}

        # Track milestone starts with midpoint-based capex assignment year
        for mn, mv, mc, is_olms, dur in [
            ("land", ls, pad.capex_land_mm, False, ld),
            ("permit", ps, pad.capex_permit_mm, False, pd_),
            ("con", cs, pad.capex_pad_construction_mm, False, cd),
            ("midstream", ms_, pad.capex_midstream_mm, True, md),
            ("overland", os_, pad.capex_overland_mm, True, od),
            ("drill", ds, pad.capex_drill_mm, False, dd),
            ("frac", fs, pad.capex_frac_mm, False, fd)]:
            if mc > 0:
                # Midpoint var: cost incurred at start + dur/2
                mid_var = model.NewIntVar(0, horizon + 365, f"{nm}_{mn}_mid")
                model.Add(mid_var == mv + dur // 2)
                ms_starts.append((pad, mn, mid_var, int(mc * COST_SCALE), is_olms))

    # Resource cumulative constraints
    dem = [1] * num_pads
    model.AddCumulative(all_land, dem, config.num_land_crews)
    model.AddCumulative(all_perm, dem, config.num_permit_crews)
    model.AddCumulative(all_con, dem, config.num_construction_crews)
    model.AddCumulative(all_drl, dem, config.num_rigs)
    model.AddCumulative(all_frc, dem, config.num_frac_crews)

    # --- Annual CAPEX constraint (CAGR-adjusted, with separate OL/MS sub-budget) ---
    for yr in range(num_years):
        ys, ye = yr * 365, (yr + 1) * 365
        yr_ceiling = int((config.capex_limit_for_year(yr, global_overwrites) +
                          config.capex_tolerance_mm) * COST_SCALE)
        yr_ol_ms_limit = config.ol_ms_limit_for_year(yr, global_overwrites)
        yr_ol_ms_ceiling = int(yr_ol_ms_limit * COST_SCALE) if yr_ol_ms_limit != float("inf") else None

        yct_all = []
        yct_olms = []
        for pad, mn, mid_var, csc, is_olms in ms_starts:
            iy = model.NewBoolVar(f"{pad.name}_{mn}_y{yr}")
            b1 = model.NewBoolVar(f"{pad.name}_{mn}_y{yr}_ge")
            model.Add(mid_var >= ys).OnlyEnforceIf(b1)
            model.Add(mid_var < ys).OnlyEnforceIf(b1.Not())
            b2 = model.NewBoolVar(f"{pad.name}_{mn}_y{yr}_lt")
            model.Add(mid_var < ye).OnlyEnforceIf(b2)
            model.Add(mid_var >= ye).OnlyEnforceIf(b2.Not())
            model.AddBoolAnd([b1, b2]).OnlyEnforceIf(iy)
            model.AddBoolOr([b1.Not(), b2.Not()]).OnlyEnforceIf(iy.Not())
            yct_all.append(csc * iy)
            if is_olms:
                yct_olms.append(csc * iy)
        if yct_all:
            model.Add(sum(yct_all) <= yr_ceiling)
        if yct_olms and yr_ol_ms_ceiling is not None:
            model.Add(sum(yct_olms) <= yr_ol_ms_ceiling)

    # --- Production ceiling constraint (CAGR-adjusted) ---
    if config.production_ceiling_mcfd > 0:
        for yr in range(num_years):
            ceil_mcfd = config.production_limit_for_year(yr, global_overwrites)
            if ceil_mcfd == float("inf"):
                continue
            ys_day = yr * 365
            # Approximate: producing pads' qi sum < ceiling - base avg
            yr_months = [mi for mi in range(num_months) if mi * 30 >= ys_day and mi * 30 < ys_day + 365]
            if not yr_months:
                continue
            avg_base_yr = np.mean([monthly_base[mi] for mi in yr_months]) if yr_months else 0
            headroom = max(0, int(ceil_mcfd - avg_base_yr))
            if headroom > 0:
                yr_mid_day = ys_day + 182
                prod_terms = []
                for pad in pads_subset:
                    ip = model.NewBoolVar(f"{pad.name}_prod_y{yr}")
                    model.Add(pad_vars[pad.name]["prod_s"] <= yr_mid_day).OnlyEnforceIf(ip)
                    model.Add(pad_vars[pad.name]["prod_s"] > yr_mid_day).OnlyEnforceIf(ip.Not())
                    prod_terms.append(int(pad.total_qi_mcfd * 0.5) * ip)
                if prod_terms:
                    model.Add(sum(prod_terms) <= headroom)

    # --- Objective: minimize (PVI-weighted start time + shortfall penalty - FCF reward) ---
    obj = []

    # 1) PVI rank order: earlier start for high-PVI pads
    for pad in pads_subset:
        w = max(1, int(pad.pvi * config.cpsat_pvi_weight_multiplier)) * max(1, int(pad.total_qi_mcfd / 1000))
        obj.append(pad_vars[pad.name]["prod_s"] * w)

    # 2) Shortfall penalty
    SF_W = config.cpsat_shortfall_weight
    for mi in range(num_months):
        tgt = monthly_target[mi] - config.shortfall_tolerance_mcfd
        if tgt <= 0:
            continue
        nwt = tgt - monthly_base[mi]
        if nwt <= 0:
            continue
        md_ = mi * 30
        pt = []
        for pad in pads_subset:
            ip = model.NewBoolVar(f"{pad.name}_p_m{mi}")
            model.Add(pad_vars[pad.name]["prod_s"] <= md_).OnlyEnforceIf(ip)
            model.Add(pad_vars[pad.name]["prod_s"] > md_).OnlyEnforceIf(ip.Not())
            pt.append(int(pad.total_qi_mcfd * 0.5) * ip)
        tn = model.NewIntVar(0, int(sum(p.total_qi_mcfd for p in pads_subset)), f"tn_m{mi}")
        model.Add(tn == sum(pt))
        sf = model.NewIntVar(0, int(nwt), f"sf_m{mi}")
        model.Add(sf >= int(nwt) - tn)
        obj.append(sf * SF_W)

    # 3) FCF reward: earlier production → more revenue → lower objective (negative contribution)
    FCF_W = config.cpsat_fcf_weight
    if FCF_W > 0 and config.commodity_price_per_mcf > 0:
        for pad in pads_subset:
            # Approximate daily revenue potential; earlier prod_s → more total revenue
            # Use negative penalty: penalize late production start proportional to revenue potential
            daily_rev_approx = pad.total_qi_mcfd * 0.5 * config.commodity_price_per_mcf / 1e3
            daily_opex_approx = pad.annual_opex_mm / 365.0 * 1e3
            fcf_coeff = max(1, int((daily_rev_approx - daily_opex_approx) * FCF_W / 1000))
            obj.append(pad_vars[pad.name]["prod_s"] * fcf_coeff)

    model.Minimize(sum(obj))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = config.cpsat_time_limit_seconds
    solver.parameters.num_workers = config.cpsat_num_workers
    solver.parameters.random_seed = 1

    callback = SolutionLogger(pad_vars, pads_subset)
    status = solver.Solve(model, callback)
    solver_stats = extract_solver_stats(solver, status, callback)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        res = [(p.name, solver.Value(pad_vars[p.name]["drill_s"]),
                p.pvi, p.is_mandatory) for p in pads_subset]
        res.sort(key=lambda x: x[1])
        return [r[0] for r in res], callback, solver_stats
    return None, callback, solver_stats


def run_cpsat_tiered(
    config: SimConfig, all_pads: List[WellPad],
    base_production: np.ndarray, minimum_volumes: np.ndarray,
    global_overwrites: Optional[Dict] = None,
) -> Tuple[Optional[List[str]], int, str, Dict]:
    diagnostics = {
        "callback": None, "solver_stats": None,
        "tier_attempts": [], "final_subset_size": 0,
    }
    if not ORTOOLS_AVAILABLE:
        return None, 0, "OR-Tools not available", diagnostics

    sorted_pads = sorted(all_pads, key=lambda p: p.pvi, reverse=True)
    mandatory = [p for p in sorted_pads if p.is_mandatory]
    mandatory_names = {p.name for p in mandatory}

    if config.cpsat_max_pads_override is not None:
        max_n = config.cpsat_max_pads_override
    else:
        max_n = estimate_max_pads(all_pads, config)
    max_n = max(max_n, len(mandatory) + 1)

    candidates = sorted(set([
        max_n,
        max(int(max_n * 0.75), len(mandatory) + 1),
        max(int(max_n * 0.5), len(mandatory) + 1),
        len(mandatory) + 3,
    ]), reverse=True)

    print(f"\n  🧮 CP-SAT TIERED (auto max: {max_n}, mandatory: {len(mandatory)})")

    for n in candidates:
        n = min(n, len(sorted_pads))
        if n < len(mandatory) + 1:
            continue
        non_mandatory = [p for p in sorted_pads if p.name not in mandatory_names]
        top_non_mandatory = non_mandatory[:n - len(mandatory)]
        subset = mandatory + top_non_mandatory

        print(f"     Trying {len(subset)} pads "
              f"({len(mandatory)} mandatory + {len(top_non_mandatory)} by PVI)...")

        order, callback, solver_stats = run_cpsat_on_subset(
            config, subset, base_production, minimum_volumes,
            global_overwrites=global_overwrites)

        diagnostics["tier_attempts"].append({
            "subset_size": len(subset), "callback": callback,
            "solver_stats": solver_stats, "feasible": order is not None,
        })

        if order is not None:
            optimized_names = set(order)
            remaining = [p.name for p in sorted_pads if p.name not in optimized_names]
            full_order = order + remaining

            print(f"     ✅ FEASIBLE with {len(subset)} pads, "
                  f"{len(remaining)} remaining in PVI order")
            print(f"     {'#':>3} {'Pad':20s} {'PVI':>5} {'Mand':>5}")
            for i, nm in enumerate(order):
                pad = next(p for p in subset if p.name == nm)
                print(f"     {i+1:3d} {nm:20s} {pad.pvi:5.2f} "
                      f"{'★' if pad.is_mandatory else ' '}")

            if solver_stats:
                gap = solver_stats.get("optimality_gap_pct", None)
                gap_str = f"{gap:.2f}%" if gap is not None else "N/A"
                n_sols = solver_stats.get("num_solutions_found", 0)
                wt = solver_stats.get("wall_time_sec", 0)
                print(f"\n     📊 CP-SAT Stats: {n_sols} solutions found, "
                      f"gap={gap_str}, time={wt:.1f}s, "
                      f"branches={solver_stats.get('num_branches', 0):,}, "
                      f"conflicts={solver_stats.get('num_conflicts', 0):,}")

            diagnostics["callback"] = callback
            diagnostics["solver_stats"] = solver_stats
            diagnostics["final_subset_size"] = len(subset)

            return full_order, len(subset), f"FEASIBLE ({len(subset)} pads optimized)", diagnostics
        else:
            print(f"     ❌ INFEASIBLE with {len(subset)} pads")
            if solver_stats:
                print(f"        (status={solver_stats['status']}, "
                      f"time={solver_stats.get('wall_time_sec', 0):.1f}s)")

    print(f"\n     🔍 DIAGNOSIS:")
    total_capex = sum(p.total_capex_mm for p in sorted_pads)
    num_years = config.simulation_days // 365
    total_budget = config.effective_annual_capex_mm * num_years
    print(f"       Total pad capex: ${total_capex:,.1f}MM")
    print(f"       Total budget ({num_years}yr): ${total_budget:,.1f}MM")
    total_drill = sum(p.drill_days for p in sorted_pads)
    rig_capacity = config.num_rigs * config.simulation_days
    print(f"       Total drill days: {total_drill:,.0f}, "
          f"Rig capacity: {rig_capacity:,.0f}")

    return None, 0, "INFEASIBLE (all subset sizes failed)", diagnostics


# =============================================================================
# 6. GREEDY PVI RESHUFFLE (FALLBACK)
# =============================================================================

def attempt_pvi_reshuffle(
    config: SimConfig, base_production: np.ndarray, minimum_volumes: np.ndarray,
    current_order: List[str], current_shortfall: float,
    pad_pvi_map: Dict[str, float], pad_cycle_map: Dict[str, float],
    mandatory_pads: set = None,
    template_pads: Optional[List[WellPad]] = None,
    global_overwrites: Optional[Dict] = None,
) -> Tuple[Optional[List[str]], float, List[str]]:
    if mandatory_pads is None:
        mandatory_pads = set()
    tol = config.pvi_reshuffle_tolerance
    min_pvi = config.pvi_reshuffle_min_pvi
    window = config.pvi_reshuffle_window
    sf_tol = config.shortfall_tolerance_mcfd
    best_order, best_sf, swap_log = None, current_shortfall, []
    order = list(current_order)
    improved = True
    while improved:
        improved = False
        best_swap, bsf = None, best_sf
        for i in range(len(order)):
            pi = pad_pvi_map.get(order[i], 0)
            if pi <= min_pvi:
                continue
            if order[i] in mandatory_pads:
                continue
            for j in range(i + 1, min(i + window + 1, len(order))):
                pj = pad_pvi_map.get(order[j], 0)
                if pj <= min_pvi:
                    continue
                if order[j] in mandatory_pads:
                    continue
                if abs(pi - pj) > tol:
                    continue
                if pad_cycle_map.get(order[j], 0) >= pad_cycle_map.get(order[i], 0):
                    continue
                to = list(order)
                to[i], to[j] = to[j], to[i]
                _pads = template_pads if template_pads is not None else _load_fresh_sim(config)[0]
                sim = OrderedDrillingSimulator(_pads, config, base_production,
                                              minimum_volumes, pad_order=to,
                                              global_overwrites=global_overwrites)
                mo = sim.run()
                _, sf = sim.check_minimum_volumes(mo, tolerance=sf_tol)
                ns = sf["avg_shortfall_mcfd"].max() if not sf.empty else 0
                if ns < bsf:
                    best_swap, bsf, bso = (i, j), ns, to
        if best_swap and bsf < best_sf:
            i, j = best_swap
            swap_log.append(f"SWAP [{i+1}] {order[i]} ↔ [{j+1}] {order[j]}")
            order, best_sf = bso, bsf
            best_order = list(order)
            improved = True
            if best_sf <= 0:
                break
    return best_order, best_sf, swap_log


# =============================================================================
# 7. SINGLE-RUN SIMULATION HELPER
# =============================================================================

def run_single_simulation(
    config: SimConfig, base_production: np.ndarray, minimum_volumes: np.ndarray,
    pad_order: Optional[List[str]] = None, label: str = "run",
    template_pads: Optional[List[WellPad]] = None,
    global_overwrites: Optional[Dict] = None,
) -> Dict:
    """
    Run a single simulation with the given config and pad_order.
    Returns a results dict with sim, monthly, shortfall info, FCF, etc.
    """
    if template_pads is not None:
        pads = template_pads
    else:
        pads, _ = _load_fresh_sim(config)
    sim = OrderedDrillingSimulator(pads, config, base_production,
                                  minimum_volumes, pad_order=pad_order,
                                  global_overwrites=global_overwrites)
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


# =============================================================================
# 8. AUTO-ADJUSTMENT ENGINE
# =============================================================================

def run_with_auto_adjustment(
    config: SimConfig, base_production: np.ndarray, minimum_volumes: np.ndarray,
    global_overwrites: Optional[Dict] = None,
) -> Tuple[OrderedDrillingSimulator, pd.DataFrame, List[Dict], List[Dict], Dict, Dict]:
    """
    Returns (sim, monthly, adj_log, all_iters, cpsat_diagnostics, all_strategy_results).

    all_strategy_results is a dict keyed by strategy label, each containing
    the full simulation results dict from run_single_simulation().
    """
    cfg = deepcopy(config)
    adj_log, all_iters = [], []
    iteration, cycle_step = 0, 0
    current_order = None
    current_optimizer = "pvi_default"
    reshuffle_attempted = False
    sf_tol = cfg.shortfall_tolerance_mcfd
    cpsat_diagnostics = {}
    all_strategy_results = {}
    ow = global_overwrites or {}

    print("=" * 80)
    print("AUTO-ADJUSTMENT ENGINE")
    mode = "CP-SAT (tiered) + greedy fallback" if cfg.cpsat_enabled else "Greedy only"
    print(f"  Mode: {mode}")
    print(f"  SF tol: {sf_tol:,.0f} MCFD | Capex: ${cfg.annual_capex_limit_mm}MM "
          f"+ ${cfg.capex_tolerance_mm}MM = ${cfg.effective_annual_capex_mm}MM")
    if cfg.capex_cagr_pct > 0:
        print(f"  Capex CAGR: {cfg.capex_cagr_pct}% for {cfg.capex_cagr_years} years")
    if cfg.overland_midstream_budget_mm > 0:
        print(f"  OL/MS Sub-budget: ${cfg.overland_midstream_budget_mm}MM")
    if cfg.production_ceiling_mcfd > 0:
        print(f"  Production Ceiling: {cfg.production_ceiling_mcfd:,.0f} MCFD "
              f"({cfg.production_cagr_pct}% CAGR, {cfg.production_cagr_years} yr)")
    if ow:
        print(f"  Global Overwrites: years {sorted(ow.keys())}")
    print("=" * 80)

    # Load pad/well data once — OrderedDrillingSimulator deepcopies on each run
    _template_pads, _ = _load_fresh_sim(cfg)

    # Check if any auto-adjustment is possible
    no_room_to_adjust = (
        cfg.num_land_crews >= cfg.max_land_crews and
        cfg.num_permit_crews >= cfg.max_permit_crews and
        cfg.num_construction_crews >= cfg.max_construction_crews and
        cfg.num_rigs >= cfg.max_rigs and
        cfg.num_frac_crews >= cfg.max_frac_crews and
        cfg.annual_capex_limit_mm >= cfg.max_capex_mm)

    while True:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration}")
        print(f"  {cfg.num_land_crews}L {cfg.num_permit_crews}P "
              f"{cfg.num_construction_crews}C {cfg.num_rigs}R "
              f"{cfg.num_frac_crews}F | ${cfg.annual_capex_limit_mm}MM/yr")
        print(f"{'='*60}")

        # ==================================================================
        # PHASE 0: BASELINE — flat PVI rank (always run first each iteration)
        # ==================================================================
        baseline_label = f"baseline_pvi_it{iteration}"
        print(f"\n  📊 Running BASELINE (flat PVI rank)...")
        baseline_result = run_single_simulation(
            cfg, base_production, minimum_volumes,
            pad_order=None,  # None = default PVI sort inside simulator
            label=baseline_label, template_pads=_template_pads,
            global_overwrites=ow)
        all_strategy_results[baseline_label] = baseline_result
        bp = "✅" if baseline_result["passed"] else "❌"
        print(f"     Baseline: {bp} | SF: {baseline_result['raw_max_shortfall']:,.0f} MCFD "
              f"| Prod: {baseline_result['total_prod_mcf']/1e6:,.1f} MMCF "
              f"| Peak: {baseline_result['peak_mcfd']:,.0f} MCFD")

        # ==================================================================
        # PHASE 1: CP-SAT tiered optimization
        # ==================================================================
        cpsat_succeeded = False
        cpsat_n_optimized = 0
        if cfg.cpsat_enabled and ORTOOLS_AVAILABLE:
            cpsat_order, cpsat_n_optimized, cpsat_status, iter_diagnostics = run_cpsat_tiered(
                cfg, _template_pads, base_production, minimum_volumes,
                global_overwrites=ow)

            if cpsat_order is not None:
                current_order = cpsat_order
                current_optimizer = f"cpsat({cpsat_n_optimized})"
                cpsat_succeeded = True
                cpsat_diagnostics = iter_diagnostics
                print(f"  ✅ CP-SAT: {cpsat_status}")

                # Simulate and record the CP-SAT result
                cpsat_label = f"cpsat_it{iteration}"
                cpsat_result = run_single_simulation(
                    cfg, base_production, minimum_volumes,
                    pad_order=cpsat_order, label=cpsat_label, template_pads=_template_pads,
                    global_overwrites=ow)
                all_strategy_results[cpsat_label] = cpsat_result
                cp = "✅" if cpsat_result["passed"] else "❌"
                print(f"     CP-SAT:   {cp} | SF: {cpsat_result['raw_max_shortfall']:,.0f} MCFD "
                      f"| Prod: {cpsat_result['total_prod_mcf']/1e6:,.1f} MMCF "
                      f"| Peak: {cpsat_result['peak_mcfd']:,.0f} MCFD")
            else:
                print(f"  ⚠️  CP-SAT: {cpsat_status} — using fallback order")
                if not cpsat_diagnostics:
                    cpsat_diagnostics = iter_diagnostics

        # ==================================================================
        # PHASE 2: Execute the BEST strategy via greedy simulator
        # ==================================================================
        sim = OrderedDrillingSimulator(_template_pads, cfg, base_production,
                                      minimum_volumes, pad_order=current_order,
                                      global_overwrites=ow)
        monthly = sim.run()
        passed, shortfall = sim.check_minimum_volumes(monthly, tolerance=sf_tol)

        raw_sf = monthly[monthly["avg_min_vol_mcfd"] > 0]
        raw_max = raw_sf["avg_shortfall_mcfd"].max() if not raw_sf.empty else 0
        over_tol = shortfall["avg_shortfall_mcfd"].max() if not shortfall.empty else 0

        all_iters.append({
            "iteration": iteration, "config": deepcopy(cfg),
            "monthly": monthly.copy(), "capex_timeline": sim.get_capex_timeline(),
            "passed": passed, "max_shortfall": raw_max,
            "max_over_tolerance": over_tol,
            "annual_capex": dict(sim.annual_capex_spent),
            "optimizer": current_optimizer,
            "cpsat_pads_optimized": cpsat_n_optimized,
        })

        # ==================================================================
        # PHASE 3: Check result
        # ==================================================================
        if passed:
            status = "WITHIN TOLERANCE" if raw_max > 0 else "ZERO SHORTFALL"
            print(f"\n  ✅ VOLUMES MET ({status}) — {current_optimizer}")
            adj_log.append({"iteration": iteration,
                            "action": f"PASSED — {status}",
                            "land": cfg.num_land_crews, "permit": cfg.num_permit_crews,
                            "construction": cfg.num_construction_crews, "rigs": cfg.num_rigs,
                            "frac": cfg.num_frac_crews, "capex_mm": cfg.annual_capex_limit_mm,
                            "capex_tol_mm": cfg.capex_tolerance_mm,
                            "max_shortfall_mcfd": raw_max, "max_over_tol": 0,
                            "optimizer": current_optimizer})
            break

        print(f"  ❌ SHORTFALL: {over_tol:,.0f} over tol (raw: {raw_max:,.0f}) "
              f"— {current_optimizer}")

        # If starting resources == max resources and capex == max, no adjustments possible
        if no_room_to_adjust:
            print(f"\n  ⚠️  No room to adjust (start == max for all resources & capex) — stopping")
            adj_log.append({"iteration": iteration, "action": "NO ADJUSTMENT POSSIBLE",
                            "land": cfg.num_land_crews, "permit": cfg.num_permit_crews,
                            "construction": cfg.num_construction_crews, "rigs": cfg.num_rigs,
                            "frac": cfg.num_frac_crews, "capex_mm": cfg.annual_capex_limit_mm,
                            "capex_tol_mm": cfg.capex_tolerance_mm,
                            "max_shortfall_mcfd": raw_max, "max_over_tol": over_tol,
                            "optimizer": current_optimizer})
            break

        # ==================================================================
        # PHASE 4: Fallbacks if all resources maxed
        # ==================================================================
        all_maxed = (
            cfg.num_land_crews >= cfg.max_land_crews and
            cfg.num_permit_crews >= cfg.max_permit_crews and
            cfg.num_construction_crews >= cfg.max_construction_crews and
            cfg.num_rigs >= cfg.max_rigs and
            cfg.num_frac_crews >= cfg.max_frac_crews)

        if all_maxed:
            if cfg.pvi_reshuffle_enabled and not reshuffle_attempted:
                reshuffle_attempted = True
                print("  🔄 Fallback: greedy PVI reshuffle...")
                pvi_m = {p.name: p.pvi for p in sim.pads}
                cyc_m = {p.name: p.total_cycle_days for p in sim.pads}
                mand_set = {p.name for p in sim.pads if p.is_mandatory}
                ot = current_order if current_order else sim.get_pad_order()
                new_o, new_sf, swaps = attempt_pvi_reshuffle(
                    cfg, base_production, minimum_volumes, ot, over_tol,
                    pvi_m, cyc_m, mandatory_pads=mand_set,
                    template_pads=_template_pads,
                    global_overwrites=ow)
                if new_o and new_sf < over_tol:
                    current_order = new_o
                    current_optimizer = f"{current_optimizer}+reshuffle"

                    # Record reshuffle result
                    reshuffle_label = f"reshuffle_it{iteration}"
                    reshuffle_result = run_single_simulation(
                        cfg, base_production, minimum_volumes,
                        pad_order=new_o, label=reshuffle_label, template_pads=_template_pads,
                        global_overwrites=ow)
                    all_strategy_results[reshuffle_label] = reshuffle_result
                    rp = "✅" if reshuffle_result["passed"] else "❌"
                    print(f"     Reshuffle: {rp} | SF: {reshuffle_result['raw_max_shortfall']:,.0f} MCFD "
                          f"| Prod: {reshuffle_result['total_prod_mcf']/1e6:,.1f} MMCF")

                    adj_log.append({"iteration": iteration,
                                    "action": f"RESHUFFLE: {len(swaps)} swaps",
                                    "land": cfg.num_land_crews, "permit": cfg.num_permit_crews,
                                    "construction": cfg.num_construction_crews, "rigs": cfg.num_rigs,
                                    "frac": cfg.num_frac_crews, "capex_mm": cfg.annual_capex_limit_mm,
                                    "capex_tol_mm": cfg.capex_tolerance_mm,
                                    "max_shortfall_mcfd": raw_max, "max_over_tol": new_sf,
                                    "optimizer": current_optimizer})
                    continue

            if cfg.annual_capex_limit_mm < cfg.max_capex_mm:
                old = cfg.annual_capex_limit_mm
                cfg.annual_capex_limit_mm = min(
                    cfg.annual_capex_limit_mm + cfg.capex_increment_mm, cfg.max_capex_mm)
                reshuffle_attempted = False
                adj_log.append({"iteration": iteration,
                                "action": f"CAPEX: ${old:.0f}→${cfg.annual_capex_limit_mm:.0f}MM",
                                "land": cfg.num_land_crews, "permit": cfg.num_permit_crews,
                                "construction": cfg.num_construction_crews, "rigs": cfg.num_rigs,
                                "frac": cfg.num_frac_crews, "capex_mm": cfg.annual_capex_limit_mm,
                                "capex_tol_mm": cfg.capex_tolerance_mm,
                                "max_shortfall_mcfd": raw_max, "max_over_tol": over_tol,
                                "optimizer": "capex_increase"})
                if cfg.annual_capex_limit_mm >= cfg.max_capex_mm:
                    print(f"\n  ⚠️  ALL MAXED")
                    break
                continue
            else:
                print(f"\n  ⚠️  ALL MAXED")
                adj_log.append({"iteration": iteration, "action": "FAILED",
                                "land": cfg.num_land_crews, "permit": cfg.num_permit_crews,
                                "construction": cfg.num_construction_crews, "rigs": cfg.num_rigs,
                                "frac": cfg.num_frac_crews, "capex_mm": cfg.annual_capex_limit_mm,
                                "capex_tol_mm": cfg.capex_tolerance_mm,
                                "max_shortfall_mcfd": raw_max, "max_over_tol": over_tol,
                                "optimizer": "none"})
                break

        # ==================================================================
        # PHASE 5: Add resources (skip if already at max — prevents useless re-run)
        # ==================================================================
        old_resources = (cfg.num_land_crews, cfg.num_permit_crews,
                         cfg.num_construction_crews, cfg.num_rigs, cfg.num_frac_crews)

        if cycle_step == 0:
            cfg.num_land_crews = min(cfg.num_land_crews + 1, cfg.max_land_crews)
            cfg.num_permit_crews = min(cfg.num_permit_crews + 1, cfg.max_permit_crews)
            cfg.num_construction_crews = min(cfg.num_construction_crews + 1, cfg.max_construction_crews)
            action = f"PRE-DRILL +1: L→{cfg.num_land_crews} P→{cfg.num_permit_crews} C→{cfg.num_construction_crews}"
        elif cycle_step == 1:
            cfg.num_rigs = min(cfg.num_rigs + 1, cfg.max_rigs)
            action = f"RIG +1: →{cfg.num_rigs}"
        elif cycle_step == 2:
            cfg.num_frac_crews = min(cfg.num_frac_crews + 1, cfg.max_frac_crews)
            action = f"FRAC +1: →{cfg.num_frac_crews}"

        new_resources = (cfg.num_land_crews, cfg.num_permit_crews,
                         cfg.num_construction_crews, cfg.num_rigs, cfg.num_frac_crews)

        if new_resources == old_resources:
            # Resources didn't actually change (already at max) — no point re-running
            print(f"  ⚠️  Resources unchanged (already at max) — skipping to fallbacks")
            cycle_step = (cycle_step + 1) % 3
            continue

        print(f"  → {action}")
        adj_log.append({"iteration": iteration, "action": action,
                        "land": cfg.num_land_crews, "permit": cfg.num_permit_crews,
                        "construction": cfg.num_construction_crews, "rigs": cfg.num_rigs,
                        "frac": cfg.num_frac_crews, "capex_mm": cfg.annual_capex_limit_mm,
                        "capex_tol_mm": cfg.capex_tolerance_mm,
                        "max_shortfall_mcfd": raw_max, "max_over_tol": over_tol,
                        "optimizer": current_optimizer})
        cycle_step = (cycle_step + 1) % 3

    return sim, monthly, adj_log, all_iters, cpsat_diagnostics, all_strategy_results


def print_adjustment_report(adj_log: List[Dict], config: SimConfig):
    print("\n" + "=" * 80)
    print("ADJUSTMENT REPORT")
    print("=" * 80)
    print(f"  Initial: {config.num_land_crews}L {config.num_permit_crews}P "
          f"{config.num_construction_crews}C {config.num_rigs}R "
          f"{config.num_frac_crews}F | ${config.annual_capex_limit_mm}MM "
          f"(+${config.capex_tolerance_mm}MM)")
    print(f"  CP-SAT: {'TIERED (auto)' if config.cpsat_enabled else 'DISABLED'}"
          f"{f' (override: {config.cpsat_max_pads_override})' if config.cpsat_max_pads_override else ''}")
    if not adj_log:
        print("  No adjustments.")
        return
    print(f"\n  {'It':>3} | {'Action':45s} | {'L':>2} {'P':>2} {'C':>2} "
          f"{'R':>2} {'F':>2} | {'Budget':>8} | {'Raw SF':>10} | {'Optimizer':>18}")
    print(f"  {'─'*3} | {'─'*45} | {'─'*2} {'─'*2} {'─'*2} {'─'*2} {'─'*2} "
          f"| {'─'*8} | {'─'*10} | {'─'*18}")
    for e in adj_log:
        print(f"  {e['iteration']:3d} | {e['action']:45s} | "
              f"{e['land']:2d} {e['permit']:2d} {e['construction']:2d} "
              f"{e['rigs']:2d} {e['frac']:2d} | "
              f"${e['capex_mm']:>6.0f}MM | {e['max_shortfall_mcfd']:>8,.0f} | "
              f"{e.get('optimizer',''):>18}")
    f = adj_log[-1]
    print(f"\n  Final: {f['land']}L {f['permit']}P {f['construction']}C "
          f"{f['rigs']}R {f['frac']}F | ${f['capex_mm']}MM | {f['optimizer']}")
    print(f"  {'✅ MET' if 'PASSED' in f['action'] else '❌ NOT MET'}")
    print("=" * 80)


# =============================================================================
# 9. STRATEGY COMPARISON (NEW — compares baseline PVI vs CP-SAT vs reshuffle)
# =============================================================================

def print_strategy_comparison(all_strategy_results: Dict, config: SimConfig):
    """Print a side-by-side comparison table of all strategies tried."""
    if not all_strategy_results:
        print("No strategy results to compare.")
        return

    print("\n" + "=" * 120)
    print("STRATEGY COMPARISON — All Approaches Tried")
    print("=" * 120)

    rows = []
    for label, r in all_strategy_results.items():
        rows.append({
            "Strategy": label,
            "Passed": "✅" if r["passed"] else "❌",
            "Max SF (MCFD)": r["raw_max_shortfall"],
            "Over Tol (MCFD)": r["over_tol_shortfall"],
            "Total Prod (MMCF)": r["total_prod_mcf"] / 1e6,
            "Peak (MCFD)": r["peak_mcfd"],
            "FCF (MM$)": r.get("total_fcf_mm", 0),
            "Prod Ceiling OK": "✅" if r.get("prod_ceiling_ok", True) else "❌",
            "First 3 Pads": " → ".join(r["pad_order"][:3]),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Identify the best result
    passing = [r for r in rows if r["Passed"] == "✅"]
    if passing:
        best = min(passing, key=lambda x: x["Max SF (MCFD)"])
        print(f"\n  🏆 Best passing strategy: {best['Strategy']}")
    else:
        best = min(rows, key=lambda x: x["Max SF (MCFD)"])
        print(f"\n  ⚠️  No strategy passed. Least shortfall: {best['Strategy']}")

    # Compare final iteration's baseline vs best optimizer
    final_baselines = [k for k in all_strategy_results if k.startswith("baseline_pvi")]
    final_optimizers = [k for k in all_strategy_results if not k.startswith("baseline_pvi")]
    if final_baselines and final_optimizers:
        bl = all_strategy_results[final_baselines[-1]]
        opt = all_strategy_results[final_optimizers[-1]]
        prod_diff = opt["total_prod_mcf"] - bl["total_prod_mcf"]
        sf_diff = bl["raw_max_shortfall"] - opt["raw_max_shortfall"]
        print(f"\n  📊 Final Baseline vs Best Optimizer:")
        print(f"     Production delta:  {prod_diff/1e6:+,.1f} MMCF "
              f"({'better' if prod_diff > 0 else 'worse'})")
        print(f"     Shortfall delta:   {sf_diff:+,.0f} MCFD "
              f"({'improved' if sf_diff > 0 else 'degraded'})")

    print("=" * 120)
    return df


def plot_strategy_comparison(all_strategy_results: Dict, config: SimConfig,
                             plot_folder: str = ""):
    """Multi-panel chart comparing all strategies tried."""
    if not all_strategy_results or len(all_strategy_results) < 2:
        print("Not enough strategies to compare — skipping strategy comparison plot.")
        return

    labels = list(all_strategy_results.keys())
    results = [all_strategy_results[k] for k in labels]

    # Use short labels for display
    short_labels = []
    for lbl in labels:
        if "baseline" in lbl:
            short_labels.append("PVI " + lbl.split("_it")[-1])
        elif "cpsat" in lbl:
            short_labels.append("CPSAT " + lbl.split("_it")[-1])
        elif "reshuffle" in lbl:
            short_labels.append("Reshuf " + lbl.split("_it")[-1])
        else:
            short_labels.append(lbl[:15])

    n_strats = len(results)
    cmap = plt.cm.Set2(np.linspace(0, 1, max(n_strats, 3)))

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"Strategy Comparison — {n_strats} approaches "
                 f"({config.num_rigs}R {config.num_frac_crews}F "
                 f"${config.annual_capex_limit_mm}MM)",
                 fontsize=14, fontweight="bold")

    # --- Panel 1: Production profiles overlay ---
    ax = axes[0, 0]
    for i, (r, sl) in enumerate(zip(results, short_labels)):
        m = r["monthly"]
        is_baseline = "baseline" in labels[i] or "PVI" in sl
        lw = 1.5 if is_baseline else 2.5
        ls = "--" if is_baseline else "-"
        ax.plot(m["month"], m["avg_total_mcfd"], ls, lw=lw, color=cmap[i], label=sl)
    mm = results[0]["monthly"]["avg_min_vol_mcfd"] > 0
    if mm.any():
        ax.plot(results[0]["monthly"].loc[mm, "month"],
                results[0]["monthly"].loc[mm, "avg_min_vol_mcfd"],
                "r--", lw=2, label="Min Volume", zorder=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("MCFD")
    ax.set_title("Production Profiles")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Shortfall overlay ---
    ax = axes[0, 1]
    for i, (r, sl) in enumerate(zip(results, short_labels)):
        m = r["monthly"]
        is_baseline = "baseline" in labels[i] or "PVI" in sl
        lw = 1.5 if is_baseline else 2.5
        ls = "--" if is_baseline else "-"
        ax.plot(m["month"], m["avg_shortfall_mcfd"], ls, lw=lw, color=cmap[i], label=sl)
    ax.axhline(y=config.shortfall_tolerance_mcfd, color="orange", ls="--", lw=2, label="SF Tol")
    ax.set_xlabel("Month")
    ax.set_ylabel("Shortfall (MCFD)")
    ax.set_title("Shortfall Comparison")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Bar chart summary ---
    ax = axes[1, 0]
    x = np.arange(n_strats)
    bar_width = 0.35
    sfs = [r["raw_max_shortfall"] for r in results]
    prods = [r["total_prod_mcf"] / 1e6 for r in results]
    bar_colors = ["lightcoral" if not r["passed"] else "lightgreen" for r in results]

    ax2 = ax.twinx()
    bars = ax.bar(x - bar_width/2, sfs, bar_width, color=bar_colors,
                  edgecolor="black", alpha=0.8, label="Max Shortfall")
    ax2.bar(x + bar_width/2, prods, bar_width, color=[cmap[i] for i in range(n_strats)],
            edgecolor="black", alpha=0.6, label="Total Prod")
    ax.axhline(y=config.shortfall_tolerance_mcfd, color="orange", ls="--", lw=2)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Max Shortfall (MCFD)")
    ax2.set_ylabel("Total Production (MMCF)")
    ax.set_title("Shortfall vs Production by Strategy")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Pad ordering comparison heatmap ---
    ax = axes[1, 1]
    # Use the last baseline as the reference ordering
    ref_order = results[0]["pad_order"]
    n_pads_show = min(15, len(ref_order))
    ordering_matrix = []
    for r in results:
        order_map = {name: pos for pos, name in enumerate(r["pad_order"])}
        positions = [order_map.get(p, len(r["pad_order"])) for p in ref_order[:n_pads_show]]
        ordering_matrix.append(positions)
    ordering_matrix = np.array(ordering_matrix)
    im = ax.imshow(ordering_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(n_pads_show))
    ax.set_xticklabels(ref_order[:n_pads_show], rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_title(f"Pad Sequence Position (top {n_pads_show})")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Position in order", fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(plot_folder, "strategy_comparison.png") if plot_folder else "strategy_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  📊 Saved: {save_path}")


# =============================================================================
# 10. CP-SAT DIAGNOSTICS VISUALIZATION
# =============================================================================

def plot_cpsat_diagnostics(cpsat_diagnostics: Dict, config: SimConfig,
                           plot_folder: str = ""):
    callback = cpsat_diagnostics.get("callback")
    solver_stats = cpsat_diagnostics.get("solver_stats")

    if not callback or not callback.solutions:
        print("No CP-SAT solutions to plot — skipping diagnostics chart.")
        return

    sols = callback.solutions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("CP-SAT Optimization Diagnostics", fontsize=14, fontweight="bold")

    # Panel 1: Convergence
    ax = axes[0, 0]
    times = [s["wall_time"] for s in sols]
    objs = [s["objective"] for s in sols]
    bounds = [s["best_bound"] for s in sols]
    ax.plot(times, objs, "bo-", lw=2, markersize=5, label="Objective (incumbent)")
    ax.plot(times, bounds, "r--", lw=1.5, label="Best bound (lower)")
    ax.fill_between(times, bounds, objs, alpha=0.15, color="orange", label="Optimality gap")
    ax.set_xlabel("Wall Time (s)")
    ax.set_ylabel("Objective Value")
    gap_pct = solver_stats.get("optimality_gap_pct", 0) if solver_stats else 0
    gap_str = f"{gap_pct:.2f}%" if gap_pct is not None else "N/A"
    ax.set_title(f"Convergence ({len(sols)} solutions, final gap: {gap_str})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Pad ordering evolution
    ax = axes[0, 1]
    for i, sol in enumerate(sols):
        alpha = 0.2 if i < len(sols) - 1 else 1.0
        lw = 0.8 if i < len(sols) - 1 else 2.5
        color = "gray" if i < len(sols) - 1 else "blue"
        drill_starts = [d["drill_start"] for d in sol["pad_details"]]
        names = [d["pad"] for d in sol["pad_details"]]
        label = ""
        if i == 0:
            label = f"Sol 1 (first)"
        elif i == len(sols) - 1:
            label = f"Sol {i+1} (best)"
        ax.plot(drill_starts, names, "o-", alpha=alpha, lw=lw, color=color,
                markersize=3, label=label)
    ax.set_xlabel("Drill Start Day")
    ax.set_title("Pad Ordering Evolution Across Solutions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Final drill timing
    ax = axes[1, 0]
    final = sols[-1]
    pads_by_drill = sorted(final["pad_details"], key=lambda x: x["drill_start"])
    names = [p["pad"] for p in pads_by_drill]
    drill_days = [p["drill_start"] for p in pads_by_drill]
    pvis = [p["pvi"] for p in pads_by_drill]
    colors = ["gold" if p["pvi"] > 1.4 else "steelblue" for p in pads_by_drill]
    bars = ax.barh(names, drill_days, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    for bar, pvi, pad_info in zip(bars, pvis, pads_by_drill):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f"PVI={pvi:.2f}  qi={pad_info['qi_mcfd']:,.0f}",
                va="center", fontsize=6)
    ax.set_xlabel("Drill Start Day")
    ax.set_title("Final Solution: Drill Timing (gold=high PVI)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # Panel 4: Stats summary
    ax = axes[1, 1]
    ax.axis("off")
    if solver_stats:
        obj_val = solver_stats.get("objective_value", "N/A")
        obj_str = f"{obj_val:,.0f}" if isinstance(obj_val, (int, float)) and obj_val is not None else "N/A"
        bound_val = solver_stats.get("best_objective_bound", "N/A")
        bound_str = f"{bound_val:,.0f}" if isinstance(bound_val, (int, float)) else "N/A"
        gap_val = solver_stats.get("optimality_gap_pct", None)
        gap_display = f"{gap_val:.2f}%" if gap_val is not None else "N/A"
        stats_text = (
            f"Solver Statistics\n"
            f"{'─' * 40}\n"
            f"Status:            {solver_stats.get('status', 'N/A')}\n"
            f"Objective:         {obj_str}\n"
            f"Best Bound:        {bound_str}\n"
            f"Optimality Gap:    {gap_display}\n"
            f"Wall Time:         {solver_stats.get('wall_time_sec', 0):.1f}s\n"
            f"Solutions Found:   {solver_stats.get('num_solutions_found', 0)}\n"
            f"Branches:          {solver_stats.get('num_branches', 0):,}\n"
            f"Conflicts:         {solver_stats.get('num_conflicts', 0):,}\n"
            f"Bool Propagations: {solver_stats.get('num_boolean_propagations', 0):,}\n"
            f"{'─' * 40}\n"
            f"Gap = 0%  → PROVEN OPTIMAL\n"
            f"Gap > 0%  → best found within time limit\n"
            f"{'─' * 40}\n"
            f"Subset Size:       {cpsat_diagnostics.get('final_subset_size', '?')}\n"
            f"Tier Attempts:     {len(cpsat_diagnostics.get('tier_attempts', []))}"
        )
    else:
        stats_text = "No solver statistics available."
    ax.text(0.05, 0.5, stats_text, fontsize=10, family="monospace",
            va="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    save_path = os.path.join(plot_folder, "cpsat_diagnostics.png") if plot_folder else "cpsat_diagnostics.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  📊 Saved: {save_path}")


def print_cpsat_diagnostics_report(cpsat_diagnostics: Dict):
    solver_stats = cpsat_diagnostics.get("solver_stats")
    callback = cpsat_diagnostics.get("callback")
    tier_attempts = cpsat_diagnostics.get("tier_attempts", [])

    print("\n" + "=" * 80)
    print("CP-SAT DIAGNOSTICS REPORT")
    print("=" * 80)

    if tier_attempts:
        print(f"\n  Tier Attempts: {len(tier_attempts)}")
        for i, ta in enumerate(tier_attempts):
            ts = ta.get("solver_stats", {})
            feas = "✅ FEASIBLE" if ta["feasible"] else "❌ INFEASIBLE"
            n_sols = ts.get("num_solutions_found", 0) if ts else 0
            wt = ts.get("wall_time_sec", 0) if ts else 0
            print(f"    Tier {i+1}: {ta['subset_size']} pads → {feas} "
                  f"({n_sols} solutions, {wt:.1f}s)")

    if solver_stats:
        print(f"\n  Final Solver Stats:")
        print(f"    Status:            {solver_stats.get('status', 'N/A')}")
        obj = solver_stats.get('objective_value')
        print(f"    Objective:         {obj:,.0f}" if obj is not None else "    Objective:         N/A")
        bound = solver_stats.get('best_objective_bound')
        print(f"    Best Bound:        {bound:,.0f}" if isinstance(bound, (int, float)) else f"    Best Bound:        {bound}")
        gap = solver_stats.get('optimality_gap_pct')
        if gap is not None:
            print(f"    Optimality Gap:    {gap:.4f}%")
            if gap == 0:
                print(f"    → ✅ PROVEN OPTIMAL — no better solution exists")
            elif gap < 1:
                print(f"    → ✅ Near-optimal — at most {gap:.2f}% from the true optimum")
            elif gap < 5:
                print(f"    → ⚠️  Small gap — solution is good but up to {gap:.1f}% improvement possible")
            else:
                print(f"    → ❌ Large gap — consider increasing cpsat_time_limit_seconds")
        print(f"    Wall Time:         {solver_stats.get('wall_time_sec', 0):.1f}s")
        print(f"    Solutions Found:   {solver_stats.get('num_solutions_found', 0)}")
        print(f"    Branches:          {solver_stats.get('num_branches', 0):,}")
        print(f"    Conflicts:         {solver_stats.get('num_conflicts', 0):,}")
        print(f"    Bool Propagations: {solver_stats.get('num_boolean_propagations', 0):,}")

    if callback and callback.solutions:
        print(f"\n  Solution Progression:")
        print(f"    {'#':>4} {'Objective':>14} {'Bound':>14} {'Gap%':>8} {'Time':>8} {'First 3 Pads'}")
        print(f"    {'─'*4} {'─'*14} {'─'*14} {'─'*8} {'─'*8} {'─'*30}")
        for s in callback.solutions:
            first3 = " → ".join(s["pad_order"][:3])
            print(f"    {s['solution_num']:4d} {s['objective']:14,.0f} "
                  f"{s['best_bound']:14,.0f} {s['gap_pct']:7.2f}% "
                  f"{s['wall_time']:7.1f}s {first3}")

        if len(callback.solutions) >= 2:
            first_order = callback.solutions[0]["pad_order"]
            last_order = callback.solutions[-1]["pad_order"]
            if first_order != last_order:
                diffs = [(i, f, l) for i, (f, l) in enumerate(zip(first_order, last_order)) if f != l]
                print(f"\n  Order Changes (first vs best): {len(diffs)} positions differ")
                for pos, f_pad, l_pad in diffs[:10]:
                    print(f"    Position {pos+1}: {f_pad} → {l_pad}")
            else:
                print(f"\n  Order unchanged across all {len(callback.solutions)} solutions")

    print("=" * 80)


def plot_cpsat_solution_comparison(cpsat_diagnostics: Dict, config: SimConfig,
                                   base_production: np.ndarray,
                                   minimum_volumes: np.ndarray,
                                   top_k: int = 5,
                                   baseline_result: Optional[Dict] = None,
                                   plot_folder: str = "",
                                   global_overwrites: Optional[Dict] = None):
    """
    Simulate the top-K CP-SAT solutions side by side and compare outcomes.
    Overlays baseline PVI result if provided.
    """
    callback = cpsat_diagnostics.get("callback")
    if not callback or len(callback.solutions) < 2:
        print("Not enough CP-SAT solutions to compare — skipping comparison.")
        return None

    n_sols = len(callback.solutions)
    indices = sorted(set(
        [0] +
        list(range(0, n_sols, max(1, n_sols // top_k))) +
        [n_sols - 1]
    ))
    if len(indices) > top_k:
        step = max(1, len(indices) // top_k)
        indices = sorted(set(indices[::step] + [0, n_sols - 1]))

    print(f"\n  📊 Comparing {len(indices)} CP-SAT solutions via full simulation...")

    comparison = []
    sim_results = []
    template_pads = load_pads_from_csv(config.pad_filepath, config.simulation_start_date)
    template_wells = load_wells_from_csv(config.well_filepath)
    assign_wells_to_pads(template_pads, template_wells)
    for idx in indices:
        sol = callback.solutions[idx]
        order = sol["pad_order"]

        sim = OrderedDrillingSimulator(template_pads, config, base_production,
                                      minimum_volumes, pad_order=order,
                                      global_overwrites=global_overwrites)
        monthly = sim.run()
        passed, shortfall = sim.check_minimum_volumes(
            monthly, tolerance=config.shortfall_tolerance_mcfd)

        max_sf = shortfall["avg_shortfall_mcfd"].max() if not shortfall.empty else 0
        raw_sf = monthly[monthly["avg_min_vol_mcfd"] > 0]
        raw_max = raw_sf["avg_shortfall_mcfd"].max() if not raw_sf.empty else 0
        total_prod = monthly["monthly_volume_mcf"].sum()
        peak_prod = monthly["avg_total_mcfd"].max()

        comparison.append({
            "sol_#": sol["solution_num"],
            "objective": sol["objective"],
            "gap_%": sol["gap_pct"],
            "meets_vols": "✅" if passed else "❌",
            "max_shortfall": raw_max,
            "over_tol": max_sf,
            "total_prod_MCF": total_prod,
            "peak_MCFD": peak_prod,
            "first_3_pads": " → ".join(order[:3]),
            "time_s": sol["wall_time"],
        })
        sim_results.append({
            "sol_num": sol["solution_num"],
            "monthly": monthly,
            "order": order,
            "objective": sol["objective"],
            "passed": passed,
        })

    df = pd.DataFrame(comparison)
    print("\n" + "=" * 110)
    print("CP-SAT SOLUTION COMPARISON (simulated)")
    print("=" * 110)
    print(df.to_string(index=False))
    print("=" * 110)

    if len(sim_results) >= 2:
        n_cpsat = len(sim_results)
        has_baseline = baseline_result is not None
        title_extra = f" + Baseline PVI" if has_baseline else ""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"CP-SAT Solution Comparison ({n_cpsat} solutions simulated{title_extra})",
                     fontsize=13, fontweight="bold")
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, n_cpsat))

        # Panel 1: Production profiles overlay
        ax = axes[0, 0]
        if has_baseline:
            bm = baseline_result["monthly"]
            ax.plot(bm["month"], bm["avg_total_mcfd"], "k-", lw=2.5, alpha=0.8,
                    label="Baseline PVI", zorder=14)
        for i, sr in enumerate(sim_results):
            m = sr["monthly"]
            is_best = (i == n_cpsat - 1)
            lw = 3 if is_best else 1.2
            alpha = 1.0 if is_best else 0.5
            ls = "-" if is_best else "--"
            label = f"Sol {sr['sol_num']}" + (" (BEST)" if is_best else "")
            ax.plot(m["month"], m["avg_total_mcfd"], ls, lw=lw, alpha=alpha,
                    color=cmap[i], label=label)
        mm = sim_results[-1]["monthly"]["avg_min_vol_mcfd"] > 0
        if mm.any():
            ax.plot(sim_results[-1]["monthly"].loc[mm, "month"],
                    sim_results[-1]["monthly"].loc[mm, "avg_min_vol_mcfd"],
                    "r--", lw=2, label="Min Volume", zorder=15)
        ax.set_xlabel("Month")
        ax.set_ylabel("MCFD")
        ax.set_title("Production Profiles")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

        # Panel 2: Shortfall profiles overlay
        ax = axes[0, 1]
        if has_baseline:
            bm = baseline_result["monthly"]
            ax.plot(bm["month"], bm["avg_shortfall_mcfd"], "k-", lw=2.5, alpha=0.8,
                    label="Baseline PVI", zorder=14)
        for i, sr in enumerate(sim_results):
            m = sr["monthly"]
            is_best = (i == n_cpsat - 1)
            lw = 3 if is_best else 1.2
            alpha = 1.0 if is_best else 0.5
            ax.plot(m["month"], m["avg_shortfall_mcfd"], "-", lw=lw, alpha=alpha,
                    color=cmap[i], label=f"Sol {sr['sol_num']}")
        ax.axhline(y=config.shortfall_tolerance_mcfd, color="orange", ls="--",
                    lw=2, label="SF Tol")
        ax.set_xlabel("Month")
        ax.set_ylabel("Shortfall (MCFD)")
        ax.set_title("Shortfall Comparison")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

        # Panel 3: Objective vs total production scatter
        ax = axes[1, 0]
        for i, row in enumerate(comparison):
            is_best = (i == len(comparison) - 1)
            ms = 150 if is_best else 60
            ec = "red" if is_best else "black"
            ax.scatter(row["objective"], row["total_prod_MCF"] / 1e6,
                       s=ms, c=[cmap[i]], edgecolors=ec,
                       linewidths=1.5 if is_best else 0.5,
                       marker="*" if is_best else "o",
                       zorder=10 if is_best else 5)
            ax.annotate(f"Sol {row['sol_#']}",
                        (row["objective"], row["total_prod_MCF"]/1e6),
                        fontsize=7, textcoords="offset points", xytext=(5, 5))
        if has_baseline:
            bl_prod = baseline_result["total_prod_mcf"] / 1e6
            ax.axhline(y=bl_prod, color="black", ls="--", lw=1.5, alpha=0.7,
                        label=f"Baseline Prod ({bl_prod:,.0f})")
            ax.legend(fontsize=6)
        ax.set_xlabel("CP-SAT Objective (lower=better)")
        ax.set_ylabel("Total Production (MMCF)")
        ax.set_title("Objective vs Production Tradeoff")
        ax.grid(True, alpha=0.3)

        # Panel 4: Pad ordering comparison (heatmap style)
        ax = axes[1, 1]
        all_pads_in_order = sim_results[-1]["order"]
        n_pads_show = min(15, len(all_pads_in_order))
        sol_labels = []
        ordering_matrix = []
        # Add baseline as first row if available
        if has_baseline:
            bl_order_map = {name: pos for pos, name in enumerate(baseline_result["pad_order"])}
            bl_positions = [bl_order_map.get(p, len(baseline_result["pad_order"]))
                           for p in all_pads_in_order[:n_pads_show]]
            ordering_matrix.append(bl_positions)
            sol_labels.append("Baseline PVI")
        for sr in sim_results:
            order_map = {name: pos for pos, name in enumerate(sr["order"])}
            positions = [order_map.get(p, len(sr["order"]))
                        for p in all_pads_in_order[:n_pads_show]]
            ordering_matrix.append(positions)
            sol_labels.append(f"Sol {sr['sol_num']}")
        ordering_matrix = np.array(ordering_matrix)
        im = ax.imshow(ordering_matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(n_pads_show))
        ax.set_xticklabels(all_pads_in_order[:n_pads_show], rotation=45,
                           ha="right", fontsize=6)
        ax.set_yticks(range(len(sol_labels)))
        ax.set_yticklabels(sol_labels, fontsize=8)
        ax.set_title(f"Pad Sequence Position (top {n_pads_show})")
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Position in order", fontsize=8)

        plt.tight_layout()
        save_path = os.path.join(plot_folder, "cpsat_solution_comparison.png") if plot_folder else "cpsat_solution_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  📊 Saved: {save_path}")

    return df




# =============================================================================
# 11. ORIGINAL VISUALIZATION
# =============================================================================

def plot_final_results(results: pd.DataFrame, sim: OrderedDrillingSimulator,
                       config: SimConfig, optimizer_used: str, plot_folder: str = ""):
    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.suptitle(f"FINAL — {optimizer_used} | ${config.annual_capex_limit_mm}MM "
                 f"(+${config.capex_tolerance_mm}MM) | SF tol: {config.shortfall_tolerance_mcfd:,.0f}",
                 fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    ax.fill_between(results["month"], results["avg_base_mcfd"], alpha=0.3, color="gray", label="Base")
    ax.fill_between(results["month"], results["avg_base_mcfd"],
                    results["avg_total_mcfd"], alpha=0.3, color="green", label="New")
    ax.plot(results["month"], results["avg_total_mcfd"], "g-", lw=2, label="Total")
    mm = results["avg_min_vol_mcfd"] > 0
    if mm.any():
        ax.plot(results.loc[mm, "month"], results.loc[mm, "avg_min_vol_mcfd"], "r--", lw=2, label="Min")
        ax.fill_between(results.loc[mm, "month"],
                        results.loc[mm, "avg_min_vol_mcfd"] - config.shortfall_tolerance_mcfd,
                        results.loc[mm, "avg_min_vol_mcfd"],
                        alpha=0.15, color="red", label=f"Tol")
    # Production ceiling (CAGR-adjusted)
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
        # Plot CAGR-adjusted budget lines
        years = sorted(pv.index)
        budgets = [config.capex_limit_for_year(yr, sim.global_overwrites) for yr in years]
        ceilings = [b + config.capex_tolerance_mm for b in budgets]
        x_pos = range(len(years))
        ax.plot(x_pos, budgets, "r--", lw=2, label="Budget (CAGR)", marker="o", markersize=3)
        ax.plot(x_pos, ceilings, "orange", ls=":", lw=2, label="Ceiling", marker="s", markersize=3)
    ax.set_title("Capex (CAGR-adjusted)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(results["month"], results["cumulative_mcf"] / 1e6, "b-", lw=2, label="Cum. Production")
    if "cumulative_fcf_mm" in results.columns:
        ax2 = ax.twinx()
        ax2.plot(results["month"], results["cumulative_fcf_mm"], "g-", lw=2, label="Cum. FCF ($MM)")
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


def plot_iteration_overlay(all_iters: List[Dict], config: SimConfig, plot_folder: str = ""):
    n = len(all_iters)
    if n == 0:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"ALL ITERATIONS ({n})", fontsize=13, fontweight="bold")
    faded = plt.cm.tab10(np.linspace(0, 1, max(n - 1, 1)))
    fc = "green" if all_iters[-1]["passed"] else "red"

    ax = axes[0, 0]
    for i, it in enumerate(all_iters):
        m = it["monthly"]
        c = it["config"]
        lbl = f"It {it['iteration']}: {c.num_rigs}R {c.num_frac_crews}F [{it['optimizer'][:10]}]"
        if i == n - 1:
            ax.plot(m["month"], m["avg_total_mcfd"], "-", lw=3, color=fc,
                    label=lbl+" (FINAL)", zorder=10)
        else:
            ax.plot(m["month"], m["avg_total_mcfd"], "-", lw=1.5, alpha=0.6,
                    color=faded[i], label=lbl)
    mm = all_iters[-1]["monthly"]["avg_min_vol_mcfd"] > 0
    if mm.any():
        ax.plot(all_iters[-1]["monthly"].loc[mm, "month"],
                all_iters[-1]["monthly"].loc[mm, "avg_min_vol_mcfd"],
                "k--", lw=2, label="Min", zorder=11)
    ax.set_title("Total Production")
    ax.legend(fontsize=5, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for i, it in enumerate(all_iters):
        m = it["monthly"]
        if i == n - 1:
            ax.plot(m["month"], m["avg_new_mcfd"], "-", lw=3, color=fc,
                    label=f"It {it['iteration']} (FINAL)", zorder=10)
        else:
            ax.plot(m["month"], m["avg_new_mcfd"], "-", lw=1.5, alpha=0.6,
                    color=faded[i], label=f"It {it['iteration']}")
    ax.set_title("New Well Production (MCFD)")
    ax.legend(fontsize=5, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    max_yr = max((max(it["annual_capex"].keys()) + 1 if it["annual_capex"] else 0)
                 for it in all_iters)
    if max_yr > 0:
        x = np.arange(max_yr)
        w = 0.8 / max(n, 1)
        for i, it in enumerate(all_iters):
            capex = [it["annual_capex"].get(yr, 0) for yr in range(max_yr)]
            if i == n - 1:
                ax.bar(x + i*w, capex, w, alpha=0.9, color=fc,
                       label=f"It {it['iteration']} (FINAL)", edgecolor="black", lw=1.5)
            else:
                ax.bar(x + i*w, capex, w, alpha=0.4, color=faded[i],
                       label=f"It {it['iteration']}")
        ax.axhline(y=config.annual_capex_limit_mm, color="red", ls="--", lw=2, label="Budget")
        ax.axhline(y=config.effective_annual_capex_mm, color="orange", ls=":", lw=2, label="Ceiling")
        ax.set_xticks(x + (n-1)*w/2)
        ax.set_xticklabels([f"Yr {yr}" for yr in range(max_yr)], fontsize=8)
    ax.set_title("Capex")
    ax.legend(fontsize=5, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    iters = [it["iteration"] for it in all_iters]
    sfs = [it["max_shortfall"] for it in all_iters]
    bc = [fc if i == n-1 else "lightcoral" for i in range(n)]
    bars = ax.bar(iters, sfs, color=bc, alpha=0.7, edgecolor="black")
    if n > 0:
        bars[-1].set_linewidth(2.5)
    ax.axhline(y=config.shortfall_tolerance_mcfd, color="orange", ls="--", lw=2, label="SF Tol")
    for i, it in enumerate(all_iters):
        c = it["config"]
        ax.text(it["iteration"], sfs[i] + max(sfs)*0.02,
                f"{c.num_rigs}R {c.num_frac_crews}F\n{it['optimizer'][:8]}",
                ha="center", va="bottom", fontsize=5)
    ax.set_title("Shortfall + Optimizer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(plot_folder, "iteration_overlay.png") if plot_folder else "iteration_overlay.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# 12. MAIN
# =============================================================================

def main():

    # =========================================================================
    # ALL INPUTS — EDIT THIS ONE BLOCK ONLY
    # =========================================================================
    config = SimConfig(
        # Starting resources
        num_rigs=1,
        num_frac_crews=1,
        num_land_crews=3,
        num_permit_crews=3,
        num_construction_crews=3,

        # Max resources
        max_rigs=1,
        max_frac_crews=1,
        max_land_crews=3,
        max_permit_crews=3,
        max_construction_crews=3,

        # Capital (base year)
        annual_capex_limit_mm=350,
        max_capex_mm=350,
        capex_increment_mm=100,
        capex_tolerance_mm=10,

        # Capital CAGR
        capex_cagr_pct=5.0,           # 5% compound annual growth rate
        capex_cagr_years=10,          # years before CAGR drops to 0%

        # Overland / Midstream separate sub-budget
        overland_midstream_budget_mm=25.0,  # $25MM separate OL/MS budget within total
        ol_ms_cagr_pct=5.0,
        ol_ms_cagr_years=10,

        # Production ceiling
        production_ceiling_mcfd=1_500_000,   # 1.5 BCFD max annual avg production
        production_cagr_pct=5.0,
        production_cagr_years=10,

        # Free cashflow
        commodity_price_per_mcf=3.0,   # $/MCF

        # Shortfall tolerance
        shortfall_tolerance_mcfd=10000,

        # CP-SAT objective weights
        cpsat_pvi_weight_multiplier=500,   # Higher = stronger PVI bias
        cpsat_shortfall_weight=1000,        # Lower = less shortfall penalty
        cpsat_fcf_weight=100,               # FCF maximization weight

        # PVI reshuffling (FALLBACK — only if CP-SAT fails)
        pvi_reshuffle_enabled=True,
        pvi_reshuffle_tolerance=0.3,
        pvi_reshuffle_min_pvi=1.3,
        pvi_reshuffle_window=5,

        # CP-SAT — PRIMARY optimizer (tiered with auto pad count)
        cpsat_enabled=True,
        cpsat_time_limit_seconds=120,
        cpsat_num_workers=8,
        cpsat_max_pads_override=None,  # None = auto, or set to e.g. 20

        # Simulation
        simulation_days=3650,
        simulation_start_date="2026-01-01",

        # Input files (5 CSVs)
        pad_filepath=r"\\coterra.com\data\Legacy\Tulsa\Departments\ProductionOperations\GRG\Optimization Engineering\Digital Innovation\Operations Research\MBU\v2_Rig Scheduler\v1_Schedule.csv",
        well_filepath=r"\\coterra.com\data\Legacy\Tulsa\Departments\ProductionOperations\GRG\Optimization Engineering\Digital Innovation\Operations Research\MBU\v2_Rig Scheduler\v1_Schedule_decline_curve_parameters.csv",
        base_production_filepath=r"\\coterra.com\data\Legacy\Tulsa\Departments\ProductionOperations\GRG\Optimization Engineering\Digital Innovation\Operations Research\MBU\v2_Rig Scheduler\v1_base_production.csv",
        minimum_volume_filepath=r"\\coterra.com\data\Legacy\Tulsa\Departments\ProductionOperations\GRG\Optimization Engineering\Digital Innovation\Operations Research\MBU\v2_Rig Scheduler\v1_minimum_volumes.csv",
        global_overwrite_filepath=r"\\coterra.com\data\Legacy\Tulsa\Departments\ProductionOperations\GRG\Optimization Engineering\Digital Innovation\Operations Research\MBU\v2_Rig Scheduler\v1_autoplan_overwrites.csv",

        # Output files
        schedule_output=r"C:\Users\GGannaway\Downloads\pad_schedule.csv",
        capex_output=r"C:\Users\GGannaway\Downloads\capex_timeline.csv",
        well_prod_output=r"C:\Users\GGannaway\Downloads\well_production_output.csv",
        monthly_output=r"C:\Users\GGannaway\Downloads\monthly_results.csv",
        adjustment_output=r"C:\Users\GGannaway\Downloads\adjustment_log.csv",
        plot_output_folder=r"C:\Users\GGannaway\Downloads\plots",
    )

    # =========================================================================

    # Ensure plot output folder exists
    if config.plot_output_folder:
        os.makedirs(config.plot_output_folder, exist_ok=True)
    pf = config.plot_output_folder

    base_production = load_base_production(config.base_production_filepath, config.simulation_days)
    minimum_volumes = load_minimum_volumes(config.minimum_volume_filepath, config.simulation_days)
    global_overwrites = load_global_overwrites(config.global_overwrite_filepath)

    sim, monthly, adj_log, all_iters, cpsat_diagnostics, all_strategy_results = \
        run_with_auto_adjustment(
            config=config, base_production=base_production, minimum_volumes=minimum_volumes,
            global_overwrites=global_overwrites)

    sim.print_summary()
    print_adjustment_report(adj_log, config)

    passed, shortfall = sim.check_minimum_volumes(monthly, tolerance=config.shortfall_tolerance_mcfd)
    if not passed:
        print("\n--- REMAINING SHORTFALL ---")
        print(shortfall[["month", "avg_total_mcfd", "avg_min_vol_mcfd",
                         "avg_shortfall_mcfd"]].to_string(index=False))

    # Check production ceiling
    prod_ok, prod_violations = sim.check_production_ceiling(monthly)
    if not prod_ok:
        print("\n--- PRODUCTION CEILING VIOLATIONS ---")
        print(prod_violations.to_string(index=False))
    else:
        if config.production_ceiling_mcfd > 0:
            print("  ✅ Production within ceiling (all years)")

    # --- Strategy Comparison (NEW) ---
    comp_df = print_strategy_comparison(all_strategy_results, config)
    plot_strategy_comparison(all_strategy_results, config, plot_folder=pf)

    # --- CP-SAT Diagnostics ---
    if cpsat_diagnostics and cpsat_diagnostics.get("callback"):
        print_cpsat_diagnostics_report(cpsat_diagnostics)
        plot_cpsat_diagnostics(cpsat_diagnostics, config, plot_folder=pf)
        # ADD: Multi-solution comparison with baseline overlay
        final_baselines = [k for k in all_strategy_results if k.startswith("baseline_pvi")]
        baseline_for_plot = all_strategy_results[final_baselines[-1]] if final_baselines else None
        plot_cpsat_solution_comparison(cpsat_diagnostics, sim.config,
                                    base_production, minimum_volumes,
                                    top_k=5,
                                    baseline_result=baseline_for_plot,
                                    plot_folder=pf,
                                    global_overwrites=global_overwrites)

    # --- Export CSVs ---
    def safe_export(df, path, name):
        try:
            df.to_csv(path, index=False)
        except PermissionError:
            print(f"  ⚠️  Cannot save {name}")

    safe_export(sim.get_schedule(), config.schedule_output, "schedule")
    ct = sim.get_capex_timeline()
    if not ct.empty:
        safe_export(ct, config.capex_output, "capex")
    wp = sim.get_well_production()
    if not wp.empty:
        safe_export(wp, config.well_prod_output, "well_prod")
    safe_export(monthly, config.monthly_output, "monthly")
    safe_export(pd.DataFrame(adj_log), config.adjustment_output, "adj_log")

    # Export strategy comparison
    if comp_df is not None:
        comp_path = os.path.join(pf, "strategy_comparison.csv") if pf else "strategy_comparison.csv"
        safe_export(comp_df, comp_path, "strategy_comparison")

    # Export FCF summary
    fcf = sim.get_fcf_summary()
    if fcf:
        fcf_df = pd.DataFrame([{"year": yr, **info} for yr, info in fcf.items()])
        fcf_path = os.path.join(pf, "fcf_summary.csv") if pf else "fcf_summary.csv"
        safe_export(fcf_df, fcf_path, "fcf_summary")

    # --- Original plots ---
    final_optimizer = adj_log[-1].get("optimizer", "unknown") if adj_log else "unknown"
    plot_final_results(monthly, sim, sim.config, final_optimizer, plot_folder=pf)
    plot_iteration_overlay(all_iters, sim.config, plot_folder=pf)

    return monthly, sim, adj_log, all_iters, cpsat_diagnostics, all_strategy_results


# =========================================================================
# EVERYTHING BELOW RUNS AUTOMATICALLY
# =========================================================================

if __name__ == "__main__":
    monthly, sim, adj_log, all_iters, cpsat_diagnostics, all_strategy_results = main()