from __future__ import annotations

import copy
import io
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =========================================================================
# DETERMINISTIC MATH & ACCOUNTING ENGINE (PATHWAY NUR-3 | AP CALCULUS)
# =========================================================================

EMBEDDED_FIXTURE_CSV = """month,requested,capacity_override,ad_cost
2027-01,80,60,0
2027-02,88,60,100
2027-03,72,45,0
"""


def to_money(value: Any) -> Decimal:
    """Convert numeric or string input to 2-decimal USD Decimal."""
    if isinstance(value, Decimal):
        return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def default_state() -> Dict[str, Any]:
    """Return clean initial state for Lifesavers (NUR-3, AP Calculus)."""
    return {
        "team_id": "Lifesavers",
        "course": "AP Calculus",
        "pathway_id": "NUR-3",
        "business_name": "Nurse-Led Health Education Simulation",
        "selected_level": "Advanced",
        "selected_skill": "Constrained Derivative Optimization & Accumulation",
        "snapshot_id": "SNAP-000",
        "state_version": 0,
        "simulation_date": "2027-01-01",
        "period_unit": "month",
        "working_days": 22,
        "currency": "USD",
        "random_seed": 2027,
        "startup": {
            "owner_contribution": "6000.00",
            "equipment_assets": "1500.00",
            "pre_opening_expense": "500.00",
            "opening_inventory": "0.00",
            "opening_cash": "4000.00",
            "retained_earnings": "-500.00",
        },
        "parameters": {
            "base_price": "25.00",
            "D0": 80,
            "staff_hours": 30.0,
            "hours_per_unit": 0.5,
            "venue_seats": 60,
            "no_show_rate": 0.0,
            "satisfaction_mean": 4.0,
            "variable_lines": {
                "educator_delivery": "2.00",
                "materials": "2.00",
                "cleaning_consumables": "1.00",
            },
            "fixed_lines": {
                "venue": "250.00",
                "scheduling_software": "100.00",
                "accessibility_support": "200.00",
                "admin_insurance_assumption": "150.00",
            },
        },
        "cash": "4000.00",
        "debt": "0.00",
        "equipment_net": "1500.00",
        "retained_earnings": "-500.00",
        "ledger": [],
        "processed_run_ids": [],
        "assumptions": [
            "Fictional general wellness and health education workshops only; no clinical care or diagnosis.",
            "Immediate customer payment on attended seats; no payment collected for no-shows.",
            "Borrowing, interest, refunds, depreciation, and sales tax disabled at baseline unless explicitly entered.",
        ],
    }


def compute_unit_variable_cost(params: Dict[str, Any]) -> Decimal:
    return sum((to_money(v) for v in params["variable_lines"].values()), Decimal("0.00"))


def compute_monthly_fixed_cost(params: Dict[str, Any]) -> Decimal:
    return sum((to_money(v) for v in params["fixed_lines"].values()), Decimal("0.00"))


def compute_normal_capacity(params: Dict[str, Any]) -> Tuple[int, str]:
    staff_hours = float(params["staff_hours"])
    hours_per_unit = float(params["hours_per_unit"])
    venue_seats = int(params["venue_seats"])
    if hours_per_unit <= 0:
        raise ValueError("hours_per_unit must be > 0")
    if staff_hours < 0 or venue_seats < 0:
        raise ValueError("staff_hours and venue_seats must be >= 0")

    staff_cap = math.floor(staff_hours / hours_per_unit)
    cap = min(staff_cap, venue_seats)
    if staff_cap < venue_seats:
        bottleneck = f"Educator hours ({staff_cap} seats max)"
    elif venue_seats < staff_cap:
        bottleneck = f"Venue seats ({venue_seats} seats max)"
    else:
        bottleneck = f"Balanced ({cap} seats max)"
    return cap, bottleneck


def simulate_month(
    state: Dict[str, Any],
    *,
    sim_date: str,
    requested: int,
    run_id: str,
    unit_price: Optional[Decimal] = None,
    ad_cost: Decimal = Decimal("0.00"),
    capacity_override: Optional[int] = None,
    refund_amount: Decimal = Decimal("0.00"),
    loan_cash: Decimal = Decimal("0.00"),
    principal_repayment: Decimal = Decimal("0.00"),
    interest_expense: Decimal = Decimal("0.00"),
    additional_capex: Decimal = Decimal("0.00"),
    new_owner_cash: Decimal = Decimal("0.00"),
) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """Execute one deterministic monthly simulation step."""
    if not run_id or not str(run_id).strip():
        raise ValueError("run_id is required to protect against duplicate runs.")
    if requested < 0:
        raise ValueError("requested demand cannot be negative.")
    if (
        ad_cost < 0
        or refund_amount < 0
        or loan_cash < 0
        or principal_repayment < 0
        or interest_expense < 0
        or additional_capex < 0
        or new_owner_cash < 0
    ):
        raise ValueError("Monetary inputs cannot be negative.")

    if run_id in state.get("processed_run_ids", []):
        existing = next(
            (row for row in state["ledger"] if row.get("run_id") == run_id),
            {"note": "Duplicate run_id ignored; no new transaction created."},
        )
        return state, existing, True

    next_state = copy.deepcopy(state)
    params = next_state["parameters"]
    no_show_rate = float(params["no_show_rate"])
    if not (0.0 <= no_show_rate <= 1.0):
        raise ValueError("no_show_rate must be between 0 and 1.")

    price = to_money(unit_price if unit_price is not None else params["base_price"])
    if price <= Decimal("0.00"):
        raise ValueError("price must be > 0.")

    if capacity_override is not None:
        if capacity_override < 0:
            raise ValueError("capacity_override cannot be negative.")
        capacity = int(capacity_override)
        bottleneck = f"Fixture override ({capacity} seats)"
    else:
        capacity, bottleneck = compute_normal_capacity(params)

    booked = min(int(requested), capacity)
    unserved_demand = max(0, int(requested) - booked)
    attended = math.floor(booked * (1.0 - no_show_rate))
    no_shows = booked - attended
    sold = attended

    var_per_unit = compute_unit_variable_cost(params)
    fixed_expense = compute_monthly_fixed_cost(params)
    ad_cost = to_money(ad_cost)
    refund_amount = to_money(refund_amount)
    loan_cash = to_money(loan_cash)
    principal_repayment = to_money(principal_repayment)
    interest_expense = to_money(interest_expense)
    additional_capex = to_money(additional_capex)
    new_owner_cash = to_money(new_owner_cash)

    gross_revenue = to_money(price * Decimal(sold))
    net_revenue = to_money(gross_revenue - refund_amount)
    variable_expense = to_money(var_per_unit * Decimal(sold))
    total_operating_expense = to_money(variable_expense + fixed_expense + ad_cost)
    operating_profit = to_money(net_revenue - total_operating_expense)
    net_profit = to_money(operating_profit - interest_expense)

    opening_cash = to_money(next_state["cash"])
    ending_cash = to_money(
        opening_cash
        + net_revenue
        - variable_expense
        - fixed_expense
        - ad_cost
        - interest_expense
        - principal_repayment
        - additional_capex
        + new_owner_cash
        + loan_cash
    )

    new_version = int(next_state["state_version"]) + 1
    tx_id = f"TX-{new_version:04d}-{run_id}"
    snapshot_id = f"SNAP-{new_version:03d}"

    record = {
        "transaction_id": tx_id,
        "run_id": run_id,
        "date": sim_date,
        "event_type": "monthly_close",
        "offering_id": "NUR-WORKSHOP-01",
        "requested": int(requested),
        "capacity": capacity,
        "bottleneck": bottleneck,
        "booked": booked,
        "no_shows": no_shows,
        "sold": sold,
        "unserved_demand": unserved_demand,
        "unit_price": str(price),
        "revenue": str(net_revenue),
        "variable_expense": str(variable_expense),
        "fixed_expense": str(fixed_expense),
        "marketing_expense": str(ad_cost),
        "total_expense": str(total_operating_expense),
        "operating_profit": str(operating_profit),
        "interest_expense": str(interest_expense),
        "net_profit": str(net_profit),
        "opening_cash": str(opening_cash),
        "closing_cash": str(ending_cash),
        "state_version": new_version,
        "fixture_mode": capacity_override is not None,
    }

    next_state["cash"] = str(ending_cash)
    next_state["debt"] = str(to_money(to_money(next_state["debt"]) + loan_cash - principal_repayment))
    next_state["equipment_net"] = str(to_money(to_money(next_state["equipment_net"]) + additional_capex))
    next_state["retained_earnings"] = str(to_money(to_money(next_state["retained_earnings"]) + net_profit))
    next_state["state_version"] = new_version
    next_state["snapshot_id"] = snapshot_id
    next_state["simulation_date"] = sim_date
    next_state["ledger"].append(record)
    next_state["processed_run_ids"].append(run_id)

    return next_state, record, False


def replay_fixture(base_state: Dict[str, Any], df: pd.DataFrame) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Reset to clean opening state and replay monthly fixture rows."""
    state = default_state()
    state["team_id"] = base_state.get("team_id", "Lifesavers")
    state["selected_level"] = base_state.get("selected_level", "Advanced")
    results: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        month_str = str(row["month"]).strip()
        sim_date = f"{month_str}-28" if len(month_str) == 7 else month_str
        requested = int(row["requested"])
        cap_override = int(row["capacity_override"]) if pd.notna(row["capacity_override"]) else None
        ad_cost = to_money(row["ad_cost"]) if pd.notna(row["ad_cost"]) else Decimal("0.00")
        run_id = f"FIXTURE-{month_str}-{idx+1}"

        state, rec, _ = simulate_month(
            state,
            sim_date=sim_date,
            requested=requested,
            run_id=run_id,
            ad_cost=ad_cost,
            capacity_override=cap_override,
        )
        results.append(rec)

    return state, results


def export_snapshot_json(state: Dict[str, Any]) -> str:
    return json.dumps(state, indent=2)


def validate_and_import_snapshot(raw_json: str, current_state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate uploaded snapshot JSON; reject wrong team_id, negative values, or stale state_version."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON format: {exc}") from exc

    required_keys = [
        "team_id",
        "course",
        "pathway_id",
        "snapshot_id",
        "state_version",
        "simulation_date",
        "parameters",
        "cash",
        "ledger",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Missing required snapshot fields: {', '.join(missing)}")

    if data["team_id"] != current_state["team_id"]:
        raise ValueError(
            f"Team ID mismatch: imported '{data['team_id']}', expected '{current_state['team_id']}'."
        )

    imported_version = int(data["state_version"])
    current_version = int(current_state["state_version"])
    if imported_version < current_version:
        raise ValueError(
            f"Stale state_version rejected: imported v{imported_version}, current dashboard is at v{current_version}."
        )

    if to_money(data["cash"]) < Decimal("0.00"):
        raise ValueError("Invalid snapshot: cash cannot be negative.")

    params = data["parameters"]
    if float(params["base_price"]) <= 0 or float(params["hours_per_unit"]) <= 0:
        raise ValueError("Invalid snapshot: price and hours_per_unit must be > 0.")
    if int(params["D0"]) < 0 or float(params["staff_hours"]) < 0 or int(params["venue_seats"]) < 0:
        raise ValueError("Invalid snapshot: demand, staff_hours, and venue_seats must be >= 0.")

    seen_tx = set()
    for row in data["ledger"]:
        tx_id = row.get("transaction_id")
        if not tx_id or tx_id in seen_tx:
            raise ValueError(f"Duplicate or missing transaction_id in imported ledger: {tx_id}")
        seen_tx.add(tx_id)

    return data


def ap_calculus_analysis(
    D0: float,
    p0: float,
    v: float,
    F: float,
    capacity: float,
) -> Dict[str, Any]:
    """Exact AP Calculus adapter equations for constrained profit optimization."""
    a = 2.0 * float(D0)
    b = float(D0) / float(p0)
    p_max = 2.0 * float(p0)
    v = float(v)
    F = float(F)
    cap = float(capacity)

    p_unconstrained = (a + b * v) / (2.0 * b)
    q_at_unconstrained = max(0.0, a - b * p_unconstrained)
    unconstrained_profit = (p_unconstrained - v) * q_at_unconstrained - F

    p_corner = (a - cap) / b if b > 0 else 0.0
    has_corner = 0.0 < p_corner < p_max

    candidates = [0.0, p_max]
    if has_corner:
        candidates.append(p_corner)
    if (not has_corner and 0.0 <= p_unconstrained <= p_max) or (has_corner and p_corner <= p_unconstrained <= p_max):
        candidates.append(p_unconstrained)

    def eval_continuous(p_val: float) -> Tuple[float, float, float]:
        q_raw = max(0.0, a - b * p_val)
        s_cont = min(cap, q_raw)
        prof_cont = (p_val - v) * s_cont - F
        return q_raw, s_cont, prof_cont

    def eval_integer(p_val: float) -> Tuple[int, float]:
        q_raw = max(0.0, a - b * p_val)
        s_int = int(min(cap, math.floor(q_raw)))
        prof_int = (p_val - v) * s_int - F
        return s_int, prof_int

    best_p = max(candidates, key=lambda x: eval_continuous(x)[2])
    _, best_s_cont, best_prof_cont = eval_continuous(best_p)
    best_s_int, best_prof_int = eval_integer(best_p)

    grid = np.linspace(0.0, p_max, 241)
    if has_corner:
        grid = np.unique(np.sort(np.append(grid, [p_corner - 1e-4, p_corner, p_corner + 1e-4])))

    rows = []
    for p_val in grid:
        q_raw, s_cont, prof_cont = eval_continuous(float(p_val))
        s_int, prof_int = eval_integer(float(p_val))
        if has_corner and abs(p_val - p_corner) < 1e-5:
            deriv = np.nan
            branch = "Capacity Corner (Undefined)"
        elif has_corner and p_val < p_corner:
            deriv = cap
            branch = "Capacity-Capped Branch"
        else:
            deriv = a + b * v - 2.0 * b * p_val
            branch = "Demand-Limited Branch"

        rows.append(
            {
                "price": float(p_val),
                "demand_q": q_raw,
                "sales_continuous": s_cont,
                "sales_integer": s_int,
                "profit_continuous": prof_cont,
                "profit_integer": prof_int,
                "derivative": deriv,
                "branch": branch,
            }
        )

    return {
        "a": a,
        "b": b,
        "p_max": p_max,
        "p_unconstrained": p_unconstrained,
        "q_at_unconstrained": q_at_unconstrained,
        "unconstrained_profit": unconstrained_profit,
        "p_corner": p_corner,
        "has_corner": has_corner,
        "optimal_price": best_p,
        "optimal_sales_continuous": best_s_cont,
        "optimal_profit_continuous": best_prof_cont,
        "optimal_sales_integer": best_s_int,
        "optimal_profit_integer": best_prof_int,
        "curve_df": pd.DataFrame(rows),
    }


def ap_calculus_integral_campaign(
    r0: float = 100.0,
    r_slope: float = 20.0,
    c0: float = 60.0,
    c_slope: float = 5.0,
    t_start: float = 0.0,
    t_end: float = 10.0,
) -> Dict[str, Any]:
    """Isolated 10-day campaign rate integration."""
    dt = float(t_end) - float(t_start)
    if dt < 0:
        raise ValueError("t_end must be >= t_start")

    def rev_antideriv(t: float) -> float:
        return r0 * t + 0.5 * r_slope * (t ** 2)

    def cost_antideriv(t: float) -> float:
        return c0 * t + 0.5 * c_slope * (t ** 2)

    acc_rev = rev_antideriv(t_end) - rev_antideriv(t_start)
    acc_cost = cost_antideriv(t_end) - cost_antideriv(t_start)
    acc_profit = acc_rev - acc_cost

    t_grid = np.linspace(t_start, t_end, 101)
    df = pd.DataFrame(
        {
            "day_t": t_grid,
            "revenue_rate": r0 + r_slope * t_grid,
            "cost_rate": c0 + c_slope * t_grid,
            "net_rate": (r0 - c0) + (r_slope - c_slope) * t_grid,
        }
    )

    return {
        "accumulated_revenue": acc_rev,
        "accumulated_cost": acc_cost,
        "accumulated_profit": acc_profit,
        "rate_df": df,
    }


# =========================================================================
# STREAMLIT USER INTERFACE
# =========================================================================

st.set_page_config(
    page_title="Lifesavers | Career Venture Lab (NUR-3)",
    page_icon="🩺",
    layout="wide",
)

if "biz_state" not in st.session_state:
    st.session_state["biz_state"] = default_state()
if "last_message" not in st.session_state:
    st.session_state["last_message"] = None

state = st.session_state["biz_state"]
params = state["parameters"]

st.title("🩺 Lifesavers — Nurse-Led Health Education Simulation")
st.caption(
    f"**Team:** {state['team_id']} | **Pathway:** {state['pathway_id']} | "
    f"**Course:** {state['course']} | **Approved Level:** {state['selected_level']} | "
    f"**Snapshot:** `{state['snapshot_id']}` (v{state['state_version']})"
)
st.info(
    "**Educational Simulation Notice:** Fictional general wellness, caregiver education, and first-aid awareness workshops only. "
    "No medical diagnosis, treatment, or patient data. **Build Status:** Stage 1 Active (Core Engine, AP Calculus Math, Ledger, Save/Load JSON)."
)

if st.session_state["last_message"]:
    kind, msg = st.session_state["last_message"]
    getattr(st, kind)(msg)
    st.session_state["last_message"] = None

cap_now, bottleneck_now = compute_normal_capacity(params)
var_unit_now = compute_unit_variable_cost(params)
fixed_mo_now = compute_monthly_fixed_cost(params)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Spendable Cash", f"${to_money(state['cash']):,.2f}")
k2.metric("Monthly Capacity", f"{cap_now} seats", bottleneck_now)
k3.metric("Base Price / Seat", f"${to_money(params['base_price']):,.2f}")
k4.metric("Variable / Fixed Cost", f"${var_unit_now}/seat", f"${fixed_mo_now}/mo fixed")
k5.metric("State Version", f"v{state['state_version']}", state["simulation_date"])

tabs = st.tabs(
    [
        "1. Overview & Inputs",
        "2. Math Skills & Challenge Level (AP Calculus)",
        "3. Run Simulation & Fixture Tests",
        "4. Ledger & Save/Load Snapshot",
        "5. Stage 2-4 Roadmap (Placeholders)",
    ]
)

with tabs[0]:
    st.subheader("Business Parameters & Cost Structure")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Pricing, Demand & Resource Capacity")
        with st.form("params_form"):
            new_price = st.number_input(
                "Workshop Seat Price (USD)", min_value=1.0, value=float(params["base_price"]), step=1.0
            )
            new_d0 = st.number_input(
                "Baseline Monthly Requested Seats (D0)", min_value=0, value=int(params["D0"]), step=5
            )
            new_staff_hours = st.number_input(
                "Educator Productive Hours / Month", min_value=0.0, value=float(params["staff_hours"]), step=1.0
            )
            new_hours_unit = st.number_input(
                "Educator Hours per Seat", min_value=0.1, value=float(params["hours_per_unit"]), step=0.1
            )
            new_venue_seats = st.number_input(
                "Venue Seat Capacity / Month", min_value=0, value=int(params["venue_seats"]), step=5
            )
            new_no_show = st.slider(
                "No-Show Rate (0.00 to 1.00)", min_value=0.0, max_value=1.0, value=float(params["no_show_rate"]), step=0.05
            )
            new_sat = st.slider(
                "Workshop Satisfaction Mean (1 to 5)", min_value=1.0, max_value=5.0, value=float(params["satisfaction_mean"]), step=0.1
            )
            save_params = st.form_submit_button("Update Operating Assumptions")

        if save_params:
            params["base_price"] = str(to_money(new_price))
            params["D0"] = int(new_d0)
            params["staff_hours"] = float(new_staff_hours)
            params["hours_per_unit"] = float(new_hours_unit)
            params["venue_seats"] = int(new_venue_seats)
            params["no_show_rate"] = float(new_no_show)
            params["satisfaction_mean"] = float(new_sat)
            st.session_state["last_message"] = ("success", "Operating parameters updated (ledger unchanged until a simulation step runs).")
            st.rerun()

    with col_b:
        st.markdown("#### Editable Cost Lines & Startup Opening Balances")
        with st.form("costs_form"):
            st.markdown("**Variable Cost per Attended Seat (USD)**")
            v_ed = st.number_input("Educator Delivery ($/seat)", min_value=0.0, value=float(params["variable_lines"]["educator_delivery"]), step=0.5)
            v_mat = st.number_input("Workshop Materials ($/seat)", min_value=0.0, value=float(params["variable_lines"]["materials"]), step=0.5)
            v_cln = st.number_input("Cleaning Consumables ($/seat)", min_value=0.0, value=float(params["variable_lines"]["cleaning_consumables"]), step=0.5)

            st.markdown("**Monthly Fixed Expenses (USD/month)**")
            f_ven = st.number_input("Venue Rental ($/mo)", min_value=0.0, value=float(params["fixed_lines"]["venue"]), step=10.0)
            f_sw = st.number_input("Scheduling Software ($/mo)", min_value=0.0, value=float(params["fixed_lines"]["scheduling_software"]), step=10.0)
            f_acc = st.number_input("Accessibility Support ($/mo — explicit line item)", min_value=0.0, value=float(params["fixed_lines"]["accessibility_support"]), step=10.0)
            f_adm = st.number_input("Admin & Insurance Assumption ($/mo)", min_value=0.0, value=float(params["fixed_lines"]["admin_insurance_assumption"]), step=10.0)
            save_costs = st.form_submit_button("Update Cost Lines")

        if save_costs:
            params["variable_lines"]["educator_delivery"] = str(to_money(v_ed))
            params["variable_lines"]["materials"] = str(to_money(v_mat))
            params["variable_lines"]["cleaning_consumables"] = str(to_money(v_cln))
            params["fixed_lines"]["venue"] = str(to_money(f_ven))
            params["fixed_lines"]["scheduling_software"] = str(to_money(f_sw))
            params["fixed_lines"]["accessibility_support"] = str(to_money(f_acc))
            params["fixed_lines"]["admin_insurance_assumption"] = str(to_money(f_adm))
            st.session_state["last_message"] = ("success", "Cost line items updated.")
            st.rerun()

        st.markdown("**Startup Balance Summary (Post-Startup Day 1)**")
        startup_df = pd.DataFrame(
            [
                {"Item": "Owner Contribution (Equity)", "Amount ($)": state["startup"]["owner_contribution"], "Accounting Treatment": "Equity financing (not sales revenue)"},
                {"Item": "Demonstration & Accessibility Equipment", "Amount ($)": state["startup"]["equipment_assets"], "Accounting Treatment": "Capital asset (not expensed again monthly)"},
                {"Item": "Pre-Opening Setup Expense", "Amount ($)": state["startup"]["pre_opening_expense"], "Accounting Treatment": "Initial retained earnings (-$500)"},
                {"Item": "Post-Startup Opening Cash", "Amount ($)": state["startup"]["opening_cash"], "Accounting Treatment": "Starting spendable cash reserve"},
            ]
        )
        st.dataframe(startup_df, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Math Skills & Challenge Level Panel — AP Calculus (`NUR-3`)")

    lvl_col1, lvl_col2 = st.columns([1, 2])
    with lvl_col1:
        chosen_level = st.selectbox(
            "Select Challenge Level (Teacher Approved: Advanced)",
            ["Foundation", "Applied", "Advanced"],
            index=["Foundation", "Applied", "Advanced"].index(state["selected_level"]),
        )
        if chosen_level != state["selected_level"]:
            state["selected_level"] = chosen_level
            st.rerun()

    p0_val = float(params["base_price"])
    d0_val = float(params["D0"])
    v_val = float(compute_unit_variable_cost(params))
    f_val = float(compute_monthly_fixed_cost(params))
    cap_val = float(cap_now)

    calc = ap_calculus_analysis(D0=d0_val, p0=p0_val, v=v_val, F=f_val, capacity=cap_val)

    with lvl_col2:
        st.markdown(
            f"**Course:** AP Calculus | **Level:** `{chosen_level}` | "
            f"**Domain:** $0 \le p \le {calc['p_max']:.2f}$ USD/seat | "
            f"**Capacity Limit:** {int(cap_val)} seats/month"
        )
        if chosen_level == "Foundation":
            st.markdown(
                f"**Business Question (Foundation):** Ignoring capacity limits, what ticket price maximizes monthly profit?  \n"
                f"**Equations:** $q(p) = {calc['a']:.1f} - {calc['b']:.2f}p$, $\; P(p) = (p - {v_val:.2f})({calc['a']:.1f} - {calc['b']:.2f}p) - {f_val:.2f}$, "
                f"$\; P'(p) = {calc['a'] + calc['b']*v_val:.2f} - {2*calc['b']:.2f}p$ (units: dollars per dollar of price).  \n"
                f"**Unconstrained Critical Point:** $p^* = \${calc['p_unconstrained']:.2f}$ where $q(p^*) = {calc['q_at_unconstrained']:.1f}$ seats and $P(p^*) = \${calc['unconstrained_profit']:,.2f}$."
            )
        elif chosen_level == "Applied":
            st.markdown(
                f"**Business Question (Applied):** How does our **{int(cap_val)}-seat capacity ceiling** change the optimal price?  \n"
                f"**Piecewise Model:** $s(p) = \min({int(cap_val)}, \max(0, {calc['a']:.1f} - {calc['b']:.2f}p))$.  \n"
                f"**Capacity Corner:** $p_{{\text{{corner}}}} = \frac{{{calc['a']:.1f} - {int(cap_val)}}}{{{calc['b']:.2f}}} = \${calc['p_corner']:.2f}$ (derivative undefined at corner!).  \n"
                f"**Constrained Optimum:** $p^* = \${calc['optimal_price']:.2f}$ -> Continuous Profit $= \${calc['optimal_profit_continuous']:,.2f}$ | Rounded Whole-Unit Profit ({calc['optimal_sales_integer']} seats) $= \${calc['optimal_profit_integer']:,.2f}$."
            )
        else:
            st.markdown(
                f"**Business Question (Advanced):** Combine constrained derivative optimization ($p^* = \${calc['optimal_price']:.2f}$, corner at $\${calc['p_corner']:.2f}$) "
                f"with **definite integration** of daily campaign revenue and cost rates over a 10-day launch window."
            )

    test_p = st.slider(
        "Transfer Check — Predict profit & derivative BEFORE moving the price slider (USD/seat):",
        min_value=0.0,
        max_value=float(calc["p_max"]),
        value=min(p0_val, float(calc["p_max"])),
        step=0.5,
    )
    q_at_p = max(0.0, calc["a"] - calc["b"] * test_p)
    s_cont_p = min(cap_val, q_at_p)
    s_int_p = int(min(cap_val, int(q_at_p)))
    prof_cont_p = (test_p - v_val) * s_cont_p - f_val
    prof_int_p = (test_p - v_val) * s_int_p - f_val

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Demand q(p)", f"{q_at_p:.2f} seats")
    m2.metric(f"Realized Seats (Cap={int(cap_val)})", f"{s_int_p} whole ({s_cont_p:.2f} cont.)")
    m3.metric("Continuous Profit P(p)", f"${prof_cont_p:,.2f}")
    m4.metric("Integer Ledger Profit", f"${prof_int_p:,.2f}")

    df_curve = calc["curve_df"]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Monthly Profit P(p) vs. Price p (Continuous vs. Whole-Unit Integer)",
            "Derivative P'(p) ($/month per $1 price change) — Undefined at Capacity Corner",
        ),
    )
    fig.add_trace(
        go.Scatter(x=df_curve["price"], y=df_curve["profit_continuous"], name="Continuous Profit P(p)", line=dict(width=3, color="#1f77b4")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df_curve["price"], y=df_curve["profit_integer"], name="Integer-Unit Profit", line=dict(width=2, dash="dot", color="#ff7f0e")),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-Even ($0)", row=1, col=1)

    if calc["has_corner"] and chosen_level in ("Applied", "Advanced"):
        fig.add_vline(
            x=calc["p_corner"],
            line_dash="dash",
            line_color="#d62728",
            annotation_text=f"Capacity Corner p=${calc['p_corner']:.2f}",
        )

    fig.add_trace(
        go.Scatter(x=df_curve["price"], y=df_curve["derivative"], name="Derivative P'(p)", line=dict(width=3, color="#2ca02c")),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    if calc["has_corner"] and chosen_level in ("Applied", "Advanced"):
        fig.add_trace(
            go.Scatter(
                x=[calc["p_corner"]],
                y=[0],
                mode="markers",
                name="Corner (P'(p) Undefined)",
                marker=dict(size=11, color="white", line=dict(width=3, color="#d62728")),
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Seat Price p (USD)", row=2, col=1)
    fig.update_yaxes(title_text="Monthly Profit (USD)", row=1, col=1)
    fig.update_yaxes(title_text="P'(p) (USD / $1)", row=2, col=1)
    fig.update_layout(height=580, legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    sample_prices = sorted(set([10.0, 15.0, 20.0, round(calc["p_corner"], 2), 25.0, round(calc["p_unconstrained"], 2), 35.0, 40.0, 50.0]))
    table_rows = []
    for sp in sample_prices:
        if 0 <= sp <= calc["p_max"]:
            qr = max(0.0, calc["a"] - calc["b"] * sp)
            sc = min(cap_val, qr)
            si = int(min(cap_val, int(qr)))
            table_rows.append(
                {
                    "Price ($)": f"{sp:.2f}",
                    "Demand q(p)": f"{qr:.2f}",
                    "Capped Sales (Cont.)": f"{sc:.2f}",
                    "Whole-Unit Sales": si,
                    "Continuous Profit ($)": f"{(sp - v_val)*sc - f_val:,.2f}",
                    "Integer Profit ($)": f"{(sp - v_val)*si - f_val:,.2f}",
                }
            )
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    if chosen_level == "Advanced":
        st.divider()
        st.markdown("### Advanced Integration Module: Isolated 10-Day Campaign Accumulation")
        st.caption(
            "Models daily revenue rate $r(t) = 100 + 20t$ ($/day) and cost rate $c(t) = 60 + 5t$ ($/day) over $0 \le t \le 10$ days. "
            "Kept separate from the monthly accounting ledger."
        )
        ic1, ic2 = st.columns([1, 2])
        with ic1:
            t_end = st.slider("Campaign Window End Day (t_end)", min_value=1, max_value=14, value=10)
            integ = ap_calculus_integral_campaign(t_end=float(t_end))
            st.metric("Accumulated Revenue ∫r(t)dt", f"${integ['accumulated_revenue']:,.2f}")
            st.metric("Accumulated Cost ∫c(t)dt", f"${integ['accumulated_cost']:,.2f}")
            st.metric("Accumulated Net Profit ∫(r-c)dt", f"${integ['accumulated_profit']:,.2f}")
        with ic2:
            rdf = integ["rate_df"]
            fig_int = go.Figure()
            fig_int.add_trace(go.Scatter(x=rdf["day_t"], y=rdf["revenue_rate"], name="Revenue Rate r(t) ($/day)", fill="tozeroy"))
            fig_int.add_trace(go.Scatter(x=rdf["day_t"], y=rdf["cost_rate"], name="Cost Rate c(t) ($/day)", fill="tozeroy"))
            fig_int.update_layout(
                title=f"Daily Revenue & Cost Rates Integrated over [0, {t_end}] Days",
                xaxis_title="Day t",
                yaxis_title="Rate ($ / day)",
                height=340,
            )
            st.plotly_chart(fig_int, use_container_width=True)

    st.info(
        "**Interpretation Prompt:** Why is the unconstrained critical point $p^* = \$27.50$ (where $q = 72$ seats) infeasible for Lifesavers right now, "
        "and why does $P'(p)$ jump at the capacity corner $p = \$31.25$?"
    )

with tabs[2]:
    st.subheader("Advance Simulation (Normal Mode) or Run Stage 1 Fixture Tests")
    col_sim, col_fix = st.columns(2)

    with col_sim:
        st.markdown("#### Normal Mode: Advance One Month")
        with st.form("advance_month_form"):
            sim_date = st.text_input("Simulation Date (YYYY-MM-DD)", value="2027-01-31")
            run_id = st.text_input("Unique Run ID", value=f"RUN-{state['state_version'] + 1:03d}")
            req_units = st.number_input("Requested Workshop Seats", min_value=0, value=int(params["D0"]), step=1)
            ad_spend = st.number_input("Approved Own-Business Marketing Expense ($)", min_value=0.0, value=0.0, step=25.0)
            submit_step = st.form_submit_button("Approve & Advance Simulation Once")

        if submit_step:
            try:
                updated, rec, is_dup = simulate_month(
                    state,
                    sim_date=sim_date,
                    requested=int(req_units),
                    run_id=run_id.strip(),
                    ad_cost=to_money(ad_spend),
                )
                st.session_state["biz_state"] = updated
                if is_dup:
                    st.session_state["last_message"] = ("warning", f"Duplicate run_id `{run_id}` blocked—no duplicate transaction recorded.")
                else:
                    st.session_state["last_message"] = (
                        "success",
                        f"Recorded {rec['transaction_id']}: Sold {rec['sold']} seats, Profit ${rec['operating_profit']}, Closing Cash ${rec['closing_cash']}.",
                    )
                st.rerun()
            except ValueError as err:
                st.error(str(err))

    with col_fix:
        st.markdown("#### Fixture-Test Mode (`NUR-3` 3-Month Sample Fixture)")
        st.warning(
            "**Fixture-Test Mode Warning:** Replaying the 3-month sample fixture resets your business to the Day-1 opening state (`$4,000` cash) "
            "and overrides capacity using the fixture's `capacity_override` column to verify the accounting ledger."
        )
        fdf = pd.read_csv(io.StringIO(EMBEDDED_FIXTURE_CSV))
        st.dataframe(fdf, use_container_width=True, hide_index=True)
        if st.button("Reset & Replay 3-Month Acceptance Fixture"):
            new_state, _ = replay_fixture(state, fdf)
            st.session_state["biz_state"] = new_state
            st.session_state["last_message"] = (
                "success",
                "Replayed 2027-01, 2027-02, and 2027-03 fixture rows. Check the Ledger tab to verify closing cash: $4,500 -> $4,900 -> $5,100!",
            )
            st.rerun()

with tabs[3]:
    st.subheader("Dated Transaction Ledger & Durable JSON Snapshot")

    if state["ledger"]:
        ledger_df = pd.DataFrame(state["ledger"])
        st.dataframe(ledger_df, use_container_width=True, hide_index=True)
        csv_bytes = ledger_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Ledger CSV", data=csv_bytes, file_name="lifesavers_ledger.csv", mime="text/csv")
    else:
        st.info("No transactions recorded yet. Advance a month or replay the sample fixture in Tab 3.")

    st.divider()
    c_exp, c_imp = st.columns(2)
    with c_exp:
        st.markdown("#### Export Frozen Snapshot (JSON)")
        snap_json = export_snapshot_json(state)
        st.download_button(
            label=f"Download {state['snapshot_id']}_v{state['state_version']}.json",
            data=snap_json.encode("utf-8"),
            file_name=f"lifesavers_{state['snapshot_id']}_v{state['state_version']}.json",
            mime="application/json",
        )
        if st.button("Reset Business to Day-1 Opening State ($4,000 Cash)"):
            st.session_state["biz_state"] = default_state()
            st.session_state["last_message"] = ("info", "Reset to Day-1 opening state (v0, $4,000.00 cash).")
            st.rerun()

    with c_imp:
        st.markdown("#### Restore or Validate Snapshot JSON")
        uploaded = st.file_uploader("Upload a saved Lifesavers snapshot (.json)", type=["json"])
        if uploaded is not None:
            if st.button("Validate & Load Uploaded Snapshot"):
                try:
                    raw_text = uploaded.read().decode("utf-8")
                    imported_state = validate_and_import_snapshot(raw_text, state)
                    st.session_state["biz_state"] = imported_state
                    st.session_state["last_message"] = (
                        "success",
                        f"Loaded {imported_state['snapshot_id']} (v{imported_state['state_version']}) cleanly!",
                    )
                    st.rerun()
                except ValueError as err:
                    st.error(f"Import Rejected: {err}")

with tabs[4]:
    st.subheader("Upcoming Modules (Unimplemented in Stage 1)")
    st.markdown(
        """
        Per the Career Venture Lab staged build plan, these modules are intentionally marked **NOT YET IMPLEMENTED** until Stage 1 tests pass:
        - **Stage 2:** Full Report Center (Income Statement, Cash Flow, Budget Variance, Simplified Balance Sheet + PDF export), AI Departmental Recommendation Import/Approval Queue, and Costed Marketing Studio.
        - **Stage 3:** Frozen-Snapshot Practice Assessment Panel (with private server-side teacher key separation) and Cross-Course Connections Evidence Panel.
        - **Stage 4 (Optional):** Fictional Workshop Registration Storefront (`DEMONSTRATION BUSINESS — NO REAL PURCHASES`).
        """
    )
