"""Investigation layer: WHY did a metric look the way it does, and WHAT to do next.

Design principle: Python does ALL the math. Nothing here calls an LLM.
Every finding carries:
  observed        - measured facts
  tables          - the evidence behind them
  possible_causes - hypotheses, each tagged Strong / Moderate / Weak with its basis
  next_steps      - concrete actions
  confidence      - how much to trust it, based on sample size
  data_gaps       - data that would sharpen the diagnosis

Segments that concentrate returns show ASSOCIATION, not proof of cause.
"""
import numpy as np
import pandas as pd

SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}
DIM_LABELS = {
    "product_name": "Product", "category": "Category", "city": "City",
    "courier": "Courier", "delivery_bucket": "Delivery time",
}
BLANKS = {"", "nan", "none", "unknown", "null"}


def _rs(x):
    return f"Rs. {x:,.0f}"


def _confidence(n_orders):
    if n_orders < 30:
        return "Low", f"Only {n_orders} orders analysed - treat patterns as hints, not conclusions."
    if n_orders < 100:
        return "Medium", f"{n_orders} orders analysed - patterns are reasonably informative."
    return "High", f"{n_orders} orders analysed - enough volume to trust clear patterns."


def _strength(n, lift, excess):
    if n >= 10 and lift >= 1.5 and excess >= 3:
        return "Strong"
    if n >= 5 and lift >= 1.3 and excess >= 2:
        return "Moderate"
    return "Weak"


def _reason_action(reason):
    r = str(reason).lower()
    if "size" in r or "fit" in r:
        return "Add a size chart and measurement guide, verify listed dimensions against the actual product, and add fit notes to the description."
    if "transit" in r or "damag" in r or "broken" in r:
        return "Inspect packaging and courier handling; add a pre-dispatch quality check and sturdier packaging."
    if "quality" in r or "defect" in r:
        return "Audit the latest batch with the supplier and add a quality check before dispatch."
    if "described" in r or "wrong" in r or "different" in r or "colo" in r:
        return "Make photos and descriptions match the real product (colour, fabric, size) and add real-customer photos."
    if "late" in r or "delay" in r or "slow" in r:
        return "Review dispatch time and courier delivery SLAs; set realistic delivery promises at checkout."
    if "mind" in r or "no longer" in r:
        return "Check whether discounts drive impulse purchases; clarify the return policy and product details before checkout."
    return "Review these returned orders manually and tag the real root cause."


def _usable(d, col):
    """Column exists and has at least two distinct real (non-blank) values."""
    if col not in d.columns:
        return None
    s = d[col].astype(str).str.strip()
    s = s[~s.str.lower().isin(BLANKS)]
    return col if s.nunique() > 1 else None


def _segments(d, dim, overall_rate, min_n):
    """Per-segment order/return counts; only segments above baseline with enough support."""
    s = d[dim].astype(str).str.strip()
    ok = ~s.str.lower().isin(BLANKS)
    x = d[ok].assign(_seg=s[ok])
    orders = x.groupby("_seg")["order_id"].nunique()
    rets = x[x["is_returned"]].groupby("_seg")["order_id"].nunique().reindex(orders.index, fill_value=0)
    cost = x[x["is_returned"]].groupby("_seg")["return_cost"].sum().reindex(orders.index, fill_value=0)
    out = pd.DataFrame({"orders": orders, "returns": rets, "return_cost": cost})
    out = out[out["orders"] >= min_n].copy()
    out["rate"] = out["returns"] / out["orders"] * 100
    out["lift"] = out["rate"] / (overall_rate * 100) if overall_rate > 0 else 0
    out["excess"] = out["returns"] - out["orders"] * overall_rate
    return out[(out["lift"] >= 1.3) & (out["excess"] >= 1.5)].sort_values("excess", ascending=False)


def _investigate_returns(d, n_orders):
    returned = d[d["is_returned"]]
    n_ret = returned["order_id"].nunique()
    if n_ret == 0:
        return None
    overall = n_ret / n_orders
    total_cost = float(d["return_cost"].sum())
    avg_cost = total_cost / n_ret
    min_n = max(3, int(0.03 * n_orders))

    d = d.copy()
    if "delivery_time" in d.columns and (d["delivery_time"] > 0).any():
        d["delivery_bucket"] = pd.cut(
            d["delivery_time"], bins=[-1, 3, 5, np.inf], labels=["0-3 days", "4-5 days", "6+ days"]
        ).astype(str).replace("nan", "")
    dims = [c for c in ["product_name", "category", "city", "courier", "delivery_bucket"] if _usable(d, c)]

    evidence, causes, steps, gaps = [], [], [], []
    top_product = None
    for dim in dims:
        # Once a problem product is found, test the other dimensions on the REMAINING orders,
        # so one bad product cannot hide (or fake) a courier / city / category pattern.
        src, rate, mn, basis = d, overall, min_n, "All orders"
        if top_product and dim != "product_name":
            rest = d[d["product_name"].astype(str).str.strip() != top_product]
            r_orders = int(rest["order_id"].nunique())
            if r_orders >= 10:
                r_ret = int(rest.loc[rest["is_returned"], "order_id"].nunique())
                src, rate, mn = rest, r_ret / r_orders, max(3, int(0.03 * r_orders))
                basis = f"Excluding '{top_product}'"
        seg = _segments(src, dim, rate, mn)
        for name, r in seg.head(2).iterrows():
            avoidable = r["excess"] * avg_cost
            evidence.append({
                "Dimension": DIM_LABELS[dim], "Segment": name, "Basis": basis, "Orders": int(r["orders"]),
                "Returns": int(r["returns"]), "Return rate %": round(r["rate"], 1),
                "Baseline %": round(rate * 100, 1), "Times higher": round(r["lift"], 2),
                "Excess returns": round(r["excess"], 1), "Est. avoidable cost (Rs.)": round(avoidable),
            })
        if seg.empty:
            continue
        name, r = seg.index[0], seg.iloc[0]
        strength = _strength(int(r["orders"]), r["lift"], r["excess"])
        base = (f"{r['rate']:.1f}% return rate vs {rate*100:.1f}% baseline ({r['lift']:.1f}x), "
                f"about {r['excess']:.0f} more returns than expected")
        if basis != "All orders":
            base += f", after excluding '{top_product}'"
        basis = f"{int(r['returns'])} of {int(r['orders'])} orders returned (association, not proof)"
        avoid = f"Bringing {name} to the overall rate would avoid roughly {_rs(r['excess'] * avg_cost)} in return costs."
        if dim == "product_name":
            top_product = name
            causes.append({"cause": f"Returns concentrate on the product '{name}': {base}.", "strength": strength, "basis": basis})
            sub = returned[returned["product_name"].astype(str).str.strip() == name]
            rc = sub["return_reason"].astype(str).str.strip()
            rc = rc[~rc.str.lower().isin(BLANKS)].value_counts()
            if len(rc) and len(sub) >= 3 and rc.iloc[0] / len(sub) >= 0.5:
                share = rc.iloc[0] / len(sub) * 100
                st_ = "Strong" if len(sub) >= 8 else "Moderate"
                causes.append({
                    "cause": f"{share:.0f}% of '{name}' returns cite '{rc.index[0]}' - a repeated customer-reported reason points to a product/listing issue rather than random returns.",
                    "strength": st_, "basis": f"{int(rc.iloc[0])} of {len(sub)} returns of this product (customer-reported)"})
                steps.append(f"For '{name}': {_reason_action(rc.index[0])}")
            steps.append(avoid)
        elif dim == "courier":
            extra = ""
            if "delivery_time" in d.columns and (d["delivery_time"] > 0).any():
                cs = d[d["courier"].astype(str).str.strip() == name]["delivery_time"].mean()
                ov = d["delivery_time"].mean()
                if cs > ov + 0.5:
                    extra = f" Its average delivery time is {cs:.1f} days vs {ov:.1f} overall."
            causes.append({"cause": f"Courier '{name}' shows {base}.{extra} Possible delivery handling or speed problem.", "strength": strength, "basis": basis})
            steps.append(f"Compare {name}'s delivery times and damage/late-delivery complaints with other couriers; raise it with them or shift volume on affected routes. {avoid}")
        elif dim == "city":
            causes.append({"cause": f"Returns are elevated in {name}: {base}. Possible courier coverage, address-quality or COD-confirmation issue there.", "strength": strength, "basis": basis})
            steps.append(f"Check which courier serves {name}, confirm addresses/orders by phone before dispatch, and review COD rejections there.")
        elif dim == "delivery_bucket":
            causes.append({"cause": f"Orders delivered in {name} are returned more often: {base}. Slow delivery may be driving cancellations and returns.", "strength": strength, "basis": basis})
            steps.append(f"Set a delivery-time target and review dispatch and courier performance for orders that take {name}.")
        else:
            causes.append({"cause": f"{DIM_LABELS[dim]} '{name}' shows {base}.", "strength": strength, "basis": basis})
            steps.append(f"Review the '{name}' {DIM_LABELS[dim].lower()} segment for common issues.")

    tables = []
    if evidence:
        tables.append({"title": "Where returns concentrate (segments above the overall rate)", "rows": evidence})

    reasons = returned["return_reason"].astype(str).str.strip()
    reasons = reasons[~reasons.str.lower().isin(BLANKS)]
    top_reason = None
    if len(reasons):
        vc = reasons.value_counts()
        top_reason = vc.index[0]
        rows = [{"Reason (customer-reported)": k, "Returns": int(v), "Share of returns %": round(v / n_ret * 100, 1)}
                for k, v in vc.head(5).items()]
        tables.append({"title": "Why customers say they returned", "rows": rows})
        if vc.iloc[0] / n_ret >= 0.3 and not any(top_reason in c["cause"] for c in causes):
            st_ = "Strong" if vc.iloc[0] >= 8 and vc.iloc[0] / n_ret >= 0.5 else "Moderate"
            causes.append({"cause": f"'{top_reason}' is the most common stated reason ({vc.iloc[0] / n_ret * 100:.0f}% of returns).",
                           "strength": st_, "basis": f"{int(vc.iloc[0])} of {n_ret} returns (customer-reported)"})
            steps.append(_reason_action(top_reason))
    else:
        gaps.append("No return reasons recorded - capture a reason at return time to see why customers send items back")

    for col, label in [("courier", "courier"), ("city", "city"), ("category", "category")]:
        if not _usable(d, col):
            gaps.append(f"No usable {label} data - cannot test whether {label} drives returns")
    if "delivery_bucket" not in d.columns:
        gaps.append("No delivery_time data - cannot test whether slow delivery drives returns")

    if not causes:
        causes.append({"cause": "Returns are spread fairly evenly - no single product, courier or city stands out.",
                       "strength": "Moderate", "basis": f"No segment is 1.3x above the {overall*100:.1f}% overall rate with enough orders"})
        steps.append("Review return reasons and customer complaints manually for shared themes.")
    steps.append("After making changes, re-run this analysis on the next 2-4 weeks of orders to measure whether the return rate fell.")

    rate_pct = overall * 100
    sev = "High" if rate_pct >= 15 else "Medium" if rate_pct >= 8 else "Low"
    conf, note = _confidence(n_orders)
    seen, uniq = set(), []
    for s in steps:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return {
        "id": "returns", "title": f"Why returns are {rate_pct:.1f}%", "severity": sev,
        "headline": f"Return rate is {rate_pct:.1f}% ({n_ret} of {n_orders} orders), costing about {_rs(total_cost)}.",
        "observed": [f"{n_ret} of {n_orders} orders were returned ({rate_pct:.1f}%).",
                     f"Return-related cost (refunds + shipping) is about {_rs(total_cost)}, roughly {_rs(avg_cost)} per returned order."]
                    + ([f"Most common stated reason: {top_reason}."] if top_reason else []),
        "tables": tables, "possible_causes": causes, "next_steps": uniq,
        "confidence": conf, "confidence_note": note, "data_gaps": gaps,
        "focus": {"product": top_product, "return_cost": total_cost},
    }


def _investigate_margin(d, n_orders):
    g = d.groupby("product_name").agg(
        revenue=("revenue", "sum"), cost=("product_cost_total", "sum"),
        margin=("gross_margin", "sum"), return_cost=("return_cost", "sum"))
    g = g[g["revenue"] > 0]
    total_rev = float(d["revenue"].sum())
    if len(g) < 2 or total_rev <= 0:
        return None
    overall_m = float(d["gross_margin"].sum()) / total_rev * 100
    g["margin_pct"] = g["margin"] / g["revenue"] * 100
    g["net_pct"] = (g["margin"] - g["return_cost"]) / g["revenue"] * 100
    name = g["margin_pct"].idxmin()
    r = g.loc[name]
    gap = overall_m - r["margin_pct"]
    if gap < 3 and r["net_pct"] > 0:
        return None
    ratio = r["cost"] / r["revenue"] * 100
    overall_ratio = 100 - overall_m
    ret_share = r["return_cost"] / r["margin"] * 100 if r["margin"] > 0 else 100
    causes, steps = [], []
    causes.append({"cause": f"Product cost is {ratio:.0f}% of the selling price for '{name}' vs {overall_ratio:.0f}% store-wide.",
                   "strength": "Strong" if ratio >= overall_ratio + 5 else "Moderate",
                   "basis": f"Computed from {_rs(r['cost'])} cost on {_rs(r['revenue'])} revenue"})
    if ret_share >= 25:
        txt = (f"Return costs ({_rs(r['return_cost'])}) exceed the entire gross margin ({_rs(r['margin'])}) on '{name}', so its returns wipe out its profit."
               if ret_share >= 100 else
               f"Return costs consume {ret_share:.0f}% of the gross margin on '{name}', eroding profit further.")
        causes.append({"cause": txt, "strength": "Strong" if ret_share >= 40 else "Moderate",
                       "basis": f"{_rs(r['return_cost'])} return cost vs {_rs(r['margin'])} gross margin"})
    need_rev = r["cost"] / (1 - overall_m / 100) if overall_m < 100 else r["revenue"]
    price_up = (need_rev / r["revenue"] - 1) * 100
    cost_down = (1 - (r["revenue"] * (1 - overall_m / 100)) / r["cost"]) * 100
    if price_up > 0:
        steps.append(f"To match the store-average margin ({overall_m:.0f}%), '{name}' would need roughly a {price_up:.0f}% price rise (volume held constant) or a {cost_down:.0f}% supplier-cost reduction.")
    if ret_share >= 25:
        steps.append(f"Reduce returns on '{name}' first - every avoided return protects margin directly.")
    steps += [f"Check whether discounts or free shipping are being applied to '{name}' more than to other products.",
              "Test a small price increase or bundle on this product and compare margin and sales volume."]
    conf, note = _confidence(n_orders)
    rows = [{"Product": k, "Revenue (Rs.)": round(v.revenue), "Gross margin %": round(v.margin_pct, 1),
             "Margin after returns %": round(v.net_pct, 1)} for k, v in g.sort_values("margin_pct").iterrows()]
    sev = "High" if r["net_pct"] < 10 or gap >= 10 else "Medium"
    return {
        "id": "margin", "title": f"Why '{name}' has the lowest margin", "severity": sev,
        "headline": f"'{name}' margin is {r['margin_pct']:.1f}% vs {overall_m:.1f}% store-wide.",
        "observed": [f"'{name}' gross margin is {r['margin_pct']:.1f}% ({gap:.1f} points below the store average of {overall_m:.1f}%).",
                     f"After return costs its margin is about {r['net_pct']:.1f}%."],
        "tables": [{"title": "Margin by product", "rows": rows}], "possible_causes": causes,
        "next_steps": steps, "confidence": conf, "confidence_note": note,
        "data_gaps": ["No discount/promotion data - cannot tell whether discounting is lowering the margin"],
        "focus": {"product": name},
    }


def _investigate_customers(d, n_orders):
    c = d.groupby("customer_id").agg(orders=("order_id", "nunique"), returns=("is_returned", "sum"),
                                     return_cost=("return_cost", "sum"), spend=("revenue", "sum")).reset_index()
    c = c[~c["customer_id"].astype(str).str.lower().isin(BLANKS)]
    total_ret = int(d["is_returned"].sum())
    rep = c[(c["orders"] >= 3) & (c["returns"] >= 2) & (c["returns"] / c["orders"] >= 0.6)]
    if total_ret == 0 or rep.empty:
        return None
    rep = rep.assign(rate=rep["returns"] / rep["orders"] * 100).sort_values("returns", ascending=False)
    share = rep["returns"].sum() / total_ret * 100
    cost = float(rep["return_cost"].sum())
    conf, note = _confidence(n_orders)
    rows = [{"Customer": r.customer_id, "Orders": int(r.orders), "Returns": int(r.returns),
             "Return rate %": round(r.rate, 1), "Return cost (Rs.)": round(r.return_cost)} for r in rep.head(5).itertuples()]
    return {
        "id": "customers", "title": "Repeat customers who return most of what they buy",
        "severity": "Medium" if share >= 25 else "Low",
        "headline": f"{len(rep)} repeat customers account for {share:.0f}% of returns.",
        "observed": [f"{len(rep)} customers with 3+ orders returned at least 60% of them.",
                     f"Together they account for {share:.0f}% of all returns and about {_rs(cost)} in return cost."],
        "tables": [{"title": "Highest-return repeat customers", "rows": rows}],
        "possible_causes": [{"cause": "Habitual ordering-to-try-and-return, or a repeated fit/quality problem for these customers (the data cannot tell which).",
                             "strength": "Weak", "basis": f"{len(rep)} customers, small sample per customer"}],
        "next_steps": ["Contact the top returners to learn why they return - their answers may reveal fixable product or size issues.",
                       "If returns are habitual, offer size exchanges instead of refunds, or set limits for repeat returners.",
                       "Flag these customers for confirmation calls before dispatching high-value orders."],
        "confidence": conf, "confidence_note": note,
        "data_gaps": ["No per-customer complaint history beyond the returns themselves"],
        "focus": {},
    }


def _investigate_zero_revenue(d, health):
    if float(d["revenue"].sum()) > 0:
        return None
    causes = [{"cause": w, "strength": "Strong", "basis": "Detected while reading the uploaded columns"}
              for w in (health or {}).get("warnings", [])[:4]]
    if not causes:
        causes = [{"cause": "Price or quantity values are empty or zero in every row.", "strength": "Moderate", "basis": "Revenue = quantity x selling price summed to 0"}]
    return {
        "id": "revenue_zero", "title": "Why revenue shows Rs. 0", "severity": "High",
        "headline": "Revenue is 0 - the file's price/quantity columns could not be used.",
        "observed": ["Revenue is calculated as quantity x selling price for each order; the total is 0."],
        "tables": [], "possible_causes": causes,
        "next_steps": ["Check the Data Health panel for which columns were found, missing or unreadable.",
                       "Rename the price column to 'selling_price' (or 'price' / 'unit_price') and the quantity column to 'quantity', then re-upload.",
                       "Make sure price values are numeric (e.g. 2500, not 'Rs. 2,500 only')."],
        "confidence": "High", "confidence_note": "This is a data-reading problem, not a sample-size one.",
        "data_gaps": [], "focus": {},
    }


def investigate(d, health=None):
    """Run all investigations on a normalised DataFrame; most severe first."""
    if d is None or len(d) == 0:
        return []
    n_orders = int(d["order_id"].nunique())
    findings = [_investigate_zero_revenue(d, health), _investigate_returns(d, n_orders),
                _investigate_margin(d, n_orders), _investigate_customers(d, n_orders)]
    findings = [f for f in findings if f]
    return sorted(findings, key=lambda f: SEVERITY_ORDER.get(f["severity"], 3))
