import pandas as pd
import numpy as np
from investigation import investigate

ALIASES = {
    "price": "selling_price", "unit_price": "selling_price", "sale_price": "selling_price",
    "item_price": "selling_price", "price_per_unit": "selling_price", "sell_price": "selling_price",
    "cost": "product_cost", "unit_cost": "product_cost", "cost_price": "product_cost",
    "qty": "quantity", "quantity_ordered": "quantity", "order_quantity": "quantity",
    "order_item_quantity": "quantity", "units": "quantity",
    "customer": "customer_id", "customerid": "customer_id",
    "product": "product_name", "item_name": "product_name",
    "returns": "return_status", "returned": "return_status", "is_returned": "return_status",
    "status": "order_status", "delivery_days": "delivery_time", "shipping_fee": "shipping_cost",
    "order_no": "order_id", "orderid": "order_id", "date": "order_date",
}
NUMERIC = ["quantity", "selling_price", "product_cost", "stock_quantity",
           "refund_amount", "delivery_time", "shipping_cost", "return_shipping_cost"]
TEXT = ["return_status", "return_reason", "customer_complaint", "order_status"]
DEFAULTS = {"quantity": 1}  # all other missing numeric columns default to 0
RETURNED_STATUSES = ["yes", "returned", "true", "1", "y"]
RETURNED_ORDER_STATUSES = ["returned", "return", "refunded"]

# (column, why it matters) - used by the Data Health panel
HEALTH_COLUMNS = [
    ("selling_price", "Revenue and margin"), ("quantity", "Revenue (defaults to 1 per row if missing)"),
    ("product_cost", "Gross margin"), ("order_id", "Order counts"), ("customer_id", "Customer analysis"),
    ("product_name", "Product analysis"), ("return_status", "Returns analysis"),
    ("return_reason", "Why customers return"), ("refund_amount", "Return cost"),
    ("shipping_cost", "Return cost"), ("order_status", "Return detection (backup)"),
    ("city", "Investigation: where returns happen"), ("courier", "Investigation: delivery partner"),
    ("category", "Investigation: category"), ("delivery_time", "Investigation: delivery speed"),
    ("stock_quantity", "Stock insight"),
]


def _to_number(s):
    """Numeric parse. Falls back to stripping symbols ('Rs. 1,200' -> 1200) only where plain parsing fails.
    Returns (values, n_cleaned, n_unreadable). Works on an in-memory copy; originals are never touched."""
    parsed = pd.to_numeric(s, errors="coerce")
    mask = parsed.isna() & s.notna() & (s.astype(str).str.strip() != "")
    n_clean = n_bad = 0
    if mask.any():
        cleaned = s[mask].astype(str).str.replace(r"[^0-9.\-]", "", regex=True).str.strip(".").replace("", np.nan)
        parsed.loc[mask] = pd.to_numeric(cleaned, errors="coerce")
        n_bad = int(parsed[mask].isna().sum())
        n_clean = int(mask.sum()) - n_bad
    return parsed, n_clean, n_bad


def _prepare(df):
    """Normalise a raw DataFrame and report what could / could not be read."""
    d = df.copy()
    d.columns = [str(c).strip().lower().replace(" ", "_") for c in d.columns]
    d = d.rename(columns={c: ALIASES.get(c, c) for c in d.columns})
    d = d.loc[:, ~d.columns.duplicated()]
    present = set(d.columns)
    warnings, infos, defaulted = [], [], {}

    for c in NUMERIC:
        if c not in d:
            d[c] = DEFAULTS.get(c, 0)
            defaulted[c] = DEFAULTS.get(c, 0)
            continue
        d[c], n_clean, n_bad = _to_number(d[c])
        d[c] = d[c].fillna(0)
        if n_clean:
            infos.append(f"'{c}': {n_clean} text values (like 'Rs. 1,200') were cleaned into numbers.")
        if n_bad:
            warnings.append(f"'{c}': {n_bad} values could not be read as numbers and were treated as 0.")
    for c in TEXT:
        if c not in d:
            d[c] = ""
        d[c] = d[c].fillna("").astype(str).str.strip()

    if "order_id" not in d:
        d["order_id"] = range(1, len(d) + 1)
    if "customer_id" not in d:
        d["customer_id"] = "Unknown"
    if "product_name" not in d:
        d["product_name"] = "Unknown"

    flag = d["return_status"].str.lower().isin(RETURNED_STATUSES)
    if "order_status" in present:
        flag = flag | d["order_status"].str.lower().isin(RETURNED_ORDER_STATUSES)
    d["is_returned"] = flag
    d["revenue"] = d["quantity"] * d["selling_price"]
    d["product_cost_total"] = d["quantity"] * d["product_cost"]
    d["gross_margin"] = d["revenue"] - d["product_cost_total"]
    d["return_cost"] = np.where(d["is_returned"], d["shipping_cost"] + d["refund_amount"], 0)

    # --- health report ---
    if len(d) == 0:
        warnings.append("The file has no data rows.")
    if "selling_price" in defaulted:
        warnings.append("No price column found (looked for selling_price, price, unit_price, item_price, ...). Revenue will be 0.")
    elif len(d) and (d["selling_price"] <= 0).all():
        warnings.append("Every selling price is 0 or unreadable. Revenue will be 0.")
    if "quantity" in defaulted:
        warnings.append("No quantity column found (looked for quantity, qty, ...). Assumed 1 unit per row, so revenue = price x 1.")
    elif len(d) and (d["quantity"] <= 0).all():
        warnings.append("Every quantity is 0 or unreadable. Revenue will be 0.")
    if "product_cost" in defaulted:
        warnings.append("No product cost column found - gross margin equals revenue and margin insights are not meaningful.")
    if "order_id" in present and d["order_id"].duplicated().any():
        infos.append(f"{int(d['order_id'].duplicated().sum())} rows share an order_id (multi-item orders or duplicates); orders are counted once per unique order_id.")
    if "return_status" not in present and "order_status" in present and d["is_returned"].any():
        infos.append("No return column found - returns were detected from order_status instead.")
    elif "return_status" not in present and not d["is_returned"].any():
        infos.append("No return column found - the returns analysis will show no returns.")

    rows = []
    for col, why in HEALTH_COLUMNS:
        if col in present:
            status, note = "Found", why
        elif col in defaulted:
            status, note = f"Missing - default {defaulted[col]}", why
        else:
            status, note = "Missing", why
        rows.append({"Column": col, "Status": status, "Used for": note})
    health = {"rows": len(d), "columns": rows, "warnings": warnings, "infos": infos}
    return d, health


def normalize(df):
    return _prepare(df)[0]


def run_analysis(df):
    d, health = _prepare(df)
    orders = int(d["order_id"].nunique())
    revenue = float(d["revenue"].sum())
    returns = int(d.loc[d["is_returned"], "order_id"].nunique())
    return_rate = returns / orders * 100 if orders else 0
    return_cost = float(d["return_cost"].sum())
    aov = revenue / orders if orders else 0
    customers = d["customer_id"].nunique()

    sales = d.groupby("product_name").agg(
        revenue=("revenue", "sum"), orders=("order_id", "nunique"), quantity=("quantity", "sum")
    ).sort_values("revenue", ascending=False).reset_index()
    sales["aov"] = sales["revenue"] / sales["orders"].replace(0, 1)

    prod = d.groupby("product_name").agg(
        revenue=("revenue", "sum"), cost=("product_cost_total", "sum"),
        margin=("gross_margin", "sum"), stock=("stock_quantity", "mean"),
        returns=("is_returned", "sum"), orders=("order_id", "nunique"), units=("quantity", "sum")
    ).reset_index()
    prod["margin_pct"] = np.where(prod["revenue"] > 0, prod["margin"] / prod["revenue"] * 100, 0)
    prod["return_rate"] = np.where(prod["orders"] > 0, prod["returns"] / prod["orders"] * 100, 0)

    cust = d.groupby("customer_id").agg(
        orders=("order_id", "nunique"), spend=("revenue", "sum"),
        returns=("is_returned", "sum")
    ).reset_index().sort_values("spend", ascending=False)
    cust["return_rate"] = cust["returns"] / cust["orders"].replace(0, 1) * 100

    reasons = d[d["is_returned"]].groupby("return_reason").agg(
        returns=("order_id", "count"), refund=("refund_amount", "sum")
    ).reset_index().sort_values("returns", ascending=False)

    sales_high = []
    if not sales.empty:
        sales_high.append(f"Top revenue product: {sales.iloc[0].product_name}")
        sales_high.append(f"Top product revenue: Rs. {sales.iloc[0].revenue:,.0f}")
    sales_high.append(f"Average order value: Rs. {aov:,.0f}")
    sales_high.append(f"{orders:,} orders analyzed")

    product_high = []
    if not prod.empty:
        low_margin = prod.sort_values("margin_pct").iloc[0]
        product_high.append(f"Lowest-margin product: {low_margin.product_name}")
        product_high.append(f"Lowest margin rate: {low_margin.margin_pct:.1f}%")
        low_stock = prod.sort_values("stock").iloc[0]
        product_high.append(f"Lowest average stock: {low_stock.product_name}")
        high_return = prod.sort_values("return_rate", ascending=False).iloc[0]
        product_high.append(f"Highest return-rate product: {high_return.product_name}")

    customer_high = []
    if not cust.empty:
        top = cust.iloc[0]
        customer_high.append(f"Highest-value customer: {top.customer_id}")
        customer_high.append(f"Top customer spend: Rs. {top.spend:,.0f}")
        customer_high.append(f"Customers analyzed: {customers}")
        repeat = int((cust["orders"] > 1).sum())
        customer_high.append(f"Repeat customers: {repeat}")

    return_high = [
        f"Overall return rate: {return_rate:.1f}%",
        f"Returned orders: {returns:,}",
        f"Return/refund cost: Rs. {return_cost:,.0f}",
        f"Top return reason: {reasons.iloc[0].return_reason if not reasons.empty else 'No returns'}"
    ]

    fin_high = [
        f"Revenue: Rs. {revenue:,.0f}",
        f"Gross margin: Rs. {d['gross_margin'].sum():,.0f}",
        f"Refund amount: Rs. {d['refund_amount'].sum():,.0f}",
        f"Estimated return cost: Rs. {return_cost:,.0f}"
    ]

    # Investigation layer: why it happened + what to do next
    findings = investigate(d, health)
    by_id = {f["id"]: f for f in findings}

    actions = []
    # Cross-agent signal: the same product is the top returner AND the lowest-margin product
    rf, mf = by_id.get("returns"), by_id.get("margin")
    if rf and mf and rf["focus"].get("product") and rf["focus"]["product"] == mf["focus"].get("product"):
        pname = rf["focus"]["product"]
        actions.append({
            "priority": "High",
            "problem": f"{pname} is both the top returned product (Returns agent) and the lowest-margin product (Financial agent).",
            "evidence": f"{rf['headline']} {mf['headline']}",
            "financial_impact": f"Each return erodes an already thin margin; total return cost is about Rs. {rf['focus']['return_cost']:,.0f}.",
            "suggested_action": rf["next_steps"][0] if rf["next_steps"] else "Fix return causes on this product before promoting it."
        })
    if return_rate >= 10:
        actions.append({
            "priority": "High",
            "problem": "Returns are materially affecting the order base.",
            "evidence": f"Return rate is {return_rate:.1f}% across {orders} orders.",
            "financial_impact": f"Refund/return-related cost is approximately Rs. {return_cost:,.0f}.",
            "suggested_action": "Investigate the highest-return products and reasons before increasing promotion."
        })
    if not prod.empty:
        hp = prod.sort_values("return_rate", ascending=False).iloc[0]
        if hp.return_rate > 0:
            hp_cost = float(d.loc[(d["product_name"] == hp.product_name), "return_cost"].sum())
            actions.append({
                "priority": "High",
                "problem": f"{hp.product_name} shows the highest observed return rate.",
                "evidence": f"Observed return rate is {hp.return_rate:.1f}%.",
                "financial_impact": f"Return-related cost for this product is about Rs. {hp_cost:,.0f}.",
                "suggested_action": "Check product quality, sizing/description accuracy, and customer complaints."
            })
    if not prod.empty:
        lm = prod.sort_values("margin_pct").iloc[0]
        actions.append({
            "priority": "Medium",
            "problem": f"{lm.product_name} has the lowest observed margin percentage.",
            "evidence": f"Margin rate is {lm.margin_pct:.1f}%.",
            "financial_impact": f"Revenue is Rs. {lm.revenue:,.0f} while gross margin is Rs. {lm.margin:,.0f}.",
            "suggested_action": "Review pricing, product cost, discounts, and return-related costs."
        })

    return {
        "kpis": {"revenue": revenue, "orders": orders, "return_rate": return_rate,
                 "return_cost": return_cost, "aov": aov, "customers": customers},
        "health": health,
        "investigation": findings,
        "agents": [
            {"name": "Sales Agent", "icon": "🛒", "highlights": sales_high},
            {"name": "Product Agent", "icon": "📦", "highlights": product_high},
            {"name": "Customer Agent", "icon": "👥", "highlights": customer_high},
            {"name": "Returns Agent", "icon": "🔄", "highlights": return_high},
            {"name": "Financial Agent", "icon": "💰", "highlights": fin_high},
        ],
        "details": {
            "sales": sales.to_dict("records"),
            "product": prod.to_dict("records"),
            "customer": cust.to_dict("records"),
            "returns": reasons.to_dict("records"),
            "financial": [
                {"metric": "Revenue", "value": revenue},
                {"metric": "Product Cost", "value": float(d["product_cost_total"].sum())},
                {"metric": "Gross Margin", "value": float(d["gross_margin"].sum())},
                {"metric": "Refund Amount", "value": float(d["refund_amount"].sum())},
                {"metric": "Return Cost", "value": return_cost},
            ]
        },
        "orchestrator": {"actions": actions}
    }
