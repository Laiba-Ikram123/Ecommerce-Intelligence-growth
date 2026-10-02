"""Demo dataset for the E-Commerce Intelligence app.

Deterministic (fixed seed) so every demo run shows the same numbers.
Same 19-column schema as the original 10-row demo, but ~180 orders with
realistic patterns the Investigation layer can discover:
  * Classic Kurti has a very high return rate, mostly "Size issue".
  * TCS deliveries that take 6+ days return more often (late / damaged).
  * A few repeat customers return most of what they buy.
"""
import numpy as np
import pandas as pd

COLUMNS = [
    "order_id", "order_date", "product_id", "product_name", "category", "quantity",
    "selling_price", "product_cost", "customer_id", "city", "order_status",
    "stock_quantity", "return_status", "return_reason", "refund_amount",
    "customer_complaint", "courier", "delivery_time", "shipping_cost",
]

# id, name, category, price, cost, base return probability, base stock, sales weight
PRODUCTS = [
    ("P01", "Classic Kurti", "Fashion", 2500, 1700, 0.36, 14, 0.32),
    ("P02", "Linen Set", "Fashion", 4200, 2700, 0.07, 40, 0.22),
    ("P03", "Everyday Tote", "Accessories", 1800, 950, 0.05, 55, 0.26),
    ("P04", "Silk Scarf", "Accessories", 1200, 600, 0.05, 80, 0.20),
]

CITY_COURIERS = {
    "Karachi": (["TCS", "Leopards"], [0.7, 0.3]),
    "Rawalpindi": (["TCS", "M&P"], [0.6, 0.4]),
    "Lahore": (["Leopards", "M&P"], [0.6, 0.4]),
    "Islamabad": (["M&P", "Leopards"], [0.6, 0.4]),
    "Peshawar": (["TCS", "Leopards"], [0.5, 0.5]),
}
COURIER_COST = {"TCS": 250, "Leopards": 300, "M&P": 200}
COURIER_DAYS = {"TCS": (6.0, 1.0), "Leopards": (4.0, 1.0), "M&P": (3.0, 0.8)}
SERIAL_RETURNERS = {"C007", "C023", "C041"}


def demo_data(n_orders: int = 180, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 90, n_orders)), unit="D"
    )
    weights = np.array([p[7] for p in PRODUCTS])
    cities = list(CITY_COURIERS)
    rows = []
    for i in range(n_orders):
        pid, pname, cat, price, cost, base_p, base_stock, _ = PRODUCTS[
            rng.choice(len(PRODUCTS), p=weights / weights.sum())
        ]
        city = cities[rng.integers(0, len(cities))]
        couriers, cprobs = CITY_COURIERS[city]
        courier = couriers[rng.choice(len(couriers), p=cprobs)]
        mu, sd = COURIER_DAYS[courier]
        days = int(np.clip(round(rng.normal(mu, sd)), 1, 10))
        cust = f"C{rng.integers(1, 61):03d}"
        qty = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))

        p_ret = base_p
        if courier == "TCS" and days >= 6:
            p_ret += 0.10
        if cust in SERIAL_RETURNERS:
            p_ret += 0.30
        returned = rng.random() < min(p_ret, 0.9)

        reason = ""
        if returned:
            if pname == "Classic Kurti":
                reason = rng.choice(["Size issue", "Quality issue", "Changed mind"], p=[0.7, 0.2, 0.1])
            elif courier == "TCS" and days >= 6:
                reason = rng.choice(["Late delivery", "Damaged in transit", "Changed mind"], p=[0.45, 0.4, 0.15])
            else:
                reason = rng.choice(["Quality issue", "Not as described", "Changed mind"], p=[0.4, 0.3, 0.3])

        rows.append([
            f"O{1001 + i}", dates[i].strftime("%Y-%m-%d"), pid, pname, cat, qty,
            price, cost, cust, city, "Returned" if returned else "Delivered",
            max(0, base_stock + int(rng.integers(-3, 4))),
            "Yes" if returned else "No", str(reason), price * qty if returned else 0,
            str(reason) if returned else "No", courier, days, COURIER_COST[courier],
        ])
    return pd.DataFrame(rows, columns=COLUMNS)
