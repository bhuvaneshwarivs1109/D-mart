"""
DMart: Market Basket Analysis + Perishables Wastage Planner
AI Foundations for Managers - project demo (uses DUMMY data)

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

# ----------------------------------------------------------------------
# 1. DUMMY DATA
# name: (is_perishable, selling price Rs, cost price Rs, base units sold/day)
# ----------------------------------------------------------------------
PRODUCTS = {
    "Tomato": (True, 30, 22, 120), "Onion": (True, 35, 27, 140),
    "Potato": (True, 28, 21, 150), "Banana": (True, 50, 38, 90),
    "Spinach": (True, 20, 14, 60), "Milk": (True, 56, 52, 200),
    "Curd": (True, 40, 33, 90), "Paneer": (True, 95, 80, 45),
    "Bread": (True, 45, 36, 110), "Butter": (True, 58, 50, 50),
    "Rice": (False, 70, 60, 80), "Dal": (False, 130, 112, 60),
    "Cooking Oil": (False, 150, 135, 70), "Atta": (False, 55, 47, 90),
    "Sugar": (False, 48, 42, 70), "Tea": (False, 120, 100, 50),
    "Biscuits": (False, 30, 24, 100),
}
PERISHABLES = [p for p, v in PRODUCTS.items() if v[0]]

# Items that people tend to buy together (used to create realistic dummy bills)
PATTERNS = [
    (["Tomato", "Onion", "Potato"], 0.18),
    (["Milk", "Bread", "Butter"], 0.14),
    (["Rice", "Dal", "Cooking Oil"], 0.16),
    (["Milk", "Tea", "Sugar"], 0.12),
    (["Curd", "Rice"], 0.08),
    (["Paneer", "Spinach", "Onion"], 0.07),
    (["Banana", "Milk"], 0.08),
    (["Atta", "Cooking Oil", "Sugar"], 0.10),
    (["Biscuits", "Tea"], 0.09),
]
FESTIVALS = pd.to_datetime(["2026-08-15", "2026-09-14", "2026-10-20"])


@st.cache_data
def generate_baskets(n_bills=2000, seed=42):
    rng = np.random.default_rng(seed)
    names = list(PRODUCTS)
    bills = []
    for _ in range(n_bills):
        basket = set()
        for items, prob in PATTERNS:
            if rng.random() < prob:
                basket.update(i for i in items if rng.random() < 0.9)
        extra = rng.integers(1, 4)
        basket.update(rng.choice(names, size=extra, replace=False))
        bills.append(sorted(basket))
    return bills


@st.cache_data
def generate_daily_sales(seed=7):
    """124 days of perishable sales. Current store rule: order = last 7-day
    average sold x 1.20 (it ignores weekends and festivals)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-07-01", periods=124)
    rows = []
    for name in PERISHABLES:
        _, price, cost, base = PRODUCTS[name]
        history = [base] * 7
        for d in dates:
            weekend = d.dayofweek >= 5
            festival = d in FESTIVALS
            demand = base * (1.35 if weekend else 1.0) * (1.4 if festival else 1.0)
            demand = max(int(round(demand * rng.normal(1, 0.12))), 0)
            stock = int(round(np.mean(history[-7:]) * 1.20))
            sold = min(demand, stock)
            history.append(sold)
            rows.append(dict(date=d, product=name, price=price, cost=cost,
                             dow=d.dayofweek, is_weekend=int(weekend),
                             is_festival=int(festival), demand=demand,
                             ordered=stock, sold=sold))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 2. MARKET BASKET ANALYSIS
# ----------------------------------------------------------------------
@st.cache_data
def run_basket_analysis(bills, min_support=0.04, min_lift=1.2):
    te = TransactionEncoder()
    onehot = pd.DataFrame(te.fit(bills).transform(bills), columns=te.columns_)
    freq = apriori(onehot, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=min_lift)
    rules = rules[(rules["antecedents"].apply(len) == 1) &
                  (rules["consequents"].apply(len) == 1)].copy()
    rules["if_customer_buys"] = rules["antecedents"].apply(lambda s: next(iter(s)))
    rules["they_also_buy"] = rules["consequents"].apply(lambda s: next(iter(s)))
    rules = rules[["if_customer_buys", "they_also_buy", "support",
                   "confidence", "lift"]].sort_values("lift", ascending=False)
    return rules.reset_index(drop=True), onehot


def basket_uplift(onehot, antecedent, consequent, conversion):
    """What-if: some customers who buy `antecedent` but not `consequent`
    add `consequent` because of shelf placement / combo offer."""
    candidates = ((onehot[antecedent]) & (~onehot[consequent])).sum()
    extra_items = candidates * conversion
    old_size = onehot.sum(axis=1).mean()
    new_size = old_size + extra_items / len(onehot)
    return old_size, new_size, int(candidates)


# ----------------------------------------------------------------------
# 3. WASTAGE PREDICTOR
# ----------------------------------------------------------------------
FEATURES = ["product_code", "dow", "is_weekend", "is_festival", "lag1", "roll7"]


def add_features(df):
    df = df.sort_values(["product", "date"]).copy()
    g = df.groupby("product")["sold"]
    df["lag1"] = g.shift(1)
    df["roll7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["product_code"] = df["product"].map({p: i for i, p in enumerate(PERISHABLES)})
    return df.dropna()


@st.cache_resource
def train_model(daily):
    df = add_features(daily)
    cutoff = df["date"].max() - pd.Timedelta(days=27)  # last 28 days = test
    train, test = df[df["date"] < cutoff], df[df["date"] >= cutoff].copy()
    model = RandomForestRegressor(n_estimators=200, random_state=0)
    model.fit(train[FEATURES], train["sold"])
    test["predicted"] = model.predict(test[FEATURES])
    mape = mean_absolute_percentage_error(test["sold"], test["predicted"])
    return model, test, mape


def kpis(demand, ordered, price, cost):
    """Wastage %, lost sales and margin for a given ordering plan."""
    sold = np.minimum(demand, ordered)
    wasted = ordered - sold
    margin = (sold * (price - cost)).sum() - (wasted * cost).sum()
    return dict(wastage_pct=100 * wasted.sum() / ordered.sum(),
                units_wasted=int(wasted.sum()),
                lost_sales_units=int((demand - sold).sum()),
                margin=float(margin))


def compare_plans(test, buffer):
    model_order = np.ceil(test["predicted"] * (1 + buffer))
    old = kpis(test["demand"], test["ordered"], test["price"], test["cost"])
    new = kpis(test["demand"], model_order, test["price"], test["cost"])
    return old, new


def next_day_plan(daily, model, buffer, is_festival, uplift_products, uplift):
    last = daily["date"].max()
    day = last + pd.Timedelta(days=1)
    rows = []
    for p in PERISHABLES:
        hist = daily[daily["product"] == p].sort_values("date")["sold"]
        x = pd.DataFrame([dict(product_code=PERISHABLES.index(p), dow=day.dayofweek,
                               is_weekend=int(day.dayofweek >= 5),
                               is_festival=int(is_festival),
                               lag1=hist.iloc[-1], roll7=hist.iloc[-7:].mean())])
        pred = model.predict(x[FEATURES])[0]
        boost = (1 + uplift) if p in uplift_products else 1.0
        rows.append(dict(product=p, expected_demand=round(pred * boost),
                         recommended_order=int(np.ceil(pred * boost * (1 + buffer))),
                         combo_uplift_applied="Yes" if p in uplift_products else ""))
    return day, pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 4. DASHBOARD
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="DMart AI Planner", layout="wide")
    st.title("DMart: Basket Analysis + Perishables Wastage Planner")
    st.caption("AI Foundations for Managers | demo with dummy data")

    bills = generate_baskets()
    daily = generate_daily_sales()

    st.sidebar.header("Settings")
    min_support = st.sidebar.slider("Min support (basket analysis)", 0.02, 0.15, 0.04, 0.01)
    min_lift = st.sidebar.slider("Min lift", 1.0, 3.0, 1.2, 0.1)
    buffer_pct = st.sidebar.slider("Safety buffer on orders (%)", 0, 30, 8)
    buffer = buffer_pct / 100

    rules, onehot = run_basket_analysis(bills, min_support, min_lift)
    model, test, mape = train_model(daily)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "1. Market basket", "2. Wastage predictor", "3. Combined plan"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Bills analysed", f"{len(bills):,}")
        c2.metric("Avg basket size (items)", f"{onehot.sum(axis=1).mean():.2f}")
        c3.metric("Rules found", len(rules))
        st.markdown(
            "**Business problem:** DMart runs on thin margins. Spoiled perishables and "
            "stockouts both cut profit, and shelf placement affects basket size.\n\n"
            "**AI approach:** (1) Association rules (Apriori) find what sells together. "
            "(2) A Random Forest predicts daily demand to set smarter order quantities. "
            "(3) The combined plan raises the order for items in a promoted combo.")
        st.subheader("Top-selling items in dummy bills")
        st.bar_chart(onehot.sum().sort_values(ascending=False))

    with tab2:
        st.subheader("Products that sell together")
        st.write("**Support** = how common the pair is. **Confidence** = chance the "
                 "second item is bought when the first is. **Lift** > 1 means they "
                 "are bought together more than by chance.")
        st.dataframe(rules.style.format({"support": "{:.3f}", "confidence": "{:.2f}",
                                         "lift": "{:.2f}"}), width='stretch')
        if len(rules):
            top = rules.head(10).copy()
            top["pair"] = top["if_customer_buys"] + " -> " + top["they_also_buy"]
            st.bar_chart(top.set_index("pair")["lift"])
            best = rules.iloc[0]
            st.info(f"Shelf layout idea: place **{best['if_customer_buys']}** next to "
                    f"**{best['they_also_buy']}** (lift {best['lift']:.2f}).")

    with tab3:
        st.subheader("Order quantities: current rule vs AI model (last 28 days)")
        old, new = compare_plans(test, buffer)
        st.caption(f"Model forecast error (MAPE): {mape*100:.1f}%. Current rule = last "
                   "7-day average x 1.20, which ignores weekends and festivals.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Wastage % (current)", f"{old['wastage_pct']:.1f}%")
        c2.metric("Wastage % (AI)", f"{new['wastage_pct']:.1f}%",
                  f"{new['wastage_pct']-old['wastage_pct']:.1f} pts", delta_color="inverse")
        c3.metric("Lost sales units (AI)", new["lost_sales_units"],
                  new["lost_sales_units"] - old["lost_sales_units"], delta_color="inverse")
        c4.metric("Margin saved (Rs)", f"{new['margin']-old['margin']:,.0f}")
        daily_view = test.groupby("date")[["demand", "ordered", "predicted"]].sum()
        st.line_chart(daily_view.rename(columns={
            "demand": "Actual demand", "ordered": "Current order",
            "predicted": "AI forecast"}))

    with tab4:
        st.subheader("Combined plan: promote a combo, then order for it")
        perishable_rules = rules[rules["if_customer_buys"].isin(PERISHABLES) |
                                 rules["they_also_buy"].isin(PERISHABLES)]
        if perishable_rules.empty:
            st.warning("No rules with perishables. Lower min support or min lift.")
            return
        labels = (perishable_rules["if_customer_buys"] + " -> " +
                  perishable_rules["they_also_buy"] +
                  "  (lift " + perishable_rules["lift"].round(2).astype(str) + ")")
        choice = st.selectbox("Choose a combo to promote", labels.tolist())
        row = perishable_rules.iloc[labels.tolist().index(choice)]
        a, b = row["if_customer_buys"], row["they_also_buy"]

        conv = st.slider("Share of eligible customers who add the 2nd item (%)",
                         0, 50, 10) / 100
        old_size, new_size, cand = basket_uplift(onehot, a, b, conv)
        c1, c2, c3 = st.columns(3)
        c1.metric("Eligible bills", cand)
        c2.metric("Avg basket size now", f"{old_size:.2f}")
        c3.metric("Avg basket size with combo", f"{new_size:.2f}",
                  f"{new_size-old_size:+.3f}")

        st.markdown("#### Tomorrow's order plan")
        festival = st.checkbox("Tomorrow is a festival day")
        uplift_items = [i for i in (a, b) if i in PERISHABLES]
        day, plan = next_day_plan(daily, model, buffer, festival, uplift_items,
                                  uplift=conv)
        st.write(f"Plan for **{day.strftime('%A, %d %b %Y')}** "
                 f"(buffer {buffer_pct}%). Items in the combo get extra stock.")
        st.dataframe(plan, width='stretch', hide_index=True)
        st.download_button("Download order plan (CSV)",
                           plan.to_csv(index=False), "order_plan.csv", "text/csv")


if __name__ == "__main__":
    main()
