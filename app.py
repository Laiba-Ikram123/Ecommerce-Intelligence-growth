import hashlib
import json
import pandas as pd
import streamlit as st
from analysis import run_analysis
from ai_orchestrator import generate_ai_insights, build_compact_payload
from demo_data import demo_data

try:
    import plotly.express as px
except Exception:  # plotly is optional; the app still works without charts
    px = None

# Streamlit >= 1.52 prefers width="stretch"; older versions only know use_container_width.
try:
    _v = tuple(int(x) for x in st.__version__.split(".")[:2])
except ValueError:
    _v = (0, 0)
STRETCH = {"width": "stretch"} if _v >= (1, 52) else {"use_container_width": True}

st.set_page_config(
    page_title="E-Commerce Intelligence",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-Commerce Intelligence & Growth Analyst")
st.caption("Five-agent business intelligence system: Sales • Product • Customer • Returns • Financial — plus an Investigation layer that explains why it happened and what to do next")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_demo = st.button("Load Demo Dataset", **STRETCH)
    st.divider()
    st.subheader("AI Orchestrator")
    groq_key = st.text_input("Groq API Key (optional)", type="password")
    st.caption("Without a key, the rule-based orchestrator still works.")


def read_csv(file):
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")


@st.cache_data(show_spinner=False)
def cached_analysis(frame):
    return run_analysis(frame)


# Keep the dataset in session_state so it survives Streamlit reruns
# (e.g. typing the API key no longer wipes a loaded demo dataset).
if use_demo:
    st.session_state["df"] = demo_data()
    st.session_state["source"] = "Demo dataset"
elif uploaded is not None:
    upload_key = (uploaded.name, uploaded.size)
    if st.session_state.get("upload_key") != upload_key:
        try:
            st.session_state["df"] = read_csv(uploaded)
            st.session_state["source"] = uploaded.name
            st.session_state["upload_key"] = upload_key
        except Exception as e:
            st.error(f"Could not read this file as a CSV: {e}")
            st.stop()

df = st.session_state.get("df")
if df is None:
    st.info("Upload a CSV or click **Load Demo Dataset** to start.")
    st.stop()

st.caption(f"Source: **{st.session_state.get('source', 'dataset')}** · {len(df):,} rows · {df.shape[1]} columns")

result = cached_analysis(df)

# KPI row
k = result["kpis"]
cols = st.columns(6)
for col, (label, value) in zip(cols, [
    ("Revenue", f"Rs. {k['revenue']:,.0f}"),
    ("Orders", f"{k['orders']:,}"),
    ("Return Rate", f"{k['return_rate']:.1f}%"),
    ("Return Cost", f"Rs. {k['return_cost']:,.0f}"),
    ("AOV", f"Rs. {k['aov']:,.0f}"),
    ("Customers", f"{k['customers']:,}")
]):
    col.metric(label, value)

if k["revenue"] == 0:
    st.error("Revenue is Rs. 0 — the price/quantity columns could not be used. See **Data Health** and the **Investigation** section below.")

# Data health: tells you what the analysis could and could not read
h = result["health"]
with st.expander("🩺 Data Health — what the analysis could and couldn't read", expanded=bool(h["warnings"])):
    if not h["warnings"]:
        st.success("All required columns were read successfully.")
    for w in h["warnings"]:
        st.warning(w)
    for i in h["infos"]:
        st.info(i)
    st.dataframe(pd.DataFrame(h["columns"]), **STRETCH, hide_index=True)
    st.caption("Original data is never modified; cleaning happens on an in-memory copy.")

# Charts
details = result["details"]
if px is None:
    st.info("Install plotly (`pip install plotly`) to see charts.")
else:
    c1, c2, c3 = st.columns(3)

    def style(fig):
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        return fig

    sales_df = pd.DataFrame(details["sales"])
    if not sales_df.empty:
        c1.plotly_chart(style(px.bar(sales_df, x="product_name", y="revenue", title="Revenue by product")),
                        **STRETCH, key="chart_rev")
    reasons_df = pd.DataFrame(details["returns"])
    if reasons_df.empty:
        c2.info("No returns to chart.")
    else:
        reasons_df["return_reason"] = reasons_df["return_reason"].replace("", "Not recorded")
        fig = px.bar(reasons_df, x="returns", y="return_reason", orientation="h", title="Returns by reason")
        fig.update_yaxes(categoryorder="total ascending")
        c2.plotly_chart(style(fig), **STRETCH, key="chart_reasons")
    prod_df = pd.DataFrame(details["product"])
    if not prod_df.empty:
        c3.plotly_chart(style(px.bar(prod_df, x="product_name", y="return_rate", title="Return rate by product (%)")),
                        **STRETCH, key="chart_return_rate")

st.divider()
st.subheader("🤖 Five Specialized Agents")

agent_cols = st.columns(5)
for col, agent in zip(agent_cols, result["agents"]):
    with col:
        st.markdown(f"### {agent['icon']} {agent['name']}")
        st.success("Analysis complete")
        for item in agent["highlights"][:4]:
            st.write("• " + item)

st.divider()
st.subheader("🧠 Orchestrator — Cross-Agent Reasoning")

orch = result["orchestrator"]
verification = None
if groq_key:
    # Cache per key + data so Streamlit reruns don't call the API again every time.
    cache = st.session_state.setdefault("ai_cache", {})
    ck = hashlib.sha1((groq_key + json.dumps(build_compact_payload(result), default=str)).encode()).hexdigest()
    if ck not in cache:
        with st.spinner("Generating AI reasoning..."):
            try:
                cache[ck] = ("ok", generate_ai_insights(result, groq_key))
            except Exception as e:
                cache[ck] = ("error", str(e))
    status, value = cache[ck]
    if status == "ok":
        orch = value
        verification = value.get("verification")
        st.success("AI-enhanced reasoning generated.")
    else:
        st.warning(f"AI reasoning unavailable; showing analytical reasoning instead. ({value})")

for item in orch.get("actions", []):
    with st.container(border=True):
        st.markdown(f"**Priority:** {item['priority']}")
        st.markdown(f"**Problem:** {item['problem']}")
        st.markdown(f"**Evidence:** {item['evidence']}")
        st.markdown(f"**Financial Impact:** {item['financial_impact']}")
        st.markdown(f"**Suggested Action:** {item['suggested_action']}")

if verification and verification["checked"]:
    if not verification["unmatched"]:
        st.caption(f"✅ Number check: all {verification['checked']} figures in this AI reasoning match values computed by Python.")
    else:
        st.warning("⚠️ Number check: these figures in the AI reasoning could not be matched to computed data — "
                   f"verify before relying on them: {', '.join(verification['unmatched'])}")

st.divider()
st.subheader("🔎 Investigation — why it happened & what to do next")
st.caption("All numbers below are computed with Pandas. Causes are hypotheses based on patterns in your data, tagged by evidence strength — association, not proof.")

SEV_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
findings = result["investigation"]
if not findings:
    st.info("Nothing to investigate yet — no returns, margin gaps or data problems were detected.")
for i, f in enumerate(findings):
    with st.expander(f"{SEV_ICON.get(f['severity'], '⚪')} {f['title']}  ·  {f['severity']} priority", expanded=(i == 0)):
        st.caption(f"Confidence: **{f['confidence']}** — {f['confidence_note']}")
        st.markdown("**What we observed**")
        for o in f["observed"]:
            st.write("• " + o)
        for t in f["tables"]:
            st.markdown(f"**{t['title']}**")
            st.dataframe(pd.DataFrame(t["rows"]), **STRETCH, hide_index=True)
        st.markdown("**Why it may be happening** (hypotheses, not proof)")
        for c in f["possible_causes"]:
            st.markdown(f"- **[{c['strength']}]** {c['cause']}  \n  *Basis: {c['basis']}*")
        st.markdown("**What to do next**")
        for n, step in enumerate(f["next_steps"], 1):
            st.write(f"{n}. {step}")
        if f["data_gaps"]:
            st.info("Data that would sharpen this: " + "; ".join(f["data_gaps"]))

st.divider()

tabs = st.tabs(["Sales", "Product", "Customer", "Returns", "Financial", "Data Preview"])
for tab, key in zip(tabs, ["sales", "product", "customer", "returns", "financial"]):
    with tab:
        st.dataframe(pd.DataFrame(details[key]), **STRETCH)

with tabs[-1]:
    st.dataframe(df.head(100), **STRETCH)
