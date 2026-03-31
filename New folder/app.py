"""
Student Performance Analyser — Streamlit UI (Simplified)
Run: streamlit run app.py
"""

import json
import traceback
import os

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dotenv import load_dotenv
load_dotenv()

from agents import (
    Orchestrator,
    load_uci_student_data,
    plot_grade_distribution,
    plot_risk_factors,
    plot_grade_by_group,
    rank_risk_factors,
    AzureOpenAI,
)
import agents as _ag

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="Student Performance Analyser",
    page_icon="🎓",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────
st.sidebar.title("⚙️ Configuration")

azure_endpoint = st.sidebar.text_input(
    "Azure OpenAI Endpoint",
    value=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
)
azure_key = st.sidebar.text_input(
    "API Key",
    value=os.environ.get("AZURE_OPENAI_API_KEY", ""),
    type="password",
)
deployment = st.sidebar.text_input(
    "Deployment Name",
    value=os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4o"),
)
max_iter = st.sidebar.slider("Max iterations per agent", 2, 10, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Agent pipeline:**
1. DataLoader
2. SkillsAnalyst
3. TrendForecaster
4. ReportWriter
""")

# ── Main page ─────────────────────────────────
st.title("🎓 Student Performance Analyser")
st.caption("Multi-Agent GenAI Pipeline · Azure OpenAI · UCI Dataset")
st.markdown("---")

# ── Load dataset ──────────────────────────────
@st.cache_data(show_spinner=False)
def get_df():
    return load_uci_student_data()

df = get_df()

st.subheader("📊 Dataset Preview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", len(df.columns))
col3.metric("Schools", df["school"].nunique())
col4.metric("Avg Grade (G3)", f"{df['G3'].mean():.1f}")
st.dataframe(df.head(8), use_container_width=True)

st.markdown("---")

# ── Run button ────────────────────────────────
if st.button("🚀 Run Analysis Pipeline"):

    # Patch credentials
    _ag.AZURE_ENDPOINT   = azure_endpoint
    _ag.AZURE_API_KEY    = azure_key
    _ag.DEPLOYMENT_NAME  = deployment
    _ag.MAX_ITERATIONS   = max_iter
    _ag.client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_key,
        api_version=_ag.AZURE_API_VERSION,
    )

    progress = st.progress(0, text="Starting pipeline…")

    try:
        def _cb(fraction, label):
            progress.progress(fraction, text=label)

        orchestrator = Orchestrator(df)
        report = orchestrator.run(progress_callback=_cb)
        progress.progress(1.0, text="✅ Pipeline complete!")

        # ── Always inject fallbacks ───────────────
        if not report.get("top_risk_factors"):
            report["top_risk_factors"] = rank_risk_factors(df, top_n=6)

        if not report.get("grade_by_sex"):
            stats = df.groupby("sex")["G3"].mean().reset_index()
            stats.columns = ["sex", "avg_grade"]
            report["grade_by_sex"] = stats.to_dict(orient="records")

        if not report.get("grade_by_address"):
            stats = df.groupby("address")["G3"].mean().reset_index()
            stats.columns = ["address", "avg_grade"]
            report["grade_by_address"] = stats.to_dict(orient="records")

        if not report.get("headline_stats"):
            report["headline_stats"] = {
                "mean_grade": round(float(df["G3"].mean()), 2),
                "fail_rate_pct": round((df["G3"] < 10).mean() * 100, 1),
            }

        if not report.get("dataset"):
            report["dataset"] = {
                "rows": len(df),
                "columns_count": len(df.columns),
            }

        # ── Executive summary ─────────────────────
        if report.get("executive_summary"):
            st.info(report["executive_summary"])

        # ── Metric cards ──────────────────────────
        st.markdown("---")
        st.subheader("📈 Headline Stats")
        hs = report["headline_stats"]
        ds = report["dataset"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Students",   f'{ds.get("rows", len(df)):,}')
        m2.metric("Mean Grade", f'{hs.get("mean_grade", round(float(df["G3"].mean()),2))} / 20')
        m3.metric("Fail Rate",  f'{hs.get("fail_rate_pct", round((df["G3"]<10).mean()*100,1))}%')
        m4.metric("Features",   f'{ds.get("columns_count", len(df.columns))}')

        # ── Tabs ──────────────────────────────────
        st.markdown("---")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Charts",
            "⚠️ Risk Factors",
            "🎯 Interventions",
            "📄 JSON Report",
            "🖥️ Agent Logs",
        ])

        # Charts
        with tab1:
            st.subheader("Grade Distribution")
            fig1 = plot_grade_distribution(df)
            st.pyplot(fig1, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("By Gender")
                fig2 = plot_grade_by_group(
                    report["grade_by_sex"], "sex", "Average Grade by Gender"
                )
                st.pyplot(fig2, use_container_width=True)
            with c2:
                st.subheader("By Location")
                fig3 = plot_grade_by_group(
                    report["grade_by_address"], "address", "Average Grade by Location"
                )
                st.pyplot(fig3, use_container_width=True)

            # Parental influence
            pi = report.get("parental_influence", {})
            if pi.get("by_Mjob") or pi.get("by_Fjob"):
                st.subheader("Parental Occupation Influence")
                c3, c4 = st.columns(2)
                with c3:
                    if pi.get("by_Mjob"):
                        fig4 = plot_grade_by_group(pi["by_Mjob"], "Mjob", "Mother's Job vs Grade")
                        st.pyplot(fig4, use_container_width=True)
                with c4:
                    if pi.get("by_Fjob"):
                        fig5 = plot_grade_by_group(pi["by_Fjob"], "Fjob", "Father's Job vs Grade")
                        st.pyplot(fig5, use_container_width=True)

        # Risk Factors
        with tab2:
            st.subheader("Top Failure Risk Factors")
            fig6 = plot_risk_factors(report["top_risk_factors"])
            st.pyplot(fig6, use_container_width=True)
            st.dataframe(
                pd.DataFrame(report["top_risk_factors"]),
                use_container_width=True,
                hide_index=True,
            )
            for insight in report.get("key_insights", []):
                st.info(f"💡 {insight}")

        # Interventions
        with tab3:
            st.subheader("Recommended Interventions")
            interventions = report.get("interventions", [])
            if interventions:
                for item in interventions:
                    priority = item.get("priority", "medium").lower()
                    color = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(priority, "⚪")
                    st.markdown(f"### {color} {item.get('priority','').upper()} PRIORITY")
                    st.markdown(f"**Action:** {item.get('action','')}")
                    st.markdown(f"**Rationale:** {item.get('rationale','')}")
                    st.markdown("---")
            else:
                st.warning("No interventions found in report.")

        # JSON Report
        with tab4:
            st.subheader("Full JSON Report")
            st.download_button(
                "⬇️ Download report.json",
                data=json.dumps(report, indent=2),
                file_name="student_performance_report.json",
                mime="application/json",
            )
            st.json(report)

        # Agent Logs
        with tab5:
            st.subheader("Agent Logs")
            for log in orchestrator.logs:
                if "WARNING" in log:
                    st.warning(log)
                elif "failed" in log.lower():
                    st.error(log)
                elif "Finished" in log or "complete" in log.lower():
                    st.success(log)
                else:
                    st.text(log)

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.code(traceback.format_exc())
