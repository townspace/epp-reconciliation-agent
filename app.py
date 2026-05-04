"""
EPP Reconciliation AI Agent — Streamlit Web Interface
Run with: .venv/bin/streamlit run app.py
"""
import io
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EPP Reconciliation Agent",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stProgress > div > div { background-color: #1f77b4; }
.upload-box {
    border: 2px dashed #aaa;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    background: #fafafa;
}
</style>
""", unsafe_allow_html=True)

RULE_COLORS = {
    "R1_narr_exact_date_exact":  "#2ecc71",
    "R2_narr_fuzzy_date_exact":  "#27ae60",
    "R3_narr_exact_date_range":  "#f1c40f",
    "R4_narr_fuzzy_date_range":  "#f39c12",
    "R5_date_exact_only":        "#e67e22",
    "R6_amount_only":            "#e74c3c",
    "R7_reversal_detection":     "#c0392b",
    "R8_many_to_many_flag":      "#8e44ad",
}

RULE_LABELS = {
    "R1_narr_exact_date_exact":  "R1 · Exact narration + date",
    "R2_narr_fuzzy_date_exact":  "R2 · Fuzzy narration + exact date",
    "R3_narr_exact_date_range":  "R3 · Exact narration + date range",
    "R4_narr_fuzzy_date_range":  "R4 · Fuzzy narration + date range",
    "R5_date_exact_only":        "R5 · Date exact only",
    "R6_amount_only":            "R6 · Amount only",
    "R7_reversal_detection":     "R7 · Reversal detection",
    "R8_many_to_many_flag":      "R8 · Many-to-many (manual review)",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔄 EPP Reconciliation\nAI Agent")
    st.divider()

    st.subheader("⚙️ Settings")
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-proj-...",
        help="Your OpenAI API key. Leave blank to run without AI.",
    )
    skip_ai = not bool(openai_key)
    if skip_ai:
        st.info("No API key → running without AI")
    else:
        st.success("AI enabled (GPT-4o)")

    st.divider()
    st.subheader("🔧 Tolerances")
    amount_tol  = st.number_input("Amount tolerance", value=0.01, step=0.01, format="%.2f")
    date_tol    = st.number_input("Date tolerance (days)", value=3, min_value=0, max_value=30)
    fuzzy_tol   = st.slider("Fuzzy narration threshold", 50, 100, 85)

    st.divider()
    st.caption("Built with Claude · Powered by GPT-4o")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🔄 EPP Intercompany Reconciliation")
st.markdown("Upload your **ledger** and **bank statement** files — the agent matches them automatically.")

# ── File upload — two columns ─────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 📒 Ledger")
    ledger_file = st.file_uploader(
        "Upload ledger file",
        type=["xlsx", "xls", "csv"],
        key="ledger",
        label_visibility="collapsed",
    )
    if ledger_file:
        st.success(f"✅ {ledger_file.name}  ({ledger_file.size / 1024:.1f} KB)")

with col_b:
    st.markdown("### 🏦 Bank Statement")
    bank_file = st.file_uploader(
        "Upload bank statement file",
        type=["xlsx", "xls", "csv"],
        key="bank",
        label_visibility="collapsed",
    )
    if bank_file:
        st.success(f"✅ {bank_file.name}  ({bank_file.size / 1024:.1f} KB)")

# ── Preview uploaded files ────────────────────────────────────────────────────
if ledger_file or bank_file:
    with st.expander("👀 Preview uploaded files", expanded=False):
        p1, p2 = st.columns(2)
        if ledger_file:
            with p1:
                st.markdown("**Ledger (first 5 rows)**")
                try:
                    preview = pd.read_excel(ledger_file) if not ledger_file.name.endswith(".csv") else pd.read_csv(ledger_file)
                    st.dataframe(preview.head(), use_container_width=True)
                    ledger_file.seek(0)
                except Exception as e:
                    st.error(f"Preview error: {e}")
        if bank_file:
            with p2:
                st.markdown("**Bank Statement (first 5 rows)**")
                try:
                    preview = pd.read_excel(bank_file) if not bank_file.name.endswith(".csv") else pd.read_csv(bank_file)
                    st.dataframe(preview.head(), use_container_width=True)
                    bank_file.seek(0)
                except Exception as e:
                    st.error(f"Preview error: {e}")

st.divider()

# ── Column mapping (optional override) ───────────────────────────────────────
with st.expander("🗂 Column mapping (optional — use if columns aren't auto-detected)", expanded=False):
    st.markdown("The agent auto-detects common column names. Override here if needed.")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        col_entity   = st.text_input("Entity column name",          placeholder="e.g. Company")
        col_partner  = st.text_input("Partner Entity column name",  placeholder="e.g. Counterparty")
    with mc2:
        col_date     = st.text_input("Date column name",            placeholder="e.g. Value Date")
        col_amount   = st.text_input("Amount column name",          placeholder="e.g. Net Amount")
    with mc3:
        col_currency = st.text_input("Currency column name",        placeholder="e.g. CCY")
        col_narr     = st.text_input("Narration column name",       placeholder="e.g. Description")

    custom_cols = {k: v for k, v in {
        "Entity": col_entity, "PartnerEntity": col_partner,
        "TransactionDate": col_date, "Amount": col_amount,
        "Currency": col_currency, "Narration": col_narr,
    }.items() if v.strip()}

# ── Run button ────────────────────────────────────────────────────────────────
both_uploaded = ledger_file and bank_file
if not both_uploaded:
    st.info("👆 Upload both files to enable reconciliation")

run_btn = st.button(
    "▶ Run Reconciliation",
    type="primary",
    use_container_width=True,
    disabled=not both_uploaded,
)

if run_btn and both_uploaded:
    # Save both files to temp paths
    tmp_files = []
    for f in [ledger_file, bank_file]:
        suffix = Path(f.name).suffix
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(f.read())
        tmp.close()
        tmp_files.append(tmp.name)
    ledger_tmp, bank_tmp = tmp_files

    try:
        from src.config import Config
        from src.data.loader import load_two_files
        from src.data.cleaner import normalize_amounts, normalize_dates, normalize_narration
        from src.data.validator import validate
        from src.engine.orchestrator import run_reconciliation
        from src.output.exporter import export_results

        cfg = Config(
            openai_api_key=openai_key or "",
            skip_ai=skip_ai,
            amount_tolerance=amount_tol,
            date_tolerance_days=date_tol,
            fuzzy_narration_threshold=fuzzy_tol,
        )

        progress = st.progress(0, text="Loading files…")
        status   = st.empty()

        # Step 1: Load
        status.info("📂 Loading ledger and bank statement…")
        df1, df2 = load_two_files(ledger_tmp, bank_tmp)

        # Apply any custom column renames
        if custom_cols:
            inv = {v: k for k, v in custom_cols.items()}
            df1 = df1.rename(columns=inv)
            df2 = df2.rename(columns=inv)

        progress.progress(15, text="Cleaning data…")

        # Step 2: Clean
        status.info("🧹 Cleaning and normalising…")
        for fn in [normalize_amounts, normalize_dates, normalize_narration]:
            df1 = fn(df1)
            df2 = fn(df2)
        progress.progress(30, text="Validating…")

        # Step 3: Validate
        status.info("✅ Validating data quality…")
        val = validate(pd.concat([df1, df2], ignore_index=True))
        if val.duplicates:
            st.warning(f"⚠️ {len(val.duplicates)} duplicate row(s) found — continuing")
        for ci in val.currency_issues:
            st.warning(f"⚠️ Currency mismatch: {ci}")
        progress.progress(45, text="Running matching engine…")

        # Step 4: Match
        status.info("⚙️ Running 8-rule matching engine…")
        report = run_reconciliation(df1, df2, cfg)
        df_all = pd.concat([df1.assign(side="A"), df2.assign(side="B")], ignore_index=True)
        progress.progress(65, text="Running AI analysis…")

        # Step 5: AI
        if not skip_ai:
            from src.agent.agent import analyze_unmatched, score_complex_match, generate_summary

            status.info("🤖 AI: classifying unmatched transactions…")
            unmatched_tx = [
                {**df_all.loc[idx].to_dict(), "index": idx}
                for idx in report.unmatched_transactions
                if idx in df_all.index
            ]
            report.anomaly_analysis = analyze_unmatched(unmatched_tx, cfg)
            progress.progress(75, text="AI: scoring complex groups…")

            status.info("🤖 AI: scoring complex match groups…")
            report.complex_judgments = [score_complex_match(g, df_all, cfg) for g in report.complex_groups]
            progress.progress(85, text="AI: generating executive summary…")

            status.info("🤖 AI: writing executive summary…")
            report.executive_summary = generate_summary(report, cfg)
        else:
            from src.agent.agent import _fallback_summary
            report.executive_summary = _fallback_summary(report)

        progress.progress(90, text="Exporting results…")

        # Step 6: Export
        status.info("💾 Exporting results…")
        out_dir = str(Path(ledger_tmp).parent)
        xlsx_path, json_path = export_results(df_all, report, out_dir, cfg)
        progress.progress(100, text="Done!")
        status.success("✅ Reconciliation complete!")

        st.session_state.update({
            "report": report, "df_all": df_all,
            "xlsx_path": xlsx_path, "json_path": json_path,
        })

    except Exception as e:
        st.error(f"❌ {e}")
        import traceback
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())
    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except Exception:
                pass


# ── Results ───────────────────────────────────────────────────────────────────
if "report" in st.session_state:
    report    = st.session_state["report"]
    df_all    = st.session_state["df_all"]
    xlsx_path = st.session_state["xlsx_path"]
    json_path = st.session_state["json_path"]

    st.divider()

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions", report.total_transactions)
    c2.metric("Matched Pairs",      len(report.matched_pairs))
    c3.metric("Complex Groups (R8)",len(report.complex_groups))
    c4.metric("Unmatched",          len(report.unmatched_transactions))
    c5.metric("Match Rate",         f"{report.match_rate_pct:.1f}%",
              delta="above threshold" if report.match_rate_pct >= 70 else "below 70%",
              delta_color="normal" if report.match_rate_pct >= 70 else "inverse")

    st.divider()

    # Charts
    cl, cr = st.columns(2)
    with cl:
        st.subheader("📊 Matches by Rule")
        rule_counts = Counter(m.rule.value for m in report.matched_pairs)
        if rule_counts:
            try:
                import plotly.express as px
                df_r = pd.DataFrame([
                    {"Rule": RULE_LABELS.get(r, r), "Count": c}
                    for r, c in sorted(rule_counts.items())
                ])
                fig = px.bar(df_r, x="Count", y="Rule", orientation="h", height=320,
                             color="Rule",
                             color_discrete_sequence=list(RULE_COLORS.values()))
                fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(pd.Series(rule_counts))
        else:
            st.info("No matches found")

    with cr:
        st.subheader("🥧 Transaction Breakdown")
        matched_n   = sum(len(m.left_indices)+len(m.right_indices) for m in report.matched_pairs)
        r8_n        = sum(len(g.left_indices)+len(g.right_indices) for g in report.complex_groups)
        unmatched_n = len(report.unmatched_transactions)
        try:
            import plotly.express as px
            df_pie = pd.DataFrame({
                "Category": ["Matched (R1–R7)", "Complex / R8", "Unmatched"],
                "Count":    [matched_n, r8_n, unmatched_n],
            })
            fig = px.pie(df_pie, names="Category", values="Count", height=320,
                         color_discrete_sequence=["#2ecc71","#8e44ad","#e74c3c"])
            fig.update_layout(margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.dataframe(df_pie)

    st.divider()

    # Detail tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        f"✅ Matched ({len(report.matched_pairs)})",
        f"⚠️ Manual Review / R8 ({len(report.complex_groups)})",
        f"❌ Unmatched ({len(report.unmatched_transactions)})",
        "📝 Executive Summary",
    ])

    with tab1:
        rows = []
        for m in report.matched_pairs:
            for li in m.left_indices:
                for ri in m.right_indices:
                    if li in df_all.index and ri in df_all.index:
                        ra, rb = df_all.loc[li], df_all.loc[ri]
                        rows.append({
                            "Rule":        m.rule.value.split("_",1)[0],
                            "Confidence":  f"{m.confidence_score:.0%}",
                            "Entity A":    ra.get("Entity",""),
                            "Partner A":   ra.get("PartnerEntity",""),
                            "Date A":      str(ra.get("TransactionDate",""))[:10],
                            "Amount A":    ra.get("Amount",""),
                            "Narration A": str(ra.get("narration_clean",""))[:50],
                            "Date B":      str(rb.get("TransactionDate",""))[:10],
                            "Amount B":    rb.get("Amount",""),
                            "Narration B": str(rb.get("narration_clean",""))[:50],
                        })
        st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame(), use_container_width=True, height=420)

    with tab2:
        judgment_map = {j.group_id: j for j in report.complex_judgments}
        if not report.complex_groups:
            st.info("No complex groups found")
        for grp in report.complex_groups:
            j = judgment_map.get(grp.recon_id)
            verdict = j.verdict if j else "PENDING"
            icon = {"LIKELY_VALID":"🟢","NEEDS_REVIEW":"🟡","LIKELY_ERROR":"🔴"}.get(verdict,"⚪")
            with st.expander(f"{icon} Group `{grp.recon_id[:8]}…` — {verdict}"):
                gc1, gc2 = st.columns(2)
                with gc1:
                    st.markdown("**Ledger side**")
                    left_rows = [df_all.loc[i].to_dict() for i in grp.left_indices if i in df_all.index]
                    st.dataframe(pd.DataFrame(left_rows)[["Entity","PartnerEntity","Amount","TransactionDate"]], use_container_width=True)
                with gc2:
                    st.markdown("**Bank side**")
                    right_rows = [df_all.loc[i].to_dict() for i in grp.right_indices if i in df_all.index]
                    st.dataframe(pd.DataFrame(right_rows)[["Entity","PartnerEntity","Amount","TransactionDate"]], use_container_width=True)
                if j:
                    st.markdown(f"**AI Reasoning:** {j.reasoning}")
                    st.progress(j.confidence, text=f"Confidence: {j.confidence:.0%}")

    with tab3:
        anomaly_map = {a.transaction_index: a for a in report.anomaly_analysis}
        rows = []
        for idx in report.unmatched_transactions:
            if idx not in df_all.index:
                continue
            row = df_all.loc[idx].to_dict()
            a   = anomaly_map.get(idx)
            rows.append({
                "Source":            row.get("dataset_source",""),
                "Entity":            row.get("Entity",""),
                "Partner":           row.get("PartnerEntity",""),
                "Date":              str(row.get("TransactionDate",""))[:10],
                "Amount":            row.get("Amount",""),
                "Currency":          row.get("Currency",""),
                "Narration":         str(row.get("narration_clean",""))[:60],
                "AI Classification": a.classification if a else "—",
                "AI Suggestion":     a.suggested_action if a else "—",
            })
        st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame(), use_container_width=True, height=420)

    with tab4:
        if report.executive_summary:
            st.markdown(report.executive_summary)
        else:
            st.info("No summary generated")

    st.divider()

    # Downloads
    st.subheader("⬇️ Download Results")
    d1, d2 = st.columns(2)
    with d1:
        with open(xlsx_path, "rb") as f:
            st.download_button(
                "📥 Download Excel Report",
                data=f.read(),
                file_name=f"reconciliation_{report.run_id[:8]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    with d2:
        with open(json_path, "rb") as f:
            st.download_button(
                "📥 Download Audit JSON",
                data=f.read(),
                file_name=f"audit_{report.run_id[:8]}.json",
                mime="application/json",
                use_container_width=True,
            )
