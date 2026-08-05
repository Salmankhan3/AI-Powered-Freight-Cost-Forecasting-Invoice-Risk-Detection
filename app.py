import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from inference.predict_freight_cost import freight_cost_prediction, load_model as load_model1
from inference.predict_invoice_flag import invoice_flaging, load_model as load_model2

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# SESSION STATE (keeps a running log of predictions for this session)
# ---------------------------------------------------------------------------
if "freight_history" not in st.session_state:
    st.session_state.freight_history = []
if "flag_history" not in st.session_state:
    st.session_state.flag_history = []

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("""
# 🚚 Vendor Invoice Intelligence Portal
### 🤖 AI-Driven Freight Cost Prediction & 🚩 Invoice Risk Flagging

Leverage machine learning to forecast freight costs, detect billing anomalies, and flag
high-risk vendor invoices before they impact your bottom line.
---
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("📊 **Cost Prediction**\n\nForecast freight charges using historical vendor data")
with col2:
    st.warning("🚩 **Risk Flagging**\n\nAuto-detect anomalies & overbilling patterns")
with col3:
    st.success("✅ **Audit Ready**\n\nExportable reports for compliance review")

st.divider()

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Model Selection")
selected_model = st.sidebar.radio(
    "Choose a prediction module",
    ["📦 Freight Cost Prediction", "🕵️ Invoice Manual Approval Flag"],
    help="Select which AI model you want to run against your invoice data."
)

st.sidebar.divider()
st.sidebar.markdown("### 💼 Business Impact")
st.sidebar.markdown(
    "- 📉 Improve freight cost forecasting accuracy\n"
    "- 🛡️ Reduce invoice fraud & billing anomalies\n"
    "- ⏱️ Speed up manual approval workflows\n"
    "- 📋 Strengthen audit & compliance readiness"
)

st.sidebar.divider()
with st.sidebar.expander("ℹ️ About this tool"):
    st.markdown(
        "This portal uses trained ML models to:\n"
        "1. **Predict** expected freight cost from invoice value\n"
        "2. **Flag** invoices that show abnormal cost, freight, or "
        "delivery patterns for manual review\n\n"
        "Results are session-based and not stored permanently."
    )

st.sidebar.caption(f"🕒 Session started: {datetime.now().strftime('%b %d, %Y — %H:%M')}")

# ---------------------------------------------------------------------------
# MODULE 1: FREIGHT COST PREDICTION
# ---------------------------------------------------------------------------
if selected_model == "📦 Freight Cost Prediction":
    st.subheader("📦 Freight Cost Prediction")
    st.markdown(
        "**Objective:** Predict the expected freight cost for a vendor invoice "
        "based on **Invoice Quantity** and **Invoice Dollar Amount**."
    )

    with st.expander("💡 How it works", expanded=False):
        st.markdown(
            "The model was trained on historical vendor invoices to learn the "
            "relationship between invoice value and freight charges. Enter the "
            "invoice details below to get an estimated freight cost."
        )

    with st.form("freight_form"):
        col1, col2 = st.columns(2)
        with col1:
            invoice_quantity = st.number_input(
                "📦 Quantity",
                value=120,
                min_value=1,
                help="Total number of units on the invoice."
            )
        with col2:
            invoice_dollar = st.number_input(
                "💵 Invoice Dollars",
                value=18500,
                min_value=1,
                help="Total dollar value of the invoice."
            )
        submit_freight = st.form_submit_button("🔮 Predict Freight Cost", use_container_width=True)

    if submit_freight:
        with st.spinner("Running freight cost model..."):
            try:
                model = load_model1()
                input_data = {"Dollars": [invoice_dollar]}
                prediction = freight_cost_prediction(model, input_data)
                predicted_value = prediction["prediction"][0]

                st.success("✅ Prediction completed successfully")

                m1, m2, m3 = st.columns(3)
                m1.metric("💰 Estimated Freight Cost", f"${predicted_value:,.2f}")
                m2.metric("📦 Quantity", f"{invoice_quantity:,}")
                m3.metric(
                    "📊 Freight % of Invoice",
                    f"{(predicted_value / invoice_dollar) * 100:.1f}%"
                )

                st.session_state.freight_history.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Quantity": invoice_quantity,
                    "Invoice $": invoice_dollar,
                    "Predicted Freight $": round(predicted_value, 2)
                })
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

    if st.session_state.freight_history:
        st.divider()
        st.markdown("#### 🧾 Session Prediction History")
        hist_df = pd.DataFrame(st.session_state.freight_history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download History as CSV",
            hist_df.to_csv(index=False),
            file_name="freight_prediction_history.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------------------------
# MODULE 2: INVOICE MANUAL APPROVAL FLAG
# ---------------------------------------------------------------------------
if selected_model == "🕵️ Invoice Manual Approval Flag":
    st.subheader("🕵️ Invoice Manual Approval Prediction")
    st.markdown(
        "**Objective:** Predict whether a vendor invoice should be flagged for "
        "**manual approval** based on abnormal cost, freight, or delivery patterns."
    )

    with st.expander("💡 How it works", expanded=False):
        st.markdown(
            "The model compares this invoice's line-item details against typical "
            "patterns learned from historical data. Invoices with unusual cost, "
            "freight, or quantity ratios are flagged for human review before payment."
        )

    with st.form("invoice_flag_form"):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            invoice_quantity = st.number_input(
                "📦 Quantity", value=10, min_value=1,
                help="Quantity on this specific invoice line."
            )
        with col2:
            invoice_dollar = st.number_input(
                "💵 Dollars", value=120, min_value=1,
                help="Dollar value of this invoice line."
            )
        with col3:
            Freight = st.number_input(
                "🚚 Freight", value=34, min_value=11,
                help="Freight charge billed for this invoice."
            )
        with col4:
            total_item_quantity = st.number_input(
                "📊 Total Quantity", value=5, min_value=1,
                help="Total quantity across all items on the order."
            )
        with col5:
            total_item_dollars = st.number_input(
                "💰 Total Item Dollars", value=40, min_value=13,
                help="Total dollar value across all items on the order."
            )
        submit_flag = st.form_submit_button("🔍 Evaluate Invoice Risk", use_container_width=True)

    if submit_flag:
        with st.spinner("Evaluating invoice risk..."):
            try:
                model = load_model2()
                input_data = {
                    "invoice_quantity": [invoice_quantity],
                    "invoice_dollar": [invoice_dollar],
                    "Freight": [Freight],
                    "total_item_quantity": [total_item_quantity],
                    "total_item_dollars": [total_item_dollars]
                }
                flag_prediction = invoice_flaging(input_data, model)
                is_flag = bool(flag_prediction[0])

                if is_flag:
                    st.error("🚩 **Invoice Requires Manual Approval**")
                    st.markdown(
                        "This invoice shows patterns that deviate from expected "
                        "cost/freight ratios and should be reviewed before payment."
                    )
                else:
                    st.success("✅ **Invoice is Safe for Auto-Approval**")
                    st.markdown("No significant anomalies detected in this invoice.")

                st.session_state.flag_history.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Quantity": invoice_quantity,
                    "Dollars": invoice_dollar,
                    "Freight": Freight,
                    "Total Qty": total_item_quantity,
                    "Total $": total_item_dollars,
                    "Result": "🚩 Manual Review" if is_flag else "✅ Auto-Approved"
                })
            except Exception as e:
                st.error(f"❌ Evaluation failed: {e}")

    if st.session_state.flag_history:
        st.divider()
        st.markdown("#### 🧾 Session Evaluation History")
        hist_df = pd.DataFrame(st.session_state.flag_history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download History as CSV",
            hist_df.to_csv(index=False),
            file_name="invoice_flag_history.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.divider()
st.caption("Vendor Invoice Intelligence Portal · Internal AI Tool · Predictions are model estimates, not guarantees.")