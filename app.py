
# GigGuard AI — Streamlit App
# Author: Haripriya V
# Built for: Capstone Project — Imarticus Learning, Chennai
# Run: streamlit run app.py

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import joblib
import os

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="GigGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS for better UI
# ---------------------------------------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #B71C1C;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .risk-card {
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #B71C1C;
        margin-bottom: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Saved Models
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        m1 = joblib.load('model1_burnout.pkl')
        s1 = joblib.load('scaler1.pkl')
        m2 = joblib.load('model2_attrition.pkl')
        s2 = joblib.load('scaler2.pkl')
        m3 = joblib.load('model3_freelancer.pkl')
        s3 = joblib.load('scaler3.pkl')
        return m1, s1, m2, s2, m3, s3, True
    except Exception as e:
        return None, None, None, None, None, None, False

m1, s1, m2, s2, m3, s3, models_loaded = load_models()

# ---------------------------------------------------------
# Advisory Generator
# ---------------------------------------------------------
def generate_advisory(b_pred, a_pred, f_pred, fatigue, hours, success):
    high_count = [b_pred, a_pred, f_pred].count(2)
    med_count  = [b_pred, a_pred, f_pred].count(1)

    risk_score = min(int(((b_pred+a_pred+f_pred)/6)*50 + fatigue*3 + max(0,hours-8)*2), 100)
    health_score = 100 - risk_score

    if high_count >= 2 or (high_count == 1 and fatigue >= 8):
        level   = 'CRITICAL'
        summary = f'🚨 CRITICAL ALERT — Health Score: {health_score}/100. Take immediate action!'
    elif high_count == 1 or med_count >= 2 or fatigue >= 7:
        level   = 'WARNING'
        summary = f'⚠️ WARNING — Health Score: {health_score}/100. Risk signals building up.'
    else:
        level   = 'SAFE'
        summary = f'✅ YOU ARE SAFE — Health Score: {health_score}/100. Keep this up!'

    actions = []
    if b_pred == 2: actions.append('🔴 Take 2 rest days immediately. No shifts for 48 hours.')
    elif b_pred == 1: actions.append('🟡 Limit shifts to 8 hours max for next 5 days.')
    if a_pred == 2: actions.append('🔴 Register on 2 more platforms this week to reduce income risk.')
    elif a_pred == 1: actions.append('🟡 Explore a backup platform like Urban Company or Dunzo.')
    if f_pred == 2: actions.append('🔴 Accept fewer orders — focus on quality over quantity.')
    elif f_pred == 1: actions.append('🟡 Decline jobs you cannot complete well. 5 good > 10 poor.')
    if fatigue >= 8: actions.append('😴 Sleep 8+ hours minimum. Stop all late-night shifts.')
    elif fatigue >= 6: actions.append('😴 Add 30-minute break every 4 hours during your working day.')
    if hours >= 12: actions.append('⏱️ 12+ hours daily is unsustainable. Reduce to 10 hours max.')
    if success < 60: actions.append('⭐ Success below 60%. Focus on completing accepted jobs on time.')
    actions.append('💡 GigGuard Tip: Track earnings weekly. If it drops 20%, contact platform support.')

    return level, summary, actions, health_score

# ---------------------------------------------------------
# Earning Collapse Predictor
# ---------------------------------------------------------
def predict_collapse(this_week, avg_4_weeks, success, rank_drop, hours_this_week):
    risk = 0.0
    reasons = []
    if avg_4_weeks > 0:
        ratio = this_week / avg_4_weeks
        if ratio < 0.75:
            risk += 0.35
            reasons.append(f'Earnings dropped {((1-ratio)*100):.0f}% below your 4-week average.')
        elif ratio < 0.90:
            risk += 0.20
            reasons.append(f'Earnings are slightly below your 4-week average.')
    if success < 60:
        risk += 0.30
        reasons.append(f'Success rate {success}% — platform may reduce your job allocation.')
    elif success < 75:
        risk += 0.15
        reasons.append(f'Success rate {success}% — just above warning threshold.')
    if rank_drop:
        risk += 0.25
        reasons.append('Platform ranking dropped — fewer jobs assigned to you.')
    if hours_this_week > 70:
        risk += 0.15
        reasons.append('Worked 70+ hours. Pace may drop next week.')
    prob = min(risk, 1.0)
    forecast = this_week * (0.65 if prob >= 0.6 else 0.85 if prob >= 0.35 else 1.02)
    label = 'HIGH' if prob >= 0.6 else 'MEDIUM' if prob >= 0.35 else 'LOW'
    return label, prob, ' '.join(reasons) or 'Earnings look stable.', forecast

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🛡️ GigGuard AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Burnout Detection & Earning Collapse Prediction for Indian Gig Workers</div>', unsafe_allow_html=True)

if not models_loaded:
    st.error("⚠️ Models not loaded. Make sure model1_burnout.pkl, model2_attrition.pkl, model3_freelancer.pkl are in the same folder as app.py")
    st.stop()

# ---------------------------------------------------------
# Mode Selector
# ---------------------------------------------------------
st.sidebar.markdown("## 🔍 Select Mode")
mode = st.sidebar.radio(
    "Who is using GigGuard?",
    options=["👷 Worker Risk Dashboard", "💼 Business Intelligence Dashboard"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**GigGuard AI v2**

🛡️ Protecting India's 7.7M gig workers
""")

# ==========================================================
# MODE 1: WORKER RISK DASHBOARD
# ==========================================================
if mode == "👷 Worker Risk Dashboard":

    st.markdown("### 📝 Enter Your Work Details")
    st.markdown("Fill in your details below to get your personalized burnout risk and advisory.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Your Work Profile**")
        fatigue     = st.slider("Mental Fatigue (1=Refresh, 10=Exhausted)", 1, 10, 5)
        hours       = st.slider("Hours Worked Per Day", 1, 18, 8)
        tenure      = st.number_input("Days on Platform", min_value=1, max_value=3650, value=180)
        wfh         = st.selectbox("WFH Available?", ["Yes", "No"])
        wfh_enc     = 1 if wfh == "Yes" else 0

    with col2:
        st.markdown("**AI & Automation Exposure**")
        ai_tools        = st.slider("AI Tools Used on Job (0-10)", 0, 10, 3)
        productivity    = st.slider("Productivity Score (1-10)", 1, 10, 6)
        burnout_self    = st.slider("Self-Reported Burnout Level (1-10)", 1, 10, 5)
        satisfaction    = st.slider("Job Satisfaction (1-10)", 1, 10, 6)
        fear_ai         = st.selectbox("Worried about AI replacing you?", ["Low","Medium","High"])
        fear_enc        = {"Low":0,"Medium":1,"High":2}[fear_ai]

    with col3:
        st.markdown("**Platform Performance**")
        platform      = st.selectbox("Platform", ["Fiverr","Freelancer","Toptal","Upwork"])
        exp_level     = st.selectbox("Experience Level", ["Entry","Mid","Senior","Expert"])
        jobs_done     = st.number_input("Jobs Completed", 0, 5000, 50)
        hourly_rate   = st.number_input("Hourly Rate (₹)", 0, 5000, 300)
        job_success   = st.slider("Job Success Rate (%)", 0, 100, 78)
        client_rate   = st.slider("Client Rating (1-5)", 1.0, 5.0, 4.0, step=0.1)

    st.markdown("---")

    if st.button("🚀 Analyze My Risk Now", type="primary", use_container_width=True):

        # Model 1 prediction
        n1 = s1.n_features_in_
        X1_base = [0, wfh_enc, 0, 7, fatigue, tenure]
        if len(X1_base) < n1: X1_base = X1_base + [0.0] * (n1 - len(X1_base))
        elif len(X1_base) > n1: X1_base = X1_base[:n1]
        b_pred = m1.predict(s1.transform(np.array([X1_base])))[0]

        # Model 2 prediction
        n2 = s2.n_features_in_
        X2_base = [0, tenure, 0, 0, 0, 0, 50, ai_tools, hours*5, 30, 5.0,
                   productivity, burnout_self, satisfaction, fear_enc]
        if len(X2_base) < n2: X2_base = X2_base + [0.0] * (n2 - len(X2_base))
        elif len(X2_base) > n2: X2_base = X2_base[:n2]
        a_pred = m2.predict(s2.transform(np.array([X2_base])))[0]

        # Model 3 prediction
        # Build input padded to exactly s3.n_features_in_ features
        # This avoids the ValueError: X has 8 features but scaler expects 14
        p_enc = ["Fiverr","Freelancer","Toptal","Upwork"].index(platform)
        e_enc = ["Entry","Mid","Senior","Expert"].index(exp_level)
        n3 = s3.n_features_in_
        X3_base = [p_enc, e_enc, jobs_done, hourly_rate, job_success, client_rate, 14, 0.8]
        if len(X3_base) < n3:
            X3_base = X3_base + [0.0] * (n3 - len(X3_base))
        elif len(X3_base) > n3:
            X3_base = X3_base[:n3]
        X3_input = np.array([X3_base])
        f_pred = m3.predict(s3.transform(X3_input))[0]

        label_map = {0:"Low", 1:"Medium", 2:"High"}
        level, summary, actions, health_score = generate_advisory(b_pred, a_pred, f_pred, fatigue, hours, job_success)

        # Alert Banner
        if level == 'CRITICAL':
            st.error(summary)
        elif level == 'WARNING':
            st.warning(summary)
        else:
            st.success(summary)

        st.markdown("---")

        # Risk Cards
        st.markdown("### 🔎 Your Risk Report")
        rc1, rc2, rc3, rc4 = st.columns(4)
        color_map = {"Low": "normal", "Medium": "inverse", "High": "off"}
        rc1.metric("🏠 Burnout Risk",    label_map[b_pred])
        rc2.metric("💼 Attrition Risk",  label_map[a_pred])
        rc3.metric("⭐ Freelance Risk",  label_map[f_pred])
        rc4.metric("❤️ Health Score",    f"{health_score}/100")

        st.markdown("---")

        # Charts
        left, right = st.columns(2)
        with left:
            st.markdown("### 📊 Overall Risk Gauge")
            risk_score_num = 100 - health_score
            gauge_color = "#EF5350" if risk_score_num >= 65 else "#FFA726" if risk_score_num >= 35 else "#66BB6A"
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score_num,
                title={"text": "Risk Score", "font": {"size": 16}},
                delta={"reference": 50, "increasing": {"color": "#EF5350"}, "decreasing": {"color": "#66BB6A"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":  {"color": gauge_color, "thickness": 0.3},
                    "steps": [
                        {"range": [0, 35],   "color": "#e8f5e9"},
                        {"range": [35, 65],  "color": "#fff8e1"},
                        {"range": [65, 100], "color": "#ffebee"}
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "value": 65}
                }
            ))
            fig_g.update_layout(height=300, margin=dict(t=50, b=0, l=30, r=30))
            st.plotly_chart(fig_g, use_container_width=True)

        with right:
            st.markdown("### 📡 Risk Radar Chart")
            cats = ["Burnout", "Fatigue", "Workload", "Attrition", "Job Quality", "Freelance"]
            vals = [
                b_pred * 33,
                fatigue * 10,
                min(hours * 7, 100),
                a_pred * 33,
                max(0, 100 - job_success),
                f_pred * 33
            ]
            avg_vals = [33, 45, 56, 33, 22, 33]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]],
                            fill="toself", name="Your Profile", line_color="#EF5350"))
            fig_r.add_trace(go.Scatterpolar(r=avg_vals + [avg_vals[0]], theta=cats + [cats[0]],
                            fill="toself", name="Avg Worker",
                            line=dict(dash="dot", color="#42A5F5")))
            fig_r.update_layout(height=300, margin=dict(t=30, b=30, l=30, r=30),
                                polar=dict(radialaxis=dict(range=[0, 100])))
            st.plotly_chart(fig_r, use_container_width=True)

        st.markdown("---")

        # Earning Collapse Predictor
        st.markdown("### 💰 Earning Collapse Prediction")
        ec1, ec2 = st.columns(2)
        with ec1:
            this_week_earn = st.number_input("This Week's Earnings (₹)", 0, 50000, 4000)
            avg_4w_earn    = st.number_input("Average Earnings Last 4 Weeks (₹)", 0, 50000, 5000)
        with ec2:
            rank_dropped   = st.checkbox("Did your platform ranking drop this week?")
            hrs_this_week  = st.number_input("Total Hours This Week", 0, 120, 48)

        if st.button("📈 Predict Next Week's Earnings"):
            collapse_risk, prob, reason, forecast = predict_collapse(
                this_week_earn, avg_4w_earn, job_success, rank_dropped, hrs_this_week
            )
            color = "error" if collapse_risk == 'HIGH' else "warning" if collapse_risk == 'MEDIUM' else "success"
            getattr(st, color)(
                f"**Earning Collapse Risk: {collapse_risk}** ({prob*100:.0f}% probability) | "
                f"Next Week Forecast: ₹{forecast:,.0f}"
            )
            if reason:
                st.info(f"📋 Reason: {reason}")

        st.markdown("---")

        # Advisory
        st.markdown("### 🤖 Your Personalized Advisory (AI Assistant)")
        st.markdown("*GigGuard acts as your personal HR manager — here's what it recommends for you:*")
        for i, action in enumerate(actions, 1):
            st.info(f"**Step {i}:** {action}")

        st.markdown("---")
        st.caption("GigGuard AI v2 | Advisory tool only — not medical advice.")

# ==========================================================
# MODE 2: BUSINESS INTELLIGENCE DASHBOARD
# ==========================================================
else:
    st.markdown("### 💼 Platform Business Intelligence Dashboard")
    st.markdown("*This view is designed for Swiggy, Zomato, Ola, Upwork, and other gig platforms*")

    st.markdown("#### Enter Platform Workforce Parameters")
    b1, b2, b3 = st.columns(3)
    with b1:
        total_workers     = st.number_input("Total Active Workers", 100, 10000000, 50000)
        dropout_rate      = st.slider("Current Annual Dropout Rate (%)", 0, 100, 35)
    with b2:
        onboarding_cost   = st.number_input("Onboarding Cost per Worker (₹)", 0, 50000, 8000)
        avg_weekly_earn   = st.number_input("Avg Weekly Earnings per Worker (₹)", 0, 50000, 4500)
    with b3:
        high_risk_pct     = st.slider("Estimated High-Risk Workers (%)", 0, 100, 20)
        gigguard_catch    = st.slider("GigGuard Prevention Efficiency (%)", 0, 100, 40)

    if st.button("📊 Generate Business Report", type="primary", use_container_width=True):

        dropouts     = int(total_workers * dropout_rate / 100)
        total_cost   = dropouts * onboarding_cost
        prevented    = int(dropouts * gigguard_catch / 100)
        saved        = prevented * onboarding_cost
        high_risk_workers = int(total_workers * high_risk_pct / 100)
        productivity_loss = int(high_risk_workers * 0.25 * avg_weekly_earn * 52)

        st.markdown("---")
        st.markdown("### 📈 Business Impact Report")

        m1c, m2c, m3c, m4c = st.columns(4)
        m1c.metric("Annual Dropouts",      f"{dropouts:,}",  delta=f"-{prevented:,} with GigGuard", delta_color="inverse")
        m2c.metric("Replacement Cost",     f"₹{total_cost/1e7:.1f} Cr")
        m3c.metric("Cost Saved",           f"₹{saved/1e7:.1f} Cr",   delta="with GigGuard", delta_color="normal")
        m4c.metric("High-Risk Workers",    f"{high_risk_workers:,}",  help="Workers who need immediate attention")

        st.markdown("---")

        # ROI Chart
        fig_roi = go.Figure(data=[
            go.Bar(name='Without GigGuard', x=['Replacement Cost', 'Productivity Loss'],
                   y=[total_cost/1e7, productivity_loss/1e7], marker_color='#EF5350'),
            go.Bar(name='With GigGuard',    x=['Replacement Cost', 'Productivity Loss'],
                   y=[(total_cost-saved)/1e7, productivity_loss*0.6/1e7], marker_color='#66BB6A'),
        ])
        fig_roi.update_layout(
            title='ROI Impact: Cost Without vs With GigGuard (₹ Crore)',
            barmode='group', height=400,
            yaxis_title='Amount (₹ Crore)'
        )
        st.plotly_chart(fig_roi, use_container_width=True)

        # Risk Distribution Donut Chart
        low_pct  = 100 - high_risk_pct - min(35, 100-high_risk_pct)
        mid_pct  = min(35, 100-high_risk_pct)
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Low Risk', 'Medium Risk', 'High Risk'],
            values=[max(low_pct,0), mid_pct, high_risk_pct],
            hole=0.4,
            marker_colors=['#66BB6A', '#FFA726', '#EF5350']
        )])
        fig_pie.update_layout(
            title=f'Predicted Workforce Risk Distribution (Total: {total_workers:,} workers)',
            height=350
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Recommendations for Platform
        st.markdown("### 💡 GigGuard Recommendations for Your Platform")
        st.success(f"✅ Deploy GigGuard weekly alerts to your top {high_risk_workers:,} high-risk workers.")
        st.warning(f"⚠️ {dropouts:,} workers are expected to drop out this year — GigGuard can prevent {prevented:,} of them.")
        st.info(f"💰 Estimated savings with GigGuard: **₹{saved:,}** per year in onboarding costs alone.")
        st.info(f"📊 Additional productivity recovery potential: **₹{int(productivity_loss*0.4):,}** per year.")

        st.markdown("---")
        st.caption("GigGuard AI v2 | Business Intelligence Module | For platform partnerships, contact: Haripriya V — Imarticus Learning Chennai")
