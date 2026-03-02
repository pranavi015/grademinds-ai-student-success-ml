import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GradeMinds AI · Learning Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@400;500;600;700;800&family=Instrument+Sans:wght@300;400;500&display=swap');

[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }

html, body, [class*="css"] {
    font-family: 'Instrument Sans', sans-serif;
    background-color: #0c0e14;
    color: #dde2ef;
}
h1, h2, h3 { font-family: 'Bricolage Grotesque', sans-serif !important; }

.main-content { padding: 12px 8px 48px 8px; }

.about-card {
    background: linear-gradient(135deg, #111827 0%, #141c2e 100%);
    border: 1px solid #1e3050;
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 32px;
}
.about-title { font-family:'Bricolage Grotesque',sans-serif; font-size:1.55rem; font-weight:700; color:#ffffff; margin-bottom:10px; }
.about-desc  { font-size:0.93rem; color:#8892a4; line-height:1.8; max-width:820px; }
.about-pills { display:flex; flex-wrap:wrap; gap:8px; margin-top:20px; }
.pill        { background:#1a2540; border:1px solid #2a3a5e; color:#7c9fff; padding:5px 14px; border-radius:999px; font-size:0.77rem; font-weight:500; }

.metric-card  { background:#13172060; border:1px solid #1e2438; border-radius:12px; padding:22px 18px; text-align:center; }
.metric-value { font-family:'Bricolage Grotesque',sans-serif; font-size:2.4rem; font-weight:800; color:#7c9fff; line-height:1; }
.metric-label { font-size:0.7rem; color:#5a6480; text-transform:uppercase; letter-spacing:1.4px; margin-top:7px; }

/* ── BADGES ── */
.badge-atrisk      { background:#2a0f0f; color:#f87171; border:1px solid #f8717160; padding:5px 14px; border-radius:999px; font-size:0.82rem; font-weight:600; }
.badge-improve     { background:#2d1a00; color:#fb923c; border:1px solid #fb923c60; padding:5px 14px; border-radius:999px; font-size:0.82rem; font-weight:600; }
.badge-avg         { background:#1a2040; color:#7c9fff; border:1px solid #7c9fff60; padding:5px 14px; border-radius:999px; font-size:0.82rem; font-weight:600; }
.badge-consistent  { background:#142030; color:#38bdf8; border:1px solid #38bdf860; padding:5px 14px; border-radius:999px; font-size:0.82rem; font-weight:600; }
.badge-under       { background:#1e1040; color:#a78bfa; border:1px solid #a78bfa60; padding:5px 14px; border-radius:999px; font-size:0.82rem; font-weight:600; }
.badge-best        { background:#1a1200; color:#fbbf24; border:1px solid #fbbf2460; padding:5px 14px; border-radius:999px; font-size:0.82rem; font-weight:600; }

.badge-pass  { background:#0a2a1a; color:#34d399; border:1px solid #34d39960; padding:5px 16px; border-radius:999px; font-size:0.88rem; font-weight:600; }
.badge-fail  { background:#2a0f0f; color:#f87171; border:1px solid #f8717160; padding:5px 16px; border-radius:999px; font-size:0.88rem; font-weight:600; }

.section-title { font-family:'Bricolage Grotesque',sans-serif; font-size:1.05rem; font-weight:700; color:#c8d6ff; margin-bottom:12px; }
.page-title    { font-family:'Bricolage Grotesque',sans-serif; font-size:1.85rem; font-weight:800; color:#ffffff; margin-bottom:6px; }
.page-subtitle { font-size:0.88rem; color:#5a6480; margin-bottom:28px; }

.rec-box { background:#0e1420; border-left:3px solid #6a8fff; border-radius:0 10px 10px 0; padding:18px 22px; margin-top:14px; font-size:0.9rem; line-height:1.8; color:#c8d6ff; }
.divider { border-top:1px solid #1e2438; margin:28px 0; }

.stButton > button { background:linear-gradient(135deg,#2d4eb5,#4f6fd4); color:white; border:none; border-radius:8px; padding:10px 30px; font-family:'Bricolage Grotesque',sans-serif; font-weight:600; font-size:0.92rem; transition:opacity 0.2s; }
.stButton > button:hover { opacity:0.85; }
[data-testid="stFileUploader"] { background:#13172060; border-radius:10px; border:1.5px dashed #2a3050; padding:8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TOP NAV
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Analytics Dashboard"

st.markdown("""
<div style='display:flex; align-items:center; justify-content:space-between;
     background:rgba(14,17,26,0.95); border-bottom:1px solid #1e2438;
     padding:12px 24px; margin:-1rem -1rem 28px -1rem;'>
    <div style='font-family:Bricolage Grotesque,sans-serif; font-size:1.2rem; font-weight:800; color:#fff;'>
        GradeMinds <span style="color:#6a8fff">AI</span>
    </div>
    <div style='font-size:0.7rem; color:#3a4260; letter-spacing:0.8px; text-transform:uppercase;'>
        Milestone 1 &nbsp;·&nbsp; ML Analytics
    </div>
</div>
""", unsafe_allow_html=True)

nav1, nav2, nav3 = st.columns([2, 2, 8])
with nav1:
    if st.button("Analytics Dashboard", use_container_width=True):
        st.session_state.page = "Analytics Dashboard"
        st.rerun()
with nav2:
    if st.button("Student Predictor", use_container_width=True):
        st.session_state.page = "Student Predictor"
        st.rerun()

st.markdown("<div class='main-content'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
@st.cache_resource
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "best_model.pkl")
    features_path = os.path.join(BASE_DIR, "feature_names.pkl")

    model = joblib.load(model_path)
    features = joblib.load(features_path)

    return model, features


def assign_cluster_label(row):
    score = row["Final_Exam_Score"]
    att   = row["Attendance_Rate"]
    hrs   = row["Study_Hours_per_Week"]

    # Underachiever: high effort + attendance but low scores
    if hrs >= 20 and att >= 75 and score < 55:
        return "Underachiever"

    # Best Student: exceptional across everything
    if score >= 80 and att >= 85 and hrs >= 20:
        return "Best Student"

    # Consistent Learner: steady effort + attendance, mid scores
    if hrs >= 15 and att >= 75 and 50 <= score < 75:
        return "Consistent Learner"

    # Score-based tiers for the rest
    total = score * 0.5 + att * 0.3 + hrs * 0.2
    if total >= 65:   return "Best Student"
    elif total >= 48: return "Average"
    elif total >= 32: return "Needs Improvement"
    else:             return "At-Risk"


BADGE_MAP = {
    "At-Risk":          "badge-atrisk",
    "Needs Improvement":"badge-improve",
    "Average":          "badge-avg",
    "Consistent Learner":"badge-consistent",
    "Underachiever":    "badge-under",
    "Best Student":     "badge-best",
}

CATEGORY_ORDER = [
    "Best Student", "Consistent Learner", "Average",
    "Underachiever", "Needs Improvement", "At-Risk"
]

CATEGORY_DESC = {
    "At-Risk":           "Low attendance, low scores, showing little engagement. Immediate attention required.",
    "Needs Improvement": "Making an effort but not yet meeting expectations. Needs a stronger push.",
    "Average":           "Performing adequately. Has clear potential to move higher with the right support.",
    "Consistent Learner":"Steady attendance and study habits with mid-range scores. Recognise their effort.",
    "Underachiever":     "High effort and attendance but scores are not reflecting the work. Needs a different teaching approach.",
    "Best Student":      "Exceptional across all indicators. Top of the class — consider for mentorship or advanced tracks.",
}


def get_recommendation(pred_label, cluster, study_hours, attendance, past_scores):
    recs = []
    if pred_label == "Fail":
        recs.append("This student is currently at risk of failing. Immediate intervention is recommended.")
    else:
        recs.append("This student is on track to pass. Continue monitoring their progress.")
    if attendance < 75:
        recs.append("Attendance is below 75%. Encourage regular class participation.")
    if study_hours < 15:
        recs.append("Weekly study hours appear low. Recommend increasing to at least 15–20 hours.")
    if past_scores < 60:
        recs.append("Past scores indicate foundational gaps. Consider additional support sessions.")
    recs.append(f"Learner Category — {cluster}: {CATEGORY_DESC[cluster]}")
    return recs


# ═══════════════════════════════════════════════════════
#  PAGE 1 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════
if st.session_state.page == "Analytics Dashboard":

    st.markdown("""
    <div class='about-card'>
        <div class='about-title'>GradeMinds AI — Intelligent Learning Analytics</div>
        <div class='about-desc'>
            GradeMinds AI is a machine learning-powered platform built to help educators and academic
            mentors make data-informed decisions about student outcomes. By analysing performance
            indicators such as study hours, attendance, and past scores, the system predicts whether
            a student is likely to pass or fail, classifies them into meaningful learner categories,
            and generates personalised recommendations — all in one place.<br><br>
            Designed for teachers and mentors who want early visibility into struggling students,
            GradeMinds AI turns raw academic data into clear, actionable insight so no student
            falls through the cracks.
        </div>
        <div class='about-pills'>
            <span class='pill'>Pass / Fail Prediction</span>
            <span class='pill'>6-Tier Learner Classification</span>
            <span class='pill'>Educator Recommendations</span>
            <span class='pill'>Performance Visualisation</span>
            <span class='pill'>Decision Tree Classifier · Linear Regression</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='page-title'>Class Analytics Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Upload a student dataset to view class-wide performance insights and learner breakdowns.</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0f1e10; border:1px solid #1e4020; border-radius:8px;
         padding:12px 18px; margin-bottom:16px; font-size:0.83rem; color:#4a9060; display:flex; align-items:center; gap:10px;'>
        <span style='font-size:1rem'>&#9432;</span>
        <span>A sample dataset has been pre-loaded below to demonstrate the analytics capabilities of this platform.
        You may also upload your own CSV file to analyse a different cohort.</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload your own CSV (optional — sample data is pre-loaded)", type=["csv"])

    import os
    if uploaded:
        df = pd.read_csv(uploaded)
    elif os.path.exists("student_performance_dataset.csv"):
        df = pd.read_csv("student_performance_dataset.csv")
    else:
        df = None

    if df is not None:
        total = len(df)
        pass_count = (df["Pass_Fail"] == "Pass").sum() if "Pass_Fail" in df.columns else 0
        fail_count = total - pass_count
        avg_score  = df["Final_Exam_Score"].mean()
        avg_att    = df["Attendance_Rate"].mean()

        # KPI Row
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, (val, color, label) in zip([c1,c2,c3,c4,c5], [
            (str(total),         "#7c9fff", "Total Students"),
            (str(pass_count),    "#34d399", "Passed"),
            (str(fail_count),    "#f87171", "Failed"),
            (f"{avg_score:.1f}", "#7c9fff", "Avg Final Score"),
            (f"{avg_att:.1f}%",  "#7c9fff", "Avg Attendance"),
        ]):
            with col:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value' style='color:{color}'>{val}</div>
                    <div class='metric-label'>{label}</div></div>""", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Charts Row 1
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-title'>Final Exam Score Distribution</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor("#0c0e14"); ax.set_facecolor("#0c0e14")
            ax.hist(df["Final_Exam_Score"], bins=20, color="#4f6fd4", edgecolor="#0c0e14", alpha=0.9)
            ax.set_xlabel("Score", color="#5a6480", fontsize=9)
            ax.set_ylabel("Count", color="#5a6480", fontsize=9)
            ax.tick_params(colors="#5a6480")
            for s in ax.spines.values(): s.set_edgecolor("#1e2438")
            st.pyplot(fig); plt.close()

        with col2:
            st.markdown("<div class='section-title'>Pass / Fail Distribution</div>", unsafe_allow_html=True)
            if "Pass_Fail" in df.columns:
                fig, ax = plt.subplots(figsize=(5, 3.2))
                fig.patch.set_facecolor("#0c0e14"); ax.set_facecolor("#0c0e14")
                counts = df["Pass_Fail"].value_counts()
                wedges, texts, autotexts = ax.pie(
                    counts, labels=counts.index, autopct="%1.1f%%",
                    colors=["#34d399","#f87171"], startangle=90,
                    wedgeprops=dict(edgecolor="#0c0e14", linewidth=2))
                for t in texts: t.set_color("#5a6480")
                for at in autotexts: at.set_color("white"); at.set_fontweight("bold")
                st.pyplot(fig); plt.close()

        # Charts Row 2
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("<div class='section-title'>Attendance Rate vs Final Score</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor("#0c0e14"); ax.set_facecolor("#0c0e14")
            cmap = df["Pass_Fail"].map({"Pass":"#34d399","Fail":"#f87171"}) if "Pass_Fail" in df.columns else "#7c9fff"
            ax.scatter(df["Attendance_Rate"], df["Final_Exam_Score"], c=cmap, alpha=0.55, s=16, edgecolors="none")
            ax.set_xlabel("Attendance Rate (%)", color="#5a6480", fontsize=9)
            ax.set_ylabel("Final Score", color="#5a6480", fontsize=9)
            ax.tick_params(colors="#5a6480")
            for s in ax.spines.values(): s.set_edgecolor("#1e2438")
            ax.legend(handles=[mpatches.Patch(color="#34d399",label="Pass"), mpatches.Patch(color="#f87171",label="Fail")],
                      facecolor="#0c0e14", labelcolor="white", fontsize=8)
            st.pyplot(fig); plt.close()

        with col4:
            st.markdown("<div class='section-title'>Study Hours vs Final Score</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.2))
            fig.patch.set_facecolor("#0c0e14"); ax.set_facecolor("#0c0e14")
            ax.scatter(df["Study_Hours_per_Week"], df["Final_Exam_Score"], c="#6a8fff", alpha=0.55, s=16, edgecolors="none")
            ax.set_xlabel("Study Hours / Week", color="#5a6480", fontsize=9)
            ax.set_ylabel("Final Score", color="#5a6480", fontsize=9)
            ax.tick_params(colors="#5a6480")
            for s in ax.spines.values(): s.set_edgecolor("#1e2438")
            st.pyplot(fig); plt.close()

        # Learner Category Breakdown
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Learner Category Breakdown</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#0e1420; border:1px solid #1e3050; border-radius:8px;
             padding:12px 18px; margin-bottom:16px; font-size:0.83rem; color:#5a7090;'>
            Note: The learner categories below are derived from rule-based analysis of the uploaded
            dataset (study hours, attendance, and exam scores). There is no ML model behind this
            section — it is purely data-driven classification for educator reference.
        </div>
        """, unsafe_allow_html=True)

        df["Learner_Category"] = df.apply(assign_cluster_label, axis=1)
        cat_counts = df["Learner_Category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0).reset_index()
        cat_counts.columns = ["Category", "Count"]
        cat_counts["Percentage"] = (cat_counts["Count"] / total * 100).round(1).astype(str) + "%"

        cols = st.columns(6)
        for col, (_, row) in zip(cols, cat_counts.iterrows()):
            with col:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value' style='font-size:1.8rem'>{row['Count']}</div>
                    <div style='margin:10px 0 6px 0'>
                        <span class='{BADGE_MAP[row["Category"]]}'>{row['Category']}</span>
                    </div>
                    <div class='metric-label'>{row['Percentage']}</div>
                    <div style='font-size:0.72rem;color:#3a4260;margin-top:8px;line-height:1.5'>
                        {CATEGORY_DESC[row["Category"]][:60]}...
                    </div>
                </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background:#13172060;border:1px solid #1e2438;border-radius:14px;text-align:center;padding:40px;'>
            <div style='font-size:0.9rem;color:#3a4260;'>
                Sample dataset not found. Please upload a CSV file to load the dashboard.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  PAGE 2 — STUDENT PREDICTOR
# ═══════════════════════════════════════════════════════
else:
    st.markdown("<div class='page-title'>Student Outcome Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Enter a student's academic profile to predict their exam outcome and receive tailored guidance for educators.</div>", unsafe_allow_html=True)

    try:
        model, feature_names = load_model()
        model_loaded = True
    except:
        model_loaded = False
        st.warning("Model files not found. Make sure best_model.pkl and feature_names.pkl are in the same folder as app.py.")

    with st.form("student_form"):
        st.markdown("<div class='section-title'>Student Profile</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])

            st.markdown("**Study Hours per Week**")
            sh_col1, sh_col2 = st.columns([3,1])
            with sh_col1:
                study_hours_s = st.slider("study_hours_slider", 0, 40, 15, label_visibility="collapsed")
            with sh_col2:
                study_hours_n = st.number_input("study_hours_num", 0, 40, study_hours_s, label_visibility="collapsed")
            study_hours = study_hours_n if study_hours_n != study_hours_s else study_hours_s

            st.markdown("**Attendance Rate (%)**")
            at_col1, at_col2 = st.columns([3,1])
            with at_col1:
                attendance_s = st.slider("att_slider", 0.0, 100.0, 75.0, step=0.5, label_visibility="collapsed")
            with at_col2:
                attendance_n = st.number_input("att_num", 0.0, 100.0, attendance_s, step=0.5, label_visibility="collapsed")
            attendance = attendance_n if attendance_n != attendance_s else attendance_s

        with col2:
            extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

            st.markdown("**Past Exam Score**")
            ps_col1, ps_col2 = st.columns([3,1])
            with ps_col1:
                past_scores_s = st.slider("past_slider", 0, 100, 65, label_visibility="collapsed")
            with ps_col2:
                past_scores_n = st.number_input("past_num", 0, 100, past_scores_s, label_visibility="collapsed")
            past_scores = past_scores_n if past_scores_n != past_scores_s else past_scores_s

        submitted = st.form_submit_button("Run Prediction")

    if submitted and model_loaded:
        # Use default/median values for removed fields
        row = pd.DataFrame([[
            1 if gender == "Male" else 0,
            study_hours, attendance, past_scores,
            1,  # internet_access default Yes
            1 if extracurricular == "Yes" else 0,
            0,  # parental_edu HS default
            0,  # parental_edu Masters default
            0,  # parental_edu PhD default
        ]], columns=feature_names)

        pred       = model.predict(row)[0]
        prob       = model.predict_proba(row)[0][pred]
        pred_label = "Pass" if pred == 1 else "Fail"
        cluster    = assign_cluster_label({
            "Final_Exam_Score":    past_scores,
            "Attendance_Rate":     attendance,
            "Study_Hours_per_Week":study_hours,
        })

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Prediction Results</div>", unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            badge = "badge-pass" if pred_label == "Pass" else "badge-fail"
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label' style='margin-bottom:12px'>Predicted Outcome</div>
                <span class='{badge}'>{pred_label}</span></div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{prob*100:.1f}%</div>
                <div class='metric-label'>Model Confidence</div></div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label' style='margin-bottom:12px'>Learner Category</div>
                <span class='{BADGE_MAP[cluster]}'>{cluster}</span></div>""", unsafe_allow_html=True)

        # Category description
        st.markdown(f"""
        <div style='background:#0e1420;border:1px solid #1e2438;border-radius:10px;
             padding:14px 20px;margin-top:12px;font-size:0.88rem;color:#8892a4;'>
            <strong style='color:#c8d6ff'>{cluster}</strong> — {CATEGORY_DESC[cluster]}
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Recommendations for Educator</div>", unsafe_allow_html=True)
        recs     = get_recommendation(pred_label, cluster, study_hours, attendance, past_scores)
        rec_html = "".join(
            f"<div style='margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #1e2438'>{r}</div>"
            for r in recs
        )
        st.markdown(f"<div class='rec-box'>{rec_html}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
