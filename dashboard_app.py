"""
STUDENT SUCCESS EARLY WARNING DASHBOARD
Interactive dashboard for Ashesi University stakeholders

Author: Student Success Prediction Team
Date: 2025-11-19
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Ashesi Student Success Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling - Color-blind friendly palette
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0066cc;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.25rem;
        border-radius: 0.75rem;
        border-left: 5px solid #0066cc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .insight-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.25rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border-left: 5px solid #0066cc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.25rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border-left: 5px solid #ff9800;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1.25rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border-left: 5px solid #4caf50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    /* Button styling */
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0052a3;
        box-shadow: 0 4px 8px rgba(0,102,204,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    """Load and prepare data for dashboard"""
    # Load clean features (use sample data for deployment)
    try:
        # Try sample data first (for deployment)
        data = pd.read_csv('data/early_warning_features_sample.csv')
    except FileNotFoundError:
        # Fall back to real data (for local testing)
        data = pd.read_csv('data/early_warning_features.csv')

    # Load model comparison results
    model_results = pd.read_csv('results_phase1/model_comparison_clean.csv')

    # Load feature comparison
    feature_comp = pd.read_csv('results_phase1/feature_comparison.csv')

    return data, model_results, feature_comp

# Load data
try:
    data, model_results, feature_comp = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🎓 Ashesi Student Success")
st.sidebar.markdown("### Early Warning Dashboard")

page = st.sidebar.radio(
    "Select View:",
    ["🏠 Overview", "👥 Academic Advisor", "📚 Faculty Insights", "📊 Executive Summary"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About This Dashboard")
st.sidebar.info("""
This dashboard uses machine learning to identify students who may need academic support **before** they go on probation.

**Model Performance:**
- 92% Recall (catches 9/10 at-risk students)
- 6-12 months early warning
- Trained on 2011-2019 data, validated on 2022-2025

**Data as of:** November 2025
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")
if data_loaded:
    total_students = len(data)
    at_risk_count = data['On_Probation'].sum()
    at_risk_pct = at_risk_count / total_students * 100

    st.sidebar.metric("Total Students Analyzed", f"{total_students:,}")
    st.sidebar.metric("At-Risk Students", f"{at_risk_count} ({at_risk_pct:.1f}%)")

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================

if page == "🏠 Overview":
    st.markdown('<p class="main-header">Student Success Early Warning System</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to the Ashesi Student Success Dashboard

    This interactive dashboard helps identify students who may need academic support **before** they go on probation.
    The system analyzes historical academic patterns to provide 6-12 months of early warning.
    """)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Students Analyzed",
            value=f"{len(data):,}",
            delta="2011-2025 cohorts"
        )

    with col2:
        at_risk = data['On_Probation'].sum()
        st.metric(
            label="At-Risk Students",
            value=f"{at_risk}",
            delta=f"{at_risk/len(data)*100:.1f}% of total",
            delta_color="inverse"
        )

    with col3:
        best_model = model_results.loc[model_results['Test_F2'].idxmax()]
        st.metric(
            label="Model Recall",
            value=f"{best_model['Test_Recall']*100:.0f}%",
            delta="Catches 9/10 at-risk students"
        )

    with col4:
        st.metric(
            label="Early Warning",
            value="6-12 mo",
            delta="Before probation"
        )

    st.markdown("---")

    # Model performance visualization
    st.subheader("📊 Model Performance Overview")

    col1, col2 = st.columns(2)

    with col1:
        # ROC-AUC comparison
        fig_models = go.Figure()

        fig_models.add_trace(go.Bar(
            x=model_results['Model'],
            y=model_results['Test_Recall'] * 100,
            name='Recall',
            marker_color='steelblue'
        ))

        fig_models.add_trace(go.Bar(
            x=model_results['Model'],
            y=model_results['Test_Precision'] * 100,
            name='Precision',
            marker_color='coral'
        ))

        fig_models.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score (%)",
            barmode='group',
            height=400
        )

        st.plotly_chart(fig_models, use_container_width=True)

    with col2:
        # Feature importance
        st.markdown("### Key Predictive Factors")

        st.markdown("""
        **Top 5 Warning Signs:**

        1. **Historical CGPA < 2.5**
           - At-risk: 1.76 avg
           - Thriving: 3.05 avg

        2. **High Failure Rate**
           - At-risk: 47% of courses
           - Thriving: 8% of courses

        3. **Low GPA Improvement**
           - At-risk: +0.12 per semester
           - Thriving: +0.27 per semester

        4. **Fewer Courses Taken**
           - At-risk: 17.6 courses
           - Thriving: 23.0 courses

        5. **Low A-grade Percentage**
           - At-risk: 1.8% A's
           - Thriving: 21.7% A's
        """)

    st.markdown("---")

    # Key insights
    st.subheader("💡 Key Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>📈 GPA Trajectory Matters</h4>
        <p>Students with declining GPAs are at risk even if current CGPA > 2.0</p>
        <p><strong>Action:</strong> Monitor trends, not just snapshots</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>🚨 Early Failures Predict Risk</h4>
        <p>At-risk students fail 6x more courses historically (47% vs 8%)</p>
        <p><strong>Action:</strong> Trigger tutoring after any failure</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="insight-box">
        <h4>⏰ Intervene Early</h4>
        <p>Patterns established in semester 1-2 tend to persist</p>
        <p><strong>Action:</strong> Mandatory check-ins after semester 1</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation guidance
    st.subheader("🧭 How to Use This Dashboard")

    st.markdown("""
    **Select your role from the sidebar:**

    - **👥 Academic Advisor:** View individual student risk profiles, GPA trajectories, and intervention recommendations
    - **📚 Faculty Insights:** Analyze course difficulty patterns, identify stress points in curriculum
    - **📊 Executive Summary:** High-level KPIs, ROI projections, and strategic recommendations

    **Features:**
    - Interactive filters (program, admission year, demographics)
    - What-if scenario calculator (predict impact of GPA changes)
    - Export student lists for outreach
    - Detailed student profiles with intervention suggestions
    """)

# ============================================================================
# PAGE: ACADEMIC ADVISOR
# ============================================================================

elif page == "👥 Academic Advisor":
    st.markdown('<p class="main-header">Academic Advisor Dashboard</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Student Risk Assessment & Intervention Planning

    This view helps you identify which students need support and what type of intervention is most appropriate.
    """)

    # Filters
    st.sidebar.markdown("### Filters")

    # Risk level filter
    risk_filter = st.sidebar.multiselect(
        "Risk Level",
        options=["High Risk (Probation)", "Monitor"],
        default=["High Risk (Probation)", "Monitor"]
    )

    # Admission year filter
    years = sorted(data['Admission Year'].dropna().unique())
    year_filter = st.sidebar.multiselect(
        "Admission Year",
        options=years,
        default=years[-3:] if len(years) >= 3 else years  # Default to last 3 years
    )

    # Gender filter
    gender_filter = st.sidebar.multiselect(
        "Gender",
        options=data['Gender'].dropna().unique(),
        default=list(data['Gender'].dropna().unique())
    )

    # International status filter
    intl_filter = st.sidebar.radio(
        "Student Type",
        options=["All", "International Only", "Local Only"],
        index=0
    )

    # Apply filters
    filtered_data = data.copy()

    # Risk filter
    if "High Risk (Probation)" in risk_filter and "Monitor" not in risk_filter:
        filtered_data = filtered_data[filtered_data['On_Probation'] == 1]
    elif "Monitor" in risk_filter and "High Risk (Probation)" not in risk_filter:
        filtered_data = filtered_data[filtered_data['On_Probation'] == 0]

    # Year filter
    if year_filter:
        filtered_data = filtered_data[filtered_data['Admission Year'].isin(year_filter)]

    # Gender filter
    if gender_filter:
        filtered_data = filtered_data[filtered_data['Gender'].isin(gender_filter)]

    # International filter
    if intl_filter == "International Only":
        filtered_data = filtered_data[filtered_data['International'] == 1]
    elif intl_filter == "Local Only":
        filtered_data = filtered_data[filtered_data['International'] == 0]

    st.markdown("---")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Students in View", f"{len(filtered_data):,}")

    with col2:
        at_risk_filtered = filtered_data['On_Probation'].sum()
        st.metric(
            "At-Risk Students",
            f"{at_risk_filtered}",
            delta=f"{at_risk_filtered/len(filtered_data)*100:.1f}% of view" if len(filtered_data) > 0 else "0%"
        )

    with col3:
        avg_cgpa = filtered_data['Historical_CGPA_last'].mean()
        st.metric("Avg Historical CGPA", f"{avg_cgpa:.2f}")

    with col4:
        avg_fail_rate = filtered_data['Historical_Fail_pct'].mean()
        st.metric("Avg Failure Rate", f"{avg_fail_rate:.1f}%")

    st.markdown("---")

    # Student risk scatter plot
    st.subheader("📊 Student Risk Assessment Map")

    st.markdown("""
    **How to read this chart:**
    - **X-axis:** Historical CGPA (before final semester)
    - **Y-axis:** GPA trajectory (slope - positive = improving, negative = declining)
    - **Red dots:** Students who went on probation
    - **Green dots:** Thriving students
    - **Orange line:** CGPA 2.0 threshold (probation boundary)
    """)

    # Create scatter plot
    fig_scatter = go.Figure()

    # Thriving students
    thriving = filtered_data[filtered_data['On_Probation'] == 0]
    fig_scatter.add_trace(go.Scatter(
        x=thriving['Historical_CGPA_last'],
        y=thriving['GPA_Slope'],
        mode='markers',
        name='Thriving',
        marker=dict(
            size=8,
            color='green',
            opacity=0.6,
            line=dict(width=0.5, color='darkgreen')
        ),
        text=[f"Student {i}<br>CGPA: {cgpa:.2f}<br>Slope: {slope:.3f}"
              for i, cgpa, slope in zip(thriving.index, thriving['Historical_CGPA_last'], thriving['GPA_Slope'])],
        hovertemplate='%{text}<extra></extra>'
    ))

    # At-risk students
    at_risk_df = filtered_data[filtered_data['On_Probation'] == 1]
    fig_scatter.add_trace(go.Scatter(
        x=at_risk_df['Historical_CGPA_last'],
        y=at_risk_df['GPA_Slope'],
        mode='markers',
        name='At-Risk',
        marker=dict(
            size=12,
            color='red',
            opacity=0.8,
            line=dict(width=1, color='darkred')
        ),
        text=[f"Student {i}<br>CGPA: {cgpa:.2f}<br>Slope: {slope:.3f}<br>⚠️ HIGH RISK"
              for i, cgpa, slope in zip(at_risk_df.index, at_risk_df['Historical_CGPA_last'], at_risk_df['GPA_Slope'])],
        hovertemplate='%{text}<extra></extra>'
    ))

    # Add threshold lines
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="black",
                         annotation_text="No GPA change", annotation_position="right")
    fig_scatter.add_vline(x=2.0, line_dash="dash", line_color="orange", line_width=2,
                         annotation_text="Probation Threshold (2.0)", annotation_position="top")
    fig_scatter.add_vline(x=2.5, line_dash="dot", line_color="gold", line_width=1.5,
                         annotation_text="Monitor Threshold (2.5)", annotation_position="top")

    fig_scatter.update_layout(
        title="Student Academic Trajectory Map",
        xaxis_title="Historical CGPA (before final semester)",
        yaxis_title="GPA Slope (trajectory)",
        height=500,
        hovermode='closest'
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # Student list
    st.subheader("📋 Student List - Prioritized by Risk")

    # Prepare display data
    display_data = filtered_data[[
        'StudentRef', 'Historical_CGPA_last', 'GPA_Slope', 'Historical_Fail_pct',
        'Historical_courses_taken', 'Gender', 'International', 'On_Probation'
    ]].copy()

    # Rename columns for display
    display_data.columns = [
        'Student ID', 'Historical CGPA', 'GPA Slope', 'Failure Rate (%)',
        'Courses Taken', 'Gender', 'International', 'At Risk'
    ]

    # Convert boolean to Yes/No
    display_data['International'] = display_data['International'].map({1: 'Yes', 0: 'No'})
    display_data['At Risk'] = display_data['At Risk'].map({1: '⚠️ YES', 0: 'No'})

    # Sort by risk (at-risk first) then by CGPA
    display_data = display_data.sort_values(['At Risk', 'Historical CGPA'], ascending=[False, True])

    # Format numbers
    display_data['Historical CGPA'] = display_data['Historical CGPA'].round(2)
    display_data['GPA Slope'] = display_data['GPA Slope'].round(3)
    display_data['Failure Rate (%)'] = display_data['Failure Rate (%)'].round(1)

    # Display with row highlighting
    st.dataframe(
        display_data,
        use_container_width=True,
        height=400
    )

    # Export button
    csv = display_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Student List (CSV)",
        data=csv,
        file_name=f"at_risk_students_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # Intervention recommendations
    st.subheader("💡 Intervention Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="warning-box">
        <h4>🚨 HIGH PRIORITY (Immediate Action)</h4>
        <p><strong>Criteria:</strong></p>
        <ul>
            <li>Historical CGPA < 2.0</li>
            <li>Failure rate > 30%</li>
            <li>Negative GPA slope</li>
        </ul>
        <p><strong>Recommended Actions:</strong></p>
        <ul>
            <li>Mandatory academic advising meeting (within 1 week)</li>
            <li>Create academic recovery plan</li>
            <li>Enroll in tutoring program (required attendance)</li>
            <li>Reduce course load if overextended</li>
            <li>Weekly check-ins with advisor</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="success-box">
        <h4>⚠️  MONITOR (Proactive Outreach)</h4>
        <p><strong>Criteria:</strong></p>
        <ul>
            <li>Historical CGPA 2.0 - 2.5</li>
            <li>Any course failures</li>
            <li>Declining GPA trend</li>
        </ul>
        <p><strong>Recommended Actions:</strong></p>
        <ul>
            <li>Email outreach with support resources</li>
            <li>Optional tutoring sessions</li>
            <li>Time management workshop</li>
            <li>Connect with peer mentor</li>
            <li>Monitor progress next semester</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE: FACULTY INSIGHTS
# ============================================================================

elif page == "📚 Faculty Insights":
    st.markdown('<p class="main-header">Faculty Course Insights</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Course Difficulty Analysis & Curriculum Recommendations

    This view helps identify which courses are academic stress points and where curriculum improvements could have the biggest impact.
    """)

    st.markdown("---")

    # Summary insights
    st.subheader("📊 Key Findings")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Avg Failure Rate (At-Risk)",
            "47.3%",
            delta="+39.2pp vs thriving",
            delta_color="inverse"
        )

    with col2:
        st.metric(
            "Courses Taken (At-Risk)",
            "17.6",
            delta="-5.4 vs thriving",
            delta_color="inverse"
        )

    with col3:
        st.metric(
            "A-Grade Rate (At-Risk)",
            "1.8%",
            delta="-19.9pp vs thriving",
            delta_color="inverse"
        )

    st.markdown("---")

    # Grade distribution comparison
    st.subheader("📈 Grade Distribution: At-Risk vs Thriving Students")

    # Calculate grade distributions
    at_risk = data[data['On_Probation'] == 1]
    thriving = data[data['On_Probation'] == 0]

    grades = ['A', 'B', 'C']
    at_risk_grades = [at_risk[f'Historical_{g}_pct'].mean() for g in grades] + [at_risk['Historical_Fail_pct'].mean()]
    thriving_grades = [thriving[f'Historical_{g}_pct'].mean() for g in grades] + [thriving['Historical_Fail_pct'].mean()]

    grade_labels = ['A Grades', 'B Grades', 'C Grades', 'Failures (D/E/F)']

    fig_grades = go.Figure()

    fig_grades.add_trace(go.Bar(
        name='At-Risk Students',
        x=grade_labels,
        y=at_risk_grades,
        marker_color='coral',
        text=[f'{v:.1f}%' for v in at_risk_grades],
        textposition='outside'
    ))

    fig_grades.add_trace(go.Bar(
        name='Thriving Students',
        x=grade_labels,
        y=thriving_grades,
        marker_color='steelblue',
        text=[f'{v:.1f}%' for v in thriving_grades],
        textposition='outside'
    ))

    fig_grades.update_layout(
        title="Historical Grade Distribution Comparison",
        xaxis_title="Grade Category",
        yaxis_title="Percentage of Courses",
        barmode='group',
        height=400
    )

    st.plotly_chart(fig_grades, use_container_width=True)

    st.markdown("---")

    # Curriculum recommendations
    st.subheader("💡 Curriculum & Teaching Recommendations")

    st.markdown("""
    <div class="insight-box">
    <h4>Based on the analysis of 2,242 students across 2011-2025:</h4>

    **1. High-Failure Courses Need Intervention**
    - At-risk students fail 47% of courses historically (vs 8% for thriving students)
    - **Recommendation:** Identify top 10 courses with highest failure rates
    - **Action:** Add supplemental instruction sessions in weeks 5-7 (before midterms)

    **2. Early Course Failures Predict Future Struggles**
    - Students who fail courses in semester 1-2 are 6x more likely to go on probation
    - **Recommendation:** Mandatory tutoring after ANY course failure
    - **Action:** Create "Academic Recovery" program for students with early failures

    **3. Course Load Matters**
    - At-risk students take fewer courses (17.6 vs 23.0)
    - May indicate course drops, slow progression, or overwhelm
    - **Recommendation:** Monitor students who drop multiple courses
    - **Action:** Advising intervention when student drops 2+ courses in a semester

    **4. Grade Distribution Shows Engagement Gap**
    - At-risk students earn very few A's (1.8% vs 21.7%)
    - Not just about failing - also about lack of excellence
    - **Recommendation:** Early engagement initiatives (first 2 weeks of semester)
    - **Action:** Faculty outreach to students missing assignments in week 2

    **5. Specific Course Domains May Need Redesign**
    - While we don't have course-level data in this analysis, historical patterns suggest:
      - Computing courses: May need more scaffolding for novices
      - Math courses: Early warning indicators - failures here predict broader struggles
      - Writing-intensive courses: International students may need extra support
    - **Recommendation:** Conduct course-level failure rate audit
    - **Action:** Redesign top 3 highest-failure courses using evidence-based pedagogy
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Recommendations table
    st.subheader("📋 Action Items for Faculty & Academic Affairs")

    recommendations_df = pd.DataFrame({
        'Priority': ['🔴 High', '🔴 High', '🟡 Medium', '🟡 Medium', '🟢 Low'],
        'Action Item': [
            'Audit course failure rates by course code',
            'Implement mandatory tutoring after any course failure',
            'Add supplemental instruction to high-failure courses',
            'Create "Academic Recovery" program for students with early failures',
            'Faculty training on early warning signs'
        ],
        'Target': [
            'Top 10 highest-failure courses',
            'All students (automatic trigger)',
            'Courses with >20% failure rate',
            'Students who fail 2+ courses in first year',
            'All faculty teaching large intro courses'
        ],
        'Expected Impact': [
            'Identify curriculum redesign targets',
            'Reduce future failures by 30%',
            'Reduce course failures by 15-20%',
            'Prevent 10-15 probations/year',
            'Earlier identification of struggling students'
        ],
        'Timeline': [
            '1 semester (data analysis)',
            'Immediate (policy change)',
            '2 semesters (pilot → scale)',
            '1 year (develop program)',
            'Ongoing (annual training)'
        ]
    })

    st.dataframe(recommendations_df, use_container_width=True, height=250)

    # Export button
    csv_rec = recommendations_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Recommendations (CSV)",
        data=csv_rec,
        file_name=f"curriculum_recommendations_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ============================================================================
# PAGE: EXECUTIVE SUMMARY
# ============================================================================

elif page == "📊 Executive Summary":
    st.markdown('<p class="main-header">Executive Dashboard</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Strategic Overview for Senior Leadership

    High-level KPIs, ROI projections, and strategic recommendations for Ashesi's student success initiatives.
    """)

    st.markdown("---")

    # KPI Dashboard
    st.subheader("📊 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Current Probation Rate",
            value="4.2%",
            delta="95 of 2,242 students",
            delta_color="inverse"
        )

    with col2:
        st.metric(
            label="Early Warning Recall",
            value="92%",
            delta="Catches 9/10 at-risk students",
            delta_color="normal"
        )

    with col3:
        st.metric(
            label="Intervention Lead Time",
            value="6-12 mo",
            delta="Before probation occurs",
            delta_color="normal"
        )

    with col4:
        st.metric(
            label="Projected ROI",
            value="+270%",
            delta="$135K net benefit/year",
            delta_color="normal"
        )

    st.markdown("---")

    # Trend analysis
    st.subheader("📈 Probation Trends by Cohort")

    # Group by admission year
    trend_data = data.groupby('Admission Year').agg({
        'On_Probation': ['sum', 'count', 'mean']
    }).reset_index()
    trend_data.columns = ['Year', 'Probations', 'Total', 'Rate']
    trend_data['Rate'] = trend_data['Rate'] * 100
    trend_data = trend_data.sort_values('Year')

    fig_trend = go.Figure()

    fig_trend.add_trace(go.Bar(
        x=trend_data['Year'],
        y=trend_data['Probations'],
        name='Probations',
        marker_color='coral',
        yaxis='y'
    ))

    fig_trend.add_trace(go.Scatter(
        x=trend_data['Year'],
        y=trend_data['Rate'],
        name='Probation Rate (%)',
        marker_color='steelblue',
        mode='lines+markers',
        line=dict(width=3),
        yaxis='y2'
    ))

    fig_trend.update_layout(
        title="Academic Probation Trends (2018-2025)",
        xaxis_title="Admission Year",
        yaxis=dict(title="Number of Probations", side='left'),
        yaxis2=dict(title="Probation Rate (%)", side='right', overlaying='y', range=[0, 10]),
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")

    # ROI Analysis
    st.subheader("💰 Return on Investment Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>Intervention Costs (Annual)</h4>
        <ul>
            <li><strong>Students flagged:</strong> ~200 (10% of 2,000 enrolled)</li>
            <li><strong>Cost per student:</strong> $250/year</li>
            <ul>
                <li>Tutoring sessions: $150</li>
                <li>Advising hours: $75</li>
                <li>Workshops: $25</li>
            </ul>
            <li><strong>Total Cost:</strong> <span style="font-size:1.5rem; color:#1f77b4;">$50,000/year</span></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>Financial Benefits (Annual)</h4>
        <ul>
            <li><strong>Baseline probations:</strong> 80 students (4% rate)</li>
            <li><strong>Probations prevented:</strong> 37 students (92% recall × 50% effectiveness)</li>
            <li><strong>Tuition retained:</strong> $5,000/semester × 2 semesters</li>
            <li><strong>Total Benefit:</strong> <span style="font-size:1.5rem; color:#28a745;">$185,000/year</span></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="success-box">
    <h3 style="text-align: center;">Net ROI: <span style="font-size:2rem; color:#28a745;">+$135,000/year (+270%)</span></h3>
    <p style="text-align: center;">For every $1 invested in interventions, Ashesi gains $3.70 in retained tuition</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Strategic recommendations
    st.subheader("🎯 Strategic Recommendations")

    st.markdown("""
    ### **1. Implement Early Warning System (Immediate - Q1 2026)**

    **What:** Deploy this dashboard to academic advisors and launch tiered intervention program

    **Why:** System achieves 92% recall - can identify and support 9 out of 10 at-risk students before probation

    **How:**
    - Train advisors on dashboard usage (1-day workshop)
    - Create intervention protocols:
      - High risk (CGPA < 2.0): Mandatory advising + tutoring
      - Medium risk (CGPA 2.0-2.5): Optional support + email outreach
    - Monitor outcomes semester-by-semester

    **Investment:** $50K/year | **Return:** $185K/year | **ROI:** +270%

    ---

    ### **2. Redesign High-Failure Courses (Year 1-2)**

    **What:** Audit courses with >20% failure rates and implement evidence-based redesigns

    **Why:** At-risk students fail 47% of courses (vs 8% for thriving students) - curriculum improvements can reduce this gap

    **How:**
    - Conduct course-level failure rate analysis (semester 1)
    - Pilot redesigns in top 3 highest-failure courses (semester 2)
    - Add supplemental instruction in weeks 5-7 (before midterms)
    - Scale to top 10 courses over 2 years

    **Investment:** $30K/year | **Impact:** Reduce course failures by 15-20%

    ---

    ### **3. Launch "Academic Recovery" Program (Year 1)**

    **What:** Structured support program for students who fail any course

    **Why:** Early course failures are the strongest predictor of future probation - intervene immediately

    **How:**
    - Automatic enrollment after any course failure
    - Mandatory components:
      - Meet with academic coach within 1 week
      - Attend study skills workshop
      - Weekly check-ins for remainder of semester
    - Track outcomes: Do participants have lower subsequent failure rates?

    **Investment:** $25K/year | **Impact:** Prevent 10-15 probations/year

    ---

    ### **4. Demographic-Specific Support (Year 2)**

    **What:** Tailored programs for groups with higher probation rates

    **Why:** International students (5.9% vs 3.8%), male students (5.6% vs 2.7%) face different challenges

    **How:**
    - International students: Academic writing workshops, English language support
    - Male students: Investigate root causes (study habits? time management? social pressures?)
    - Program-specific: Engineering/Computing students → math tutoring pipeline

    **Investment:** $20K/year | **Impact:** Reduce demographic disparities by 30%

    ---

    ### **5. Continuous Model Improvement (Ongoing)**

    **What:** Retrain model annually with latest data, monitor performance

    **Why:** Curriculum changes, admissions policies, student demographics evolve - model must adapt

    **How:**
    - Annual data refresh (add latest cohort)
    - Retrain model and validate performance
    - Fairness audit (ensure no demographic bias increase)
    - Dashboard updates based on user feedback

    **Investment:** $15K/year | **Impact:** Maintain 90%+ recall over time
    """)

    st.markdown("---")

    # 5-year projection
    st.subheader("📅 5-Year Impact Projection")

    years = list(range(2026, 2031))
    probations_baseline = [80, 80, 80, 80, 80]  # Without intervention
    probations_with_intervention = [
        80 - 37,  # Year 1: 37 prevented
        80 - 42,  # Year 2: Improved effectiveness
        80 - 45,  # Year 3: Full scale
        80 - 47,  # Year 4: Continuous improvement
        80 - 50   # Year 5: Maximum impact
    ]

    fig_projection = go.Figure()

    fig_projection.add_trace(go.Scatter(
        x=years,
        y=probations_baseline,
        name='Without Intervention',
        mode='lines+markers',
        line=dict(color='coral', width=3, dash='dash'),
        marker=dict(size=10)
    ))

    fig_projection.add_trace(go.Scatter(
        x=years,
        y=probations_with_intervention,
        name='With Intervention',
        mode='lines+markers',
        line=dict(color='green', width=3),
        marker=dict(size=10),
        fill='tonexty',
        fillcolor='rgba(0,255,0,0.1)'
    ))

    fig_projection.update_layout(
        title="5-Year Probation Reduction Projection",
        xaxis_title="Year",
        yaxis_title="Number of Probations",
        height=400
    )

    st.plotly_chart(fig_projection, use_container_width=True)

    st.markdown("""
    <div class="success-box">
    <h4>Cumulative 5-Year Impact (2026-2030):</h4>
    <ul>
        <li><strong>Probations Prevented:</strong> 221 students</li>
        <li><strong>Retention Improvement:</strong> +2.2% (from 96% to 98.2%)</li>
        <li><strong>Financial Benefit:</strong> $1.1M in retained tuition</li>
        <li><strong>Total Investment:</strong> $250K over 5 years</li>
        <li><strong>Net ROI:</strong> <span style="font-size:1.5rem; color:#28a745;">+$850K (+340%)</span></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><strong>Ashesi University Student Success Early Warning System</strong></p>
    <p>Dashboard created November 2025 | Model trained on 2011-2025 data | Next update: January 2026</p>
    <p>For questions or feedback, contact Academic Affairs</p>
</div>
""", unsafe_allow_html=True)
