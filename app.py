from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Jaya Jaya Institut | Dropout Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Data" / "data.csv"
MODEL_PATH = BASE_DIR / "student_dropout_model.pkl"


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F7F9FC;
    }

    [data-testid="stSidebar"] {
        background-color: #123C35;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Search / Student Index input */
    [data-testid="stSidebar"] input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: #FFFFFF !important;
        caret-color: #000000 !important;
    }

    [data-testid="stSidebar"] input::placeholder {
        color: #555555 !important;
        -webkit-text-fill-color: #555555 !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] [data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border-radius: 10px !important;
    }

    .main-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #123C35;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        color: #61706C;
        font-size: 1rem;
        margin-bottom: 1.3rem;
    }

    .section-title {
        color: #123C35;
        font-size: 1.25rem;
        font-weight: 750;
        margin-top: 0.8rem;
        margin-bottom: 0.7rem;
    }

    .prediction-box {
        background: white;
        border: 1px solid #E3E8ED;
        border-radius: 12px;
        padding: 18px;
        margin-top: 12px;
        margin-bottom: 20px;
    }

    .risk-high {
        background-color: #FDECEC;
        border: 1px solid #F2B8B5;
        padding: 0.8rem;
        border-radius: 10px;
        color: #9C2F2B;
        font-weight: 700;
        text-align: center;
    }

    .risk-medium {
        background-color: #FFF5DF;
        border: 1px solid #F1D18A;
        padding: 0.8rem;
        border-radius: 10px;
        color: #8A5A00;
        font-weight: 700;
        text-align: center;
    }

    .risk-low {
        background-color: #EAF7EF;
        border: 1px solid #B7DEC2;
        padding: 0.8rem;
        border-radius: 10px;
        color: #22643A;
        font-weight: 700;
        text-align: center;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E5EAF0;
        padding: 0.75rem 0.9rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner="Memuat dataset...")
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di:\n{DATA_PATH}\n\n"
            "Pastikan struktur folder adalah:\n"
            "Projek 2/\n"
            "├── app.py\n"
            "└── Data/\n"
            "    └── data.csv"
        )

    df = pd.read_csv(DATA_PATH, sep=";")

    # Fallback apabila CSV menggunakan koma.
    if len(df.columns) == 1:
        df = pd.read_csv(DATA_PATH, sep=None, engine="python")

    if df.empty:
        raise ValueError("Dataset kosong.")

    return df


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner="Memuat model Machine Learning...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di:\n{MODEL_PATH}\n\n"
            "Pastikan student_dropout_model.pkl berada "
            "di folder yang sama dengan app.py."
        )

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def create_features(dataframe):
    data = dataframe.copy()

    data["Approval_Rate_1st_Sem"] = safe_divide(
        data["Curricular_units_1st_sem_approved"],
        data["Curricular_units_1st_sem_enrolled"],
    )

    data["Evaluation_Rate_1st_Sem"] = safe_divide(
        data["Curricular_units_1st_sem_evaluations"],
        data["Curricular_units_1st_sem_enrolled"],
    )

    data["Approval_Rate_2nd_Sem"] = safe_divide(
        data["Curricular_units_2nd_sem_approved"],
        data["Curricular_units_2nd_sem_enrolled"],
    )

    data["Evaluation_Rate_2nd_Sem"] = safe_divide(
        data["Curricular_units_2nd_sem_evaluations"],
        data["Curricular_units_2nd_sem_enrolled"],
    )

    data["Grade_Change"] = (
        data["Curricular_units_2nd_sem_grade"]
        - data["Curricular_units_1st_sem_grade"]
    )

    data["Approved_Change"] = (
        data["Curricular_units_2nd_sem_approved"]
        - data["Curricular_units_1st_sem_approved"]
    )

    data["Total_Approved"] = (
        data["Curricular_units_1st_sem_approved"]
        + data["Curricular_units_2nd_sem_approved"]
    )

    data["Total_Enrolled"] = (
        data["Curricular_units_1st_sem_enrolled"]
        + data["Curricular_units_2nd_sem_enrolled"]
    )

    data["Total_Evaluations"] = (
        data["Curricular_units_1st_sem_evaluations"]
        + data["Curricular_units_2nd_sem_evaluations"]
    )

    data["Total_Without_Evaluation"] = (
        data["Curricular_units_1st_sem_without_evaluations"]
        + data["Curricular_units_2nd_sem_without_evaluations"]
    )

    data["Average_Semester_Grade"] = (
        data["Curricular_units_1st_sem_grade"]
        + data["Curricular_units_2nd_sem_grade"]
    ) / 2

    data["Grade_Ratio"] = safe_divide(
        data["Curricular_units_2nd_sem_grade"],
        data["Curricular_units_1st_sem_grade"],
    )

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    return data


# ============================================================
# PREDICTION FOR SELECTED STUDENT
# ============================================================

def predict_student(row, model):
    X_student = create_features(
        row.drop(columns=["Status"], errors="ignore")
    )

    raw_prediction = model.predict(X_student)[0]
    raw_probability = model.predict_proba(X_student)[0]

    # Sesuai mapping LabelEncoder pada notebook:
    # 0 = Dropout, 1 = Enrolled, 2 = Graduate
    if np.issubdtype(np.asarray(model.classes_).dtype, np.number):
        label_map = {
            0: "Dropout",
            1: "Enrolled",
            2: "Graduate",
        }

        predicted_status = label_map.get(
            int(raw_prediction),
            str(raw_prediction),
        )

        class_names = [
            label_map.get(int(value), str(value))
            for value in model.classes_
        ]
    else:
        predicted_status = str(raw_prediction)
        class_names = [str(value) for value in model.classes_]

    probability_dict = dict(
        zip(class_names, raw_probability)
    )

    dropout_probability = float(
        probability_dict.get("Dropout", 0.0)
    )

    if dropout_probability >= 0.70:
        risk_level = "High Risk"
    elif dropout_probability >= 0.40:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    return (
        predicted_status,
        dropout_probability,
        risk_level,
        probability_dict,
    )


# ============================================================
# LOAD
# ============================================================

try:
    df = load_data()
    model = load_model()

except Exception as error:
    st.error("Aplikasi gagal memuat data atau model.")
    st.exception(error)
    st.stop()


# ============================================================
# SIDEBAR — ONLY FOR STUDENT SEARCH / MODEL
# ============================================================

with st.sidebar:
    st.markdown("## 🎓 Jaya Jaya Institut")
    st.caption("Student Dropout Analytics")
    st.divider()

    st.markdown("### 🔎 Cari Student")

    student_index = st.number_input(
        "Student Index",
        min_value=1,
        max_value=len(df),
        value=1,
        step=1,
        help="Pilih nomor mahasiswa berdasarkan urutan baris pada dataset.",
    )

    predict_button = st.button(
        "🔮 Terapkan Model",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.caption(
        f"Total data mahasiswa: {len(df):,}"
    )

    st.caption(
        "Student Index digunakan sebagai referensi baris "
        "karena dataset tidak menyediakan Student ID asli."
    )


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'DASHBOARD ANALISIS & PREDIKSI DROPOUT MAHASISWA'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Monitoring status mahasiswa, performa akademik, '
    'dan penerapan model Machine Learning untuk screening dropout.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# KPI
# ============================================================

total = len(df)

dropout = (df["Status"] == "Dropout").sum()
graduate = (df["Status"] == "Graduate").sum()
enrolled = (df["Status"] == "Enrolled").sum()

dropout_rate = dropout / total if total else 0
graduate_rate = graduate / total if total else 0
enrolled_rate = enrolled / total if total else 0

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Mahasiswa",
    f"{total:,}",
)

c2.metric(
    "Dropout Rate",
    f"{dropout_rate:.2%}",
)

c3.metric(
    "Graduate Rate",
    f"{graduate_rate:.2%}",
)

c4.metric(
    "Enrolled Rate",
    f"{enrolled_rate:.2%}",
)


# ============================================================
# MODEL RESULT — APPEARS WHEN BUTTON IS CLICKED
# ============================================================

if predict_button:

    row = df.iloc[[student_index - 1]]

    (
        predicted_status,
        dropout_probability,
        risk_level,
        probability_dict,
    ) = predict_student(row, model)

    st.markdown(
        '<div class="section-title">'
        '🔮 Hasil Penerapan Model'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="prediction-box">',
        unsafe_allow_html=True,
    )

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Student Index",
        str(student_index),
    )

    p2.metric(
        "Predicted Status",
        predicted_status,
    )

    p3.metric(
        "Dropout Probability",
        f"{dropout_probability:.2%}",
    )

    if risk_level == "High Risk":
        risk_class = "risk-high"
    elif risk_level == "Medium Risk":
        risk_class = "risk-medium"
    else:
        risk_class = "risk-low"

    st.markdown(
        f"""
        <div class="{risk_class}">
            Risk Level<br>
            <span style="font-size:1.35rem">
                {risk_level}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "#### Probabilitas Prediksi"
    )

    probability_df = pd.DataFrame(
        {
            "Status": list(probability_dict.keys()),
            "Probability": list(
                probability_dict.values()
            ),
        }
    )

    fig_probability = px.bar(
        probability_df,
        x="Status",
        y="Probability",
        text="Probability",
        title="Probabilitas Setiap Status",
    )

    fig_probability.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside",
    )

    fig_probability.update_yaxes(
        tickformat=".0%",
        range=[0, 1],
    )

    fig_probability.update_layout(
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(
        fig_probability,
        use_container_width=True,
    )

    st.markdown(
        "#### Data Mahasiswa yang Dipilih"
    )

    summary_columns = [
        "Course",
        "Gender",
        "Age_at_enrollment",
        "Admission_grade",
        "Curricular_units_1st_sem_approved",
        "Curricular_units_1st_sem_grade",
        "Curricular_units_2nd_sem_approved",
        "Curricular_units_2nd_sem_grade",
        "Debtor",
        "Tuition_fees_up_to_date",
        "Scholarship_holder",
    ]

    summary_columns = [
        column
        for column in summary_columns
        if column in row.columns
    ]

    st.dataframe(
        row[summary_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# DASHBOARD CHARTS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Distribusi Status & Dropout'
    '</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.6])


# ------------------------------------------------------------
# STATUS DONUT
# ------------------------------------------------------------

with left:

    status_counts = (
        df["Status"]
        .value_counts()
        .reindex(
            ["Graduate", "Dropout", "Enrolled"],
            fill_value=0,
        )
        .reset_index()
    )

    status_counts.columns = [
        "Status",
        "Jumlah",
    ]

    fig_status = px.pie(
        status_counts,
        names="Status",
        values="Jumlah",
        hole=0.58,
        title="Distribusi Status Mahasiswa",
    )

    fig_status.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )

    fig_status.update_layout(
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(
        fig_status,
        use_container_width=True,
    )


# ------------------------------------------------------------
# DROPOUT RATE BY COURSE
# ------------------------------------------------------------

with right:

    course_summary = (
        df.groupby("Course")
        .agg(
            Total=("Status", "size"),
            Dropout=(
                "Status",
                lambda x: (x == "Dropout").sum(),
            ),
        )
        .reset_index()
    )

    course_summary["Dropout Rate"] = (
        course_summary["Dropout"]
        / course_summary["Total"]
    )

    # Hindari course dengan jumlah data terlalu sedikit.
    course_summary = course_summary[
        course_summary["Total"] >= 20
    ]

    course_summary = (
        course_summary
        .sort_values(
            "Dropout Rate",
            ascending=False,
        )
        .head(10)
    )

    fig_course = px.bar(
        course_summary.sort_values("Dropout Rate"),
        x="Dropout Rate",
        y="Course",
        orientation="h",
        title="Top 10 Course berdasarkan Dropout Rate",
        text="Dropout Rate",
    )

    fig_course.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
    )

    fig_course.update_xaxes(
        tickformat=".0%",
    )

    fig_course.update_layout(
        margin=dict(l=10, r=35, t=55, b=10),
        xaxis_title="Dropout Rate",
        yaxis_title="Course",
    )

    st.plotly_chart(
        fig_course,
        use_container_width=True,
    )


# ============================================================
# ACADEMIC & FINANCIAL
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Performa Akademik & Kondisi Finansial'
    '</div>',
    unsafe_allow_html=True,
)

a1, a2 = st.columns(2)


# ------------------------------------------------------------
# SEMESTER PERFORMANCE
# ------------------------------------------------------------

with a1:

    academic = (
        df.groupby("Status")[
            [
                "Curricular_units_1st_sem_grade",
                "Curricular_units_2nd_sem_grade",
            ]
        ]
        .mean()
        .reindex(
            ["Dropout", "Enrolled", "Graduate"]
        )
        .reset_index()
    )

    academic_long = academic.melt(
        id_vars="Status",
        var_name="Semester",
        value_name="Average Grade",
    )

    academic_long["Semester"] = (
        academic_long["Semester"]
        .replace(
            {
                "Curricular_units_1st_sem_grade":
                    "Semester 1",
                "Curricular_units_2nd_sem_grade":
                    "Semester 2",
            }
        )
    )

    fig_academic = px.bar(
        academic_long,
        x="Status",
        y="Average Grade",
        color="Semester",
        barmode="group",
        title="Perbandingan Performa Semester 1 vs Semester 2",
    )

    st.plotly_chart(
        fig_academic,
        use_container_width=True,
    )


# ------------------------------------------------------------
# TUITION PAYMENT
# ------------------------------------------------------------

with a2:

    payment = (
        df["Tuition_fees_up_to_date"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    payment.columns = [
        "Tuition",
        "Jumlah",
    ]

    payment["Status Pembayaran"] = (
        payment["Tuition"]
        .map(
            {
                1: "Up to Date",
                0: "Not Up to Date",
            }
        )
        .fillna(
            payment["Tuition"].astype(str)
        )
    )

    fig_payment = px.pie(
        payment,
        names="Status Pembayaran",
        values="Jumlah",
        hole=0.58,
        title="Status Pembayaran Tuition Fees",
    )

    fig_payment.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )

    st.plotly_chart(
        fig_payment,
        use_container_width=True,
    )


# ============================================================
# APPROVAL RATE + MODEL RISK DISTRIBUTION
# ============================================================

b1, b2 = st.columns(2)


# ------------------------------------------------------------
# APPROVAL RATE
# ------------------------------------------------------------

with b1:

    approval = (
        df.groupby("Status")
        .agg(
            S1_Approved=(
                "Curricular_units_1st_sem_approved",
                "sum",
            ),
            S1_Enrolled=(
                "Curricular_units_1st_sem_enrolled",
                "sum",
            ),
            S2_Approved=(
                "Curricular_units_2nd_sem_approved",
                "sum",
            ),
            S2_Enrolled=(
                "Curricular_units_2nd_sem_enrolled",
                "sum",
            ),
        )
        .reindex(
            ["Dropout", "Enrolled", "Graduate"]
        )
        .reset_index()
    )

    approval["Approval Rate S1"] = (
        approval["S1_Approved"]
        / approval["S1_Enrolled"].replace(0, np.nan)
    ).fillna(0)

    approval["Approval Rate S2"] = (
        approval["S2_Approved"]
        / approval["S2_Enrolled"].replace(0, np.nan)
    ).fillna(0)

    approval_long = approval.melt(
        id_vars="Status",
        value_vars=[
            "Approval Rate S1",
            "Approval Rate S2",
        ],
        var_name="Semester",
        value_name="Approval Rate",
    )

    fig_approval = px.bar(
        approval_long,
        x="Status",
        y="Approval Rate",
        color="Semester",
        barmode="group",
        title="Approval Rate Semester 1 vs Semester 2",
    )

    fig_approval.update_yaxes(
        tickformat=".0%",
        range=[0, 1],
    )

    st.plotly_chart(
        fig_approval,
        use_container_width=True,
    )


# ------------------------------------------------------------
# MODEL RISK
# ------------------------------------------------------------

with b2:

    # Prediksi hanya digunakan untuk menampilkan ringkasan
    # model pada dashboard.
    try:
        model_X = create_features(
            df.drop(
                columns=["Status"],
                errors="ignore",
            )
        )

        raw_predictions = model.predict(model_X)
        raw_probabilities = model.predict_proba(model_X)

        if np.issubdtype(
            np.asarray(model.classes_).dtype,
            np.number,
        ):
            label_map = {
                0: "Dropout",
                1: "Enrolled",
                2: "Graduate",
            }

            class_names = [
                label_map.get(
                    int(value),
                    str(value),
                )
                for value in model.classes_
            ]

        else:
            class_names = [
                str(value)
                for value in model.classes_
            ]

        proba_model = pd.DataFrame(
            raw_probabilities,
            columns=class_names,
        )

        dropout_probability_all = (
            proba_model["Dropout"]
            if "Dropout" in proba_model.columns
            else pd.Series(
                np.zeros(len(df))
            )
        )

        risk_all = np.select(
            [
                dropout_probability_all >= 0.70,
                dropout_probability_all >= 0.40,
            ],
            [
                "High Risk",
                "Medium Risk",
            ],
            default="Low Risk",
        )

        risk_counts = (
            pd.Series(risk_all)
            .value_counts()
            .reindex(
                [
                    "High Risk",
                    "Medium Risk",
                    "Low Risk",
                ],
                fill_value=0,
            )
            .reset_index()
        )

        risk_counts.columns = [
            "Risk Level",
            "Jumlah",
        ]

        fig_risk = px.pie(
            risk_counts,
            names="Risk Level",
            values="Jumlah",
            hole=0.58,
            title="Distribusi Risk Level Model",
        )

        fig_risk.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True,
        )

    except Exception as error:
        st.warning(
            f"Distribusi risiko model tidak dapat ditampilkan: {error}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Jaya Jaya Institut — Student Dropout Analytics & "
    "Machine Learning Prototype"
)

st.caption(
    "Student Index merupakan nomor referensi baris dataset, "
    "bukan Student ID asli."
)
