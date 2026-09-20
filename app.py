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
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Data" / "data.csv"
MODEL_PATH = BASE_DIR / "student_dropout_model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.pkl"


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #F5F8F7 0%, #F8FAFC 45%, #FFFFFF 100%);
    }

    section[data-testid="stSidebar"] {
        display: none !important;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        background: linear-gradient(135deg, #123C35 0%, #176052 55%, #1E7765 100%);
        border-radius: 22px;
        padding: 30px 34px;
        margin-bottom: 18px;
        box-shadow: 0 12px 30px rgba(18, 60, 53, 0.16);
        color: white;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 850;
        line-height: 1.15;
        margin: 0 0 8px 0;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #DCEDE8;
        font-size: 1rem;
        line-height: 1.55;
        margin: 0;
        max-width: 900px;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 15px;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.22);
        color: #FFFFFF;
        font-size: 0.82rem;
        font-weight: 700;
    }

    .section-title {
        color: #123C35;
        font-size: 1.22rem;
        font-weight: 800;
        margin-top: 1.35rem;
        margin-bottom: 0.65rem;
    }

    .section-caption {
        color: #64736F;
        font-size: 0.9rem;
        margin-top: -0.3rem;
        margin-bottom: 0.8rem;
    }

    .search-note {
        background: #EAF5F1;
        border: 1px solid #CFE6DE;
        border-radius: 12px;
        padding: 12px 15px;
        color: #315A50;
        font-size: 0.88rem;
        line-height: 1.45;
        margin-top: 0.45rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.96);
        border: 1px solid #E2EAE7;
        border-radius: 16px;
        padding: 1rem 1.05rem;
        box-shadow: 0 5px 16px rgba(18,60,53,0.06);
        min-height: 108px;
    }

    div[data-testid="stMetricLabel"] {
        color: #64736F !important;
        font-weight: 650 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #123C35 !important;
        font-weight: 800 !important;
    }

    .model-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F4FAF8 100%);
        border: 1px solid #D8E8E2;
        border-radius: 18px;
        padding: 18px 20px;
        margin: 8px 0 14px 0;
        box-shadow: 0 7px 20px rgba(18,60,53,0.05);
    }

    .model-title {
        color: #123C35;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .model-note {
        color: #5D6D68;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .ready-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: #E5F6ED;
        color: #21613D;
        font-size: 0.78rem;
        font-weight: 800;
        margin-top: 8px;
    }

    .prediction-box {
        background: #FFFFFF;
        border: 1px solid #DDE8E4;
        border-radius: 18px;
        padding: 20px;
        margin: 10px 0 22px 0;
        box-shadow: 0 8px 24px rgba(18,60,53,0.07);
    }

    .risk-high {
        background-color: #FDECEC;
        border: 1px solid #F2B8B5;
        padding: 0.85rem;
        border-radius: 12px;
        color: #9C2F2B;
        font-weight: 750;
        text-align: center;
        margin-top: 12px;
    }

    .risk-medium {
        background-color: #FFF5DF;
        border: 1px solid #F1D18A;
        padding: 0.85rem;
        border-radius: 12px;
        color: #8A5A00;
        font-weight: 750;
        text-align: center;
        margin-top: 12px;
    }

    .risk-low {
        background-color: #EAF7EF;
        border: 1px solid #B7DEC2;
        padding: 0.85rem;
        border-radius: 12px;
        color: #22643A;
        font-weight: 750;
        text-align: center;
        margin-top: 12px;
    }

    .action-card {
        background: #F7FAF9;
        border: 1px solid #D9E7E2;
        border-radius: 16px;
        padding: 16px 18px;
        margin: 12px 0 18px 0;
        box-shadow: 0 5px 16px rgba(18,60,53,0.05);
    }

    .action-title {
        color: #123C35;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .action-note {
        color: #5D6D68;
        font-size: 0.86rem;
        line-height: 1.5;
    }

    .footer-card {
        background: #123C35;
        color: #DDEBE7;
        border-radius: 16px;
        padding: 16px 20px;
        margin-top: 24px;
        text-align: center;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    .stButton > button {
        border-radius: 12px;
        min-height: 44px;
        font-weight: 750;
        border: 0;
        box-shadow: 0 5px 14px rgba(18,60,53,0.12);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E0E8E5;
    }

    hr {
        border-color: #DCE6E2 !important;
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
        return None

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    # Pastikan file yang dimuat benar-benar model binary,
    # bukan model_metadata.pkl.
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError(
            "student_dropout_model.pkl bukan objek model Machine Learning "
            "yang valid. Jangan gunakan model_metadata.pkl sebagai model."
        )

    if not hasattr(model, "classes_"):
        raise TypeError("Model tidak memiliki atribut classes_.")

    classes = set(np.asarray(model.classes_).tolist())
    if classes != {0, 1}:
        raise ValueError(
            f"Model harus binary classification dengan classes [0, 1], "
            f"tetapi ditemukan: {sorted(classes)}"
        )

    return model


@st.cache_data(show_spinner=False)
def load_metadata():
    if not METADATA_PATH.exists():
        return {}

    with open(METADATA_PATH, "rb") as file:
        metadata = pickle.load(file)

    return metadata if isinstance(metadata, dict) else {}


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
# DISPLAY LABELS
# ============================================================

def map_display_labels(dataframe):
    data = dataframe.copy()

    mappings = {
        "Gender": {
            0: "Perempuan",
            1: "Laki-laki"
        },
        "Debtor": {
            0: "Tidak Berutang",
            1: "Berutang"
        },
        "Tuition_fees_up_to_date": {
            0: "Not Up to Date",
            1: "Up to Date"
        },
        "Scholarship_holder": {
            0: "Bukan Penerima Beasiswa",
            1: "Penerima Beasiswa"
        },
    }

    for column, mapping in mappings.items():
        if column in data.columns:
            data[column] = data[column].map(mapping).fillna("Tidak Diketahui")

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

    # Mapping model binary:
    # 0 = Graduate, 1 = Dropout
    if np.issubdtype(np.asarray(model.classes_).dtype, np.number):
        label_map = {
            0: "Graduate",
            1: "Dropout",
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
# ACTION ITEMS BERDASARKAN HASIL ANALISIS
# ============================================================

def build_action_items(row, risk_level):
    """
    Menyusun action items yang spesifik berdasarkan:
    1) Risk Level hasil model,
    2) indikator akademik yang menjadi temuan utama,
    3) indikator pembayaran/finansial yang ditemukan pada EDA.

    Perbandingan akademik menggunakan median dataset agar tidak memakai
    threshold angka yang dibuat secara arbitrer.
    """
    engineered = create_features(row)

    s1_approval = float(engineered["Approval_Rate_1st_Sem"].iloc[0])
    s2_approval = float(engineered["Approval_Rate_2nd_Sem"].iloc[0])
    avg_grade = float(engineered["Average_Semester_Grade"].iloc[0])

    # Baseline dataset untuk menentukan apakah indikator mahasiswa relatif rendah.
    reference = create_features(df.drop(columns=["Status"], errors="ignore"))
    s1_median = float(reference["Approval_Rate_1st_Sem"].median())
    s2_median = float(reference["Approval_Rate_2nd_Sem"].median())
    grade_median = float(reference["Average_Semester_Grade"].median())

    academic_flags = []
    if s2_approval < s2_median:
        academic_flags.append(
            f"Approval Rate Semester 2 ({s2_approval:.1%}) berada di bawah "
            f"median dataset ({s2_median:.1%})."
        )
    if s1_approval < s1_median:
        academic_flags.append(
            f"Approval Rate Semester 1 ({s1_approval:.1%}) berada di bawah "
            f"median dataset ({s1_median:.1%})."
        )
    if avg_grade < grade_median:
        academic_flags.append(
            f"Rata-rata nilai dua semester ({avg_grade:.2f}) berada di bawah "
            f"median dataset ({grade_median:.2f})."
        )

    debtor = int(row["Debtor"].iloc[0]) if "Debtor" in row.columns else 0
    tuition = int(row["Tuition_fees_up_to_date"].iloc[0]) if "Tuition_fees_up_to_date" in row.columns else 1
    scholarship = int(row["Scholarship_holder"].iloc[0]) if "Scholarship_holder" in row.columns else 1

    financial_flags = []
    if tuition == 0:
        financial_flags.append(
            "Status pembayaran tuition fees belum Up to Date."
        )
    if debtor == 1:
        financial_flags.append(
            "Mahasiswa tercatat memiliki status Berutang."
        )
    if scholarship == 0:
        financial_flags.append(
            "Mahasiswa bukan penerima beasiswa; kondisi finansial perlu "
            "diverifikasi bila terdapat kendala pembayaran."
        )

    actions = []

    if risk_level == "High Risk":
        actions.append(
            "PRIORITAS 1 — Lakukan verifikasi individual oleh Dosen PA/Program "
            "Studi dan hubungi mahasiswa untuk mengonfirmasi penyebab risiko "
            "sebelum menentukan intervensi."
        )
    elif risk_level == "Medium Risk":
        actions.append(
            "PRIORITAS 2 — Lakukan monitoring terjadwal oleh Dosen PA/Program "
            "Studi dan evaluasi ulang indikator akademik agar risiko tidak meningkat."
        )
    else:
        actions.append(
            "PRIORITAS 3 — Lanjutkan monitoring rutin oleh pihak akademik; "
            "belum diperlukan intervensi intensif hanya berdasarkan model."
        )

    if academic_flags:
        actions.append(
            "Intervensi akademik: evaluasi mata kuliah yang belum disetujui, "
            "tawarkan tutoring/pendampingan belajar, dan susun rencana studi "
            "yang lebih terarah."
        )

    if financial_flags:
        actions.append(
            "Verifikasi finansial: koordinasikan dengan Bagian Keuangan/ "
            "Kemahasiswaan untuk mengecek kendala pembayaran dan, bila sesuai "
            "kebijakan, informasikan opsi beasiswa, keringanan, atau cicilan."
        )

    if not academic_flags and not financial_flags:
        actions.append(
            "Lakukan pengecekan akademik dan administratif secara berkala "
            "sebagai langkah pencegahan."
        )

    return academic_flags, financial_flags, actions


# ============================================================
# LOAD
# ============================================================

try:
    df = load_data()
    model = load_model()
    model_metadata = load_metadata()

    # Data Enrolled tidak digunakan untuk training.
    # Data ini digunakan sebagai data screening/prediksi.
    df_enrolled = (
        df[df["Status"] == "Enrolled"]
        .copy()
        .reset_index(drop=True)
    )

except Exception as error:
    st.error("Aplikasi gagal memuat data.")
    st.exception(error)
    st.stop()

def get_valid_options(column, fallback):
    """Ambil kategori yang benar-benar ada di dataset agar input tidak OOD."""
    if column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna().unique().tolist()
        values = sorted(values)
        if values:
            return values
    return fallback

def get_numeric_bounds(column, fallback_min, fallback_max, fallback_value):
    """Gunakan rentang dataset untuk mencegah input yang tidak realistis."""
    if column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(values):
            low = float(values.min())
            high = float(values.max())
            default = float(values.median())
            return low, high, default
    return fallback_min, fallback_max, fallback_value


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🎓 Dashboard Analisis & Prediksi Dropout Mahasiswa</div>
        <p class="hero-subtitle">
            Monitoring status mahasiswa, performa akademik, dan penerapan
            Machine Learning untuk membantu proses screening dropout.
        </p>
        <span class="hero-badge">Binary Classification • Graduate vs Dropout</span>
    </div>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# PREDICTION / SCREENING
# ============================================================

st.markdown(
    '<div class="section-title">🔎 Screening & Prediksi Mahasiswa</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-caption">'
    'Prototype menyediakan screening data Enrolled dan prediksi langsung '
    'untuk mahasiswa baru.'
    '</div>',
    unsafe_allow_html=True,
)

tab_existing, tab_new = st.tabs([
    "🎓 Screening Mahasiswa Enrolled",
    "📝 Prediksi Mahasiswa",
])

# ------------------------------------------------------------
# EXISTING ENROLLED STUDENT
# ------------------------------------------------------------

with tab_existing:
    st.markdown(
        f'<div class="search-note"><strong>{len(df_enrolled):,} mahasiswa Enrolled</strong> '
        'tersedia untuk screening. Student Index adalah nomor referensi baris pada '
        'data Enrolled, bukan Student ID asli.</div>',
        unsafe_allow_html=True,
    )

    if len(df_enrolled) > 0:
        search_col, button_col = st.columns([2.2, 1.2])

        with search_col:
            student_index = st.number_input(
                "Student Index",
                min_value=1,
                max_value=len(df_enrolled),
                value=1,
                step=1,
                help="Pilih nomor referensi mahasiswa Enrolled.",
                key="existing_student_index",
            )

        with button_col:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            predict_existing_button = st.button(
                "🔮 Terapkan Model",
                type="primary",
                use_container_width=True,
                key="predict_existing",
            )
    else:
        student_index = 1
        predict_existing_button = False
        st.warning("Tidak terdapat mahasiswa dengan status Enrolled.")

# ------------------------------------------------------------
# NEW STUDENT PREDICTION
# ------------------------------------------------------------

with tab_new:
    st.info(
        "Masukkan data mahasiswa yang ingin discreening. Data tidak harus "
        "sudah ada di dataset. Kolom Status tidak diinput karena merupakan "
        "target yang diprediksi model. Prediksi menggunakan data akademik "
        "semester 1 dan semester 2 yang tersedia."
    )

    with st.expander("👤 Data Demografi & Pendaftaran", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            marital_options = get_valid_options("Marital_status", [1, 2, 3, 4, 5, 6])
            appmode_options = get_valid_options("Application_mode", list(range(1, 58)))
            apporder_options = get_valid_options("Application_order", list(range(0, 10)))
            course_options = get_valid_options("Course", [1])

            new_marital = st.selectbox("Marital status", marital_options, key="n_marital")
            new_application_mode = st.selectbox("Application mode", appmode_options, key="n_appmode")
            new_application_order = st.selectbox("Application order", apporder_options, key="n_apporder")
            new_course = st.selectbox("Course", course_options, key="n_course")
            new_daytime = st.selectbox(
                "Daytime/evening attendance", [1, 0],
                format_func=lambda x: "Siang" if x == 1 else "Malam",
                key="n_daytime",
            )
            prevqual_options = get_valid_options("Previous_qualification", [1])
            new_previous_qualification = st.selectbox(
                "Previous qualification", prevqual_options, key="n_prevqual"
            )

        with c2:
            pg_min, pg_max, pg_default = get_numeric_bounds(
                "Previous_qualification_grade", 0.0, 200.0, 120.0
            )
            new_previous_grade = st.number_input(
                "Previous qualification grade", min_value=pg_min, max_value=pg_max,
                value=pg_default, step=0.1, key="n_prevgrade"
            )

            nationality_options = get_valid_options("Nacionality", [1])
            motherqual_options = get_valid_options("Mothers_qualification", [1])
            fatherqual_options = get_valid_options("Fathers_qualification", [1])
            motherocc_options = get_valid_options("Mothers_occupation", [0])
            fatherocc_options = get_valid_options("Fathers_occupation", [0])

            new_nationality = st.selectbox("Nacionality", nationality_options, key="n_nationality")
            new_mother_qualification = st.selectbox(
                "Mother's qualification", motherqual_options, key="n_motherqual"
            )
            new_father_qualification = st.selectbox(
                "Father's qualification", fatherqual_options, key="n_fatherqual"
            )
            new_mother_occupation = st.selectbox(
                "Mother's occupation", motherocc_options, key="n_motherocc"
            )
            new_father_occupation = st.selectbox(
                "Father's occupation", fatherocc_options, key="n_fatherocc"
            )

        with c3:
            adm_min, adm_max, adm_default = get_numeric_bounds(
                "Admission_grade", 0.0, 200.0, 120.0
            )
            new_admission_grade = st.number_input(
                "Admission grade", min_value=adm_min, max_value=adm_max,
                value=adm_default, step=0.1, key="n_admission"
            )
            age_min, age_max, age_default = get_numeric_bounds(
                "Age_at_enrollment", 15, 80, 20
            )
            new_age = st.number_input(
                "Age at enrollment", min_value=int(age_min), max_value=int(age_max),
                value=int(round(age_default)), step=1, key="n_age"
            )
            new_gender = st.selectbox(
                "Gender", [0, 1],
                format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki",
                key="n_gender",
            )
            new_displaced = st.selectbox(
                "Displaced", [0, 1],
                format_func=lambda x: "Tidak" if x == 0 else "Ya",
                key="n_displaced",
            )
            new_special_needs = st.selectbox(
                "Educational special needs", [0, 1],
                format_func=lambda x: "Tidak" if x == 0 else "Ya",
                key="n_special",
            )
            new_international = st.selectbox(
                "International", [0, 1],
                format_func=lambda x: "Tidak" if x == 0 else "Ya",
                key="n_international",
            )

    with st.expander("💳 Kondisi Finansial", expanded=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            new_debtor = st.selectbox(
                "Status Utang", [0, 1],
                format_func=lambda x: "Tidak Berutang" if x == 0 else "Berutang",
                key="n_debtor",
            )
        with f2:
            new_tuition = st.selectbox(
                "Status Tuition Fees", [1, 0],
                format_func=lambda x: "Up to Date" if x == 1 else "Not Up to Date",
                key="n_tuition",
            )
        with f3:
            new_scholarship = st.selectbox(
                "Beasiswa", [0, 1],
                format_func=lambda x: "Bukan Penerima Beasiswa" if x == 0 else "Penerima Beasiswa",
                key="n_scholarship",
            )

    with st.expander("📚 Performa Akademik Semester 1", expanded=True):
        s1a, s1b = st.columns(2)
        with s1a:
            new_s1_credited = st.number_input("S1 Credited", 0, 50, 0, key="n_s1cred")
            new_s1_enrolled = st.number_input("S1 Enrolled", 0, 50, 6, key="n_s1enrolled")
            new_s1_evaluations = st.number_input("S1 Evaluations", 0, 50, 6, key="n_s1eval")
        with s1b:
            new_s1_approved = st.number_input("S1 Approved", 0, 50, 6, key="n_s1approved")
            new_s1_grade = st.number_input("S1 Grade", 0.0, 20.0, 12.0, key="n_s1grade")
            new_s1_without = st.number_input("S1 Without Evaluation", 0, 50, 0, key="n_s1without")

    with st.expander("📚 Performa Akademik Semester 2", expanded=True):
        s2a, s2b = st.columns(2)
        with s2a:
            new_s2_credited = st.number_input("S2 Credited", 0, 50, 0, key="n_s2cred")
            new_s2_enrolled = st.number_input("S2 Enrolled", 0, 50, 6, key="n_s2enrolled")
            new_s2_evaluations = st.number_input("S2 Evaluations", 0, 50, 6, key="n_s2eval")
        with s2b:
            new_s2_approved = st.number_input("S2 Approved", 0, 50, 6, key="n_s2approved")
            new_s2_grade = st.number_input("S2 Grade", 0.0, 20.0, 12.0, key="n_s2grade")
            new_s2_without = st.number_input("S2 Without Evaluation", 0, 50, 0, key="n_s2without")

    with st.expander("🌍 Indikator Ekonomi", expanded=False):
        e1, e2, e3 = st.columns(3)
        with e1:
            new_unemployment = st.number_input("Unemployment rate", -20.0, 50.0, 10.0, key="n_unemployment")
        with e2:
            new_inflation = st.number_input("Inflation rate", -20.0, 50.0, 1.0, key="n_inflation")
        with e3:
            new_gdp = st.number_input("GDP", -20.0, 50.0, 1.0, key="n_gdp")

    predict_new_button = st.button(
        "🚀 Prediksi Dropout",
        type="primary",
        use_container_width=True,
        key="predict_new",
    )

    if predict_new_button:
        new_student = pd.DataFrame([{
            "Marital_status": new_marital,
            "Application_mode": new_application_mode,
            "Application_order": new_application_order,
            "Course": new_course,
            "Daytime_evening_attendance": new_daytime,
            "Previous_qualification": new_previous_qualification,
            "Previous_qualification_grade": new_previous_grade,
            "Nacionality": new_nationality,
            "Mothers_qualification": new_mother_qualification,
            "Fathers_qualification": new_father_qualification,
            "Mothers_occupation": new_mother_occupation,
            "Fathers_occupation": new_father_occupation,
            "Admission_grade": new_admission_grade,
            "Displaced": new_displaced,
            "Educational_special_needs": new_special_needs,
            "Debtor": new_debtor,
            "Tuition_fees_up_to_date": new_tuition,
            "Gender": new_gender,
            "Scholarship_holder": new_scholarship,
            "Age_at_enrollment": new_age,
            "International": new_international,
            "Curricular_units_1st_sem_credited": new_s1_credited,
            "Curricular_units_1st_sem_enrolled": new_s1_enrolled,
            "Curricular_units_1st_sem_evaluations": new_s1_evaluations,
            "Curricular_units_1st_sem_approved": new_s1_approved,
            "Curricular_units_1st_sem_grade": new_s1_grade,
            "Curricular_units_1st_sem_without_evaluations": new_s1_without,
            "Curricular_units_2nd_sem_credited": new_s2_credited,
            "Curricular_units_2nd_sem_enrolled": new_s2_enrolled,
            "Curricular_units_2nd_sem_evaluations": new_s2_evaluations,
            "Curricular_units_2nd_sem_approved": new_s2_approved,
            "Curricular_units_2nd_sem_grade": new_s2_grade,
            "Curricular_units_2nd_sem_without_evaluations": new_s2_without,
            "Unemployment_rate": new_unemployment,
            "Inflation_rate": new_inflation,
            "GDP": new_gdp,
        }])

        if model is None:
            st.error(
                "Model belum tersedia. Upload `student_dropout_model.pkl` "
                "ke repository bersama `app.py`."
            )
        else:
            (
                predicted_status,
                dropout_probability,
                risk_level,
                probability_dict,
            ) = predict_student(new_student, model)

            st.markdown(
                '<div class="section-title">🔮 Hasil Prediksi Dropout</div>',
                unsafe_allow_html=True,
            )
            st.success(
                "Prediksi berhasil dijalankan menggunakan model binary "
                "Graduate vs Dropout."
            )

            dropout_indicated = predicted_status == "Dropout"
            prediction_label = (
                "TERINDIKASI DROPOUT" if dropout_indicated
                else "TIDAK TERINDIKASI DROPOUT"
            )
            confidence = max(probability_dict.values())

            if dropout_indicated:
                st.error(
                    f"⚠️ {prediction_label} — probabilitas dropout "
                    f"{dropout_probability:.2%}."
                )
            else:
                st.success(
                    f"✅ {prediction_label} — probabilitas dropout "
                    f"{dropout_probability:.2%}."
                )

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Indikasi Dropout", prediction_label)
            r2.metric("Probabilitas Dropout", f"{dropout_probability:.2%}")
            r3.metric("Confidence", f"{confidence:.2%}")
            r4.metric("Risk Level", risk_level)

            st.caption(
                "Klasifikasi biner menggunakan ambang probabilitas model 50%. "
                "Risk Level: Low <40%, Medium 40–69%, High ≥70%. "
                "Hasil merupakan screening berbasis data, bukan keputusan akhir."
            )

            academic_flags, financial_flags, action_items = build_action_items(
                new_student,
                risk_level
            )

            st.markdown("#### 🎯 Action Items yang Disarankan")
            st.markdown('<div class="action-card">', unsafe_allow_html=True)

            if academic_flags:
                st.markdown("**Dasar akademik:**")
                for item in academic_flags:
                    st.markdown(f"- {item}")

            if financial_flags:
                st.markdown("**Dasar administratif/finansial:**")
                for item in financial_flags:
                    st.markdown(f"- {item}")

            st.markdown("**Tindakan prioritas:**")
            for item in action_items:
                st.markdown(f"- {item}")

            st.caption(
                "Action items mengacu pada risk level model dan indikator yang "
                "teridentifikasi dari data. Feature importance dan dropout rate "
                "bersifat deskriptif/interpretatif, bukan bukti sebab-akibat."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            probability_df = pd.DataFrame({
                "Status": list(probability_dict.keys()),
                "Probability": list(probability_dict.values()),
            })

            fig_new = px.bar(
                probability_df,
                x="Status",
                y="Probability",
                text="Probability",
                title="Probabilitas Prediksi Mahasiswa",
            )
            fig_new.update_traces(
                texttemplate="%{text:.2%}",
                textposition="outside",
            )
            fig_new.update_yaxes(tickformat=".0%", range=[0, 1])
            fig_new.update_layout(
                template="plotly_white",
                margin=dict(l=10, r=10, t=55, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_new, use_container_width=True)

# ------------------------------------------------------------
# RESULT FOR EXISTING ENROLLED STUDENT
# ------------------------------------------------------------

if predict_existing_button:
    if model is None:
        st.warning(
            "Model binary belum tersedia. Letakkan `student_dropout_model.pkl` "
            "di folder yang sama dengan `app.py`."
        )
        st.stop()

    row = df_enrolled.iloc[[student_index - 1]]

    (
        predicted_status,
        dropout_probability,
        risk_level,
        probability_dict,
    ) = predict_student(row, model)

    st.markdown(
        '<div class="section-title">🔮 Hasil Screening Mahasiswa Enrolled</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Hasil screening menggunakan data akademik yang tersedia. "
        "Prediksi perlu diverifikasi oleh pihak institusi dan bukan keputusan akhir."
    )

    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

    dropout_indicated = predicted_status == "Dropout"
    prediction_label = (
        "TERINDIKASI DROPOUT" if dropout_indicated
        else "TIDAK TERINDIKASI DROPOUT"
    )
    confidence = max(probability_dict.values())

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Student Index", str(student_index))
    p2.metric("Indikasi Dropout", prediction_label)
    p3.metric("Probabilitas Dropout", f"{dropout_probability:.2%}")
    p4.metric("Confidence", f"{confidence:.2%}")

    if risk_level == "High Risk":
        risk_class = "risk-high"
    elif risk_level == "Medium Risk":
        risk_class = "risk-medium"
    else:
        risk_class = "risk-low"

    st.markdown(
        f'<div class="{risk_class}">Risk Level<br>'
        f'<span style="font-size:1.35rem">{risk_level}</span></div>',
        unsafe_allow_html=True,
    )

    academic_flags, financial_flags, action_items = build_action_items(
        row,
        risk_level
    )

    st.markdown("#### 🎯 Action Items yang Disarankan")
    st.markdown('<div class="action-card">', unsafe_allow_html=True)

    if academic_flags:
        st.markdown("**Dasar akademik:**")
        for item in academic_flags:
            st.markdown(f"- {item}")

    if financial_flags:
        st.markdown("**Dasar administratif/finansial:**")
        for item in financial_flags:
            st.markdown(f"- {item}")

    st.markdown("**Tindakan prioritas:**")
    for item in action_items:
        st.markdown(f"- {item}")

    st.caption(
        "Action items mengacu pada risk level model dan indikator yang "
        "teridentifikasi dari data. Feature importance dan dropout rate "
        "bersifat deskriptif/interpretatif, bukan bukti sebab-akibat."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Probabilitas Prediksi")

    probability_df = pd.DataFrame({
        "Status": list(probability_dict.keys()),
        "Probability": list(probability_dict.values()),
    })

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
    fig_probability.update_yaxes(tickformat=".0%", range=[0, 1])
    fig_probability.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_probability, use_container_width=True)

    st.markdown("#### Data Mahasiswa yang Dipilih")

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
        column for column in summary_columns if column in row.columns
    ]

    st.dataframe(
        map_display_labels(row[summary_columns]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)

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

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Mahasiswa", f"{total:,}")
c2.metric("Jumlah Dropout", f"{dropout:,}")
c3.metric("Dropout Rate", f"{dropout_rate:.2%}")
c4.metric("Graduate Rate", f"{graduate_rate:.2%}")
c5.metric("Enrolled Rate", f"{enrolled_rate:.2%}")


# ============================================================
# MACHINE LEARNING PROTOTYPE
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Prototype Machine Learning</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '''
    <div class="model-card">
        <div class="model-title">🤖 Binary Dropout Screening</div>
        <div class="model-note">
            Prototype menggunakan model klasifikasi biner untuk membedakan
            <strong>Graduate</strong> dan <strong>Dropout</strong>.
            Data <strong>Enrolled</strong> tidak digunakan sebagai kelas training,
            tetapi disimpan terpisah untuk proses screening/prediksi.
        </div>
        <span class="ready-badge">● Model siap digunakan</span>
    </div>
    ''',
    unsafe_allow_html=True,
)

if model is not None:
    st.success(
        "Model `student_dropout_model.pkl` berhasil dimuat dan siap melakukan prediksi."
    )
    info_cols = st.columns(4)
    info_cols[0].metric(
        "Model",
        model_metadata.get("model_name", "XGBoost")
    )
    info_cols[1].metric(
        "Accuracy",
        f"{float(model_metadata.get('accuracy_test', 0)):.2%}"
    )
    info_cols[2].metric(
        "Precision Dropout",
        f"{float(model_metadata.get('dropout_precision', 0)):.2%}"
    )
    info_cols[3].metric(
        "Recall Dropout",
        f"{float(model_metadata.get('dropout_recall', 0)):.2%}"
    )
else:
    st.info(
        "Model belum tersedia. Tambahkan `student_dropout_model.pkl` "
        "hasil training binary ke repository untuk mengaktifkan screening."
    )


# ============================================================
# DASHBOARD CHARTS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Distribusi Status & Dropout</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-caption">Ringkasan status mahasiswa dan variasi dropout berdasarkan course.</div>',
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
        template="plotly_white",
        margin=dict(l=10, r=10, t=55, b=10),
        legend_title_text="Status",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
        template="plotly_white",
        margin=dict(l=10, r=35, t=55, b=10),
        xaxis_title="Dropout Rate",
        yaxis_title="Course",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig_course,
        use_container_width=True,
    )


# ============================================================
# ACADEMIC & FINANCIAL
# ============================================================

st.markdown(
    '<div class="section-title">📚 Performa Akademik & Kondisi Finansial</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-caption">Perbandingan performa akademik serta kondisi pembayaran dan finansial mahasiswa.</div>',
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
        text_auto=".1f",
    )

    fig_academic.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
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
        .map({
            1: "Up to Date",
            0: "Not Up to Date",
        })
        .value_counts()
        .reindex(
            ["Up to Date", "Not Up to Date"],
            fill_value=0,
        )
        .reset_index()
    )

    payment.columns = [
        "Status Pembayaran",
        "Jumlah",
    ]

    fig_payment = px.pie(
        payment,
        names="Status Pembayaran",
        values="Jumlah",
        hole=0.58,
        title="Status Pembayaran Tuition Fees",
    )

    fig_payment.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
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
# FINANCIAL STATUS & APPROVAL RATE
# ============================================================

f1, f2 = st.columns(2)

# ------------------------------------------------------------
# FINANCIAL STATUS
# ------------------------------------------------------------

with f1:

    st.markdown(
        '<div class="section-title">'
        'Status Finansial Mahasiswa'
        '</div>',
        unsafe_allow_html=True,
    )

    financial = df["Debtor"].map({
        0: "Tidak Berutang",
        1: "Berutang",
    })

    financial_counts = (
        financial
        .value_counts()
        .reindex(
            ["Tidak Berutang", "Berutang"],
            fill_value=0,
        )
        .reset_index()
    )

    financial_counts.columns = [
        "Status Finansial",
        "Jumlah",
    ]

    fig_financial = px.bar(
        financial_counts,
        x="Status Finansial",
        y="Jumlah",
        text="Jumlah",
        title="Status Finansial Mahasiswa",
    )

    fig_financial.update_traces(
        textposition="outside",
    )

    fig_financial.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="Status Finansial",
        yaxis_title="Jumlah Mahasiswa",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig_financial,
        use_container_width=True,
    )


# ------------------------------------------------------------
# APPROVAL RATE
# ------------------------------------------------------------

with f2:

    st.markdown(
        '<div class="section-title">'
        'Approval Rate Semester 1 vs Semester 2'
        '</div>',
        unsafe_allow_html=True,
    )

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

    fig_approval.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="Status Mahasiswa",
        yaxis_title="Approval Rate",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig_approval,
        use_container_width=True,
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-card">
        <strong>🎓 Jaya Jaya Institut</strong><br>
        Student Dropout Analytics & Binary Dropout Screening Prototype<br>
        <span style="opacity:0.8;">
            Student Index merupakan nomor referensi baris pada data Enrolled,
            bukan Student ID asli.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
