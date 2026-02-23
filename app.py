import os
# TensorFlow ve Protobuf çakışmasını engellemek için
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Meme Kanseri Teşhis Sistemi (Modern)",
    page_icon="🧬",
    layout="wide"
)

# --- GÜVENLİ IMPORT SİSTEMİ ---
TF_AVAILABLE = False
IMPORT_ERROR = None

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception as e:
    IMPORT_ERROR = str(e)

st.title("🧬 Meme Kanseri Hayatta Kalma Tahmin Sistemi")
st.markdown("---")

# --- MODEL YOLLARI ---
BASE_DIR = os.path.dirname(__file__)
ML_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
DL_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'dl_model.keras')
LE_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')

# --- MODERM CACHE MEKANİZMASI ---
# Streamlit >= 1.18.0 st.cache_resource kullanır, aksi halde eski yönteme döner
@st.cache_resource if hasattr(st, "cache_resource") else st.experimental_singleton
def load_all_assets():
    ml_pipe = None
    dl_mod = None
    le_mod = None
    
    if os.path.exists(ML_MODEL_PATH):
        ml_pipe = joblib.load(ML_MODEL_PATH)
    
    if TF_AVAILABLE and os.path.exists(DL_MODEL_PATH):
        try:
            # En yeni Keras yükleme yöntemi
            dl_mod = tf.keras.models.load_model(DL_MODEL_PATH)
        except Exception as e:
            st.error(f"DL Yükleme Hatası: {e}")
            
    if os.path.exists(LE_PATH):
        le_mod = joblib.load(LE_PATH)
        
    return ml_pipe, dl_mod, le_mod

ml_pipeline, dl_model, le = load_all_assets()

# --- SIDEBAR GİRİŞLERİ ---
st.sidebar.header("📋 Hasta Giriş Formu")

def user_input_features():
    age = st.sidebar.number_input("Yaş", 20, 100, 50)
    gender = st.sidebar.selectbox("Cinsiyet", ["FEMALE", "MALE"])
    p1 = st.sidebar.number_input("Protein 1", value=0.0)
    p2 = st.sidebar.number_input("Protein 2", value=1.0)
    p3 = st.sidebar.number_input("Protein 3", value=0.0)
    p4 = st.sidebar.number_input("Protein 4", value=0.0)
    stage = st.sidebar.selectbox("Evre", ["I", "II", "III"])
    hist = st.sidebar.selectbox("Histoloji", ["Infiltrating Ductal Carcinoma", "Infiltrating Lobular Carcinoma", "Mucinous Carcinoma"])
    er = st.sidebar.selectbox("ER", ["Positive", "Negative"])
    pr = st.sidebar.selectbox("PR", ["Positive", "Negative"])
    her2 = st.sidebar.selectbox("HER2", ["Negative", "Positive"])
    surgery = st.sidebar.selectbox("Ameliyat", ["Modified Radical Mastectomy", "Simple Mastectomy", "Lumpectomy", "Other"])

    data = {
        'age': age, 'gender': gender,
        'protein1': p1, 'protein2': p2, 'protein3': p3, 'protein4': p4,
        'tumour_stage': stage, 'histology': hist,
        'er_status': er, 'pr_status': pr, 'her2_status': her2,
        'surgery_type': surgery
    }
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("🧐 İncelenen Hasta Verisi")
st.dataframe(input_df)

if st.button("🚀 Tahminleri Üret"):
    if ml_pipeline is None:
        st.error("ML Modeli bulunamadı!")
    else:
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        with c1:
            st.info("🤖 ML Sonucu")
            pred = ml_pipeline.predict(input_df)[0]
            res = le.inverse_transform([pred])[0] if le else str(pred)
            st.success(f"**{res}**")
            
            # Güven Skoru Hesaplama (ML)
            if hasattr(ml_pipeline, "predict_proba"):
                proba = ml_pipeline.predict_proba(input_df)[0]
                confidence = np.max(proba)
                st.metric(label="Güven Skoru", value=f"%{confidence*100:.2f}")

        with c2:
            st.info("🧠 DL Sonucu")
            if dl_model:
                prep = ml_pipeline.named_steps['prep']
                X_dl = prep.transform(input_df)
                if hasattr(X_dl, "toarray"): X_dl = X_dl.toarray()
                prob = dl_model.predict(X_dl, verbose=0)[0]
                
                # Binary classification (Sigmoid)
                if len(prob) == 1:
                    val = prob[0]
                    idx = 1 if val > 0.5 else 0
                    confidence_dl = val if idx == 1 else 1 - val
                else:
                    # Multi-class (Softmax) - gerçi notebook sigmoid kullanıyordu
                    idx = np.argmax(prob)
                    confidence_dl = prob[idx]

                res_dl = le.inverse_transform([idx])[0] if le else str(idx)
                st.success(f"**{res_dl}**")
                st.metric(label="Güven Skoru", value=f"%{confidence_dl*100:.2f}")
            else:
                st.warning("DL Modeli Aktif Değil")