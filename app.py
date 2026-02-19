import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ==========================================
# LOAD MODELS
# ==========================================

rf_model = joblib.load("models/spotify_flop_classifier.pkl")

cluster_scaler = joblib.load("models/scaler.pkl")
cluster_pca = joblib.load("models/pca.pkl")
cluster_model = joblib.load("models/kmeans.pkl")

# ==========================================
# UI CONFIG
# ==========================================

st.set_page_config(page_title="Spotify ML Intelligence", layout="wide")
st.title("Spotify Song Performance Intelligence")

st.sidebar.header("Enter Song Features")


year = st.sidebar.slider("Release Year", 1990, datetime.now().year, 2020)
acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.5)
danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.5)
instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.0)
key = st.sidebar.slider("Key (0-11)", 0, 11, 5)
mode = st.sidebar.selectbox("Mode", [0, 1])
time_signature = st.sidebar.selectbox("Time Signature", [3, 4, 5])
liveness = st.sidebar.slider("Liveness", 0.0, 1.0, 0.2)
loudness = st.sidebar.slider("Loudness", -60.0, 0.0, -10.0)
speechiness = st.sidebar.slider("Speechiness", 0.0, 1.0, 0.1)
tempo = st.sidebar.slider("Tempo", 50.0, 200.0, 120.0)
valence = st.sidebar.slider("Valence", 0.0, 1.0, 0.5)
duration_min = st.sidebar.slider("Duration (minutes)", 1.0, 10.0, 3.0)
language = st.selectbox("Language", ["English", "Hindi", "Korean","Tamil","Malayalam","Telugu","Unknown"])

# ==========================================
# CLASSIFICATION FEATURE ENGINEERING
# ==========================================

duration_ms = duration_min * 60 * 1000
#song_age = datetime.now().year - year
DATASET_MAX_YEAR = 2020
song_age = DATASET_MAX_YEAR - year
energy_loudness = energy * loudness
dance_valence = danceability * valence
speech_energy = speechiness * energy
tempo_energy = tempo * energy
acoustic_energy_ratio = acousticness / (energy + 1e-5)

# Create DataFrame with EXACT required columns
classification_df = pd.DataFrame([{
    "acousticness": acousticness,
    "danceability": danceability,
    "duration_ms": duration_ms,
    "energy": energy,
    "liveness": liveness,
    "loudness": loudness,
    "speechiness": speechiness,
    "tempo": tempo,
    "valence": valence,
    "year": year,
    "song_age": song_age,
    "energy_loudness": energy_loudness,
    "dance_valence": dance_valence,
    "speech_energy": speech_energy,
    "acoustic_energy_ratio": acoustic_energy_ratio,
    "tempo_energy": tempo_energy,
    "language": language
}])



# ==========================================
# CLUSTERING FEATURE ENGINEERING
# ==========================================

minor_mode = 1 - mode
high_acoustic = int(acousticness > 0.7)
high_speech = int(speechiness > 0.6)

cluster_df = pd.DataFrame([{
    "acousticness": acousticness,
    "danceability": danceability,
    "energy": energy,
    "instrumentalness": instrumentalness,
    "liveness": liveness,
    "loudness": loudness,
    "speechiness": speechiness,
    "tempo": tempo,
    "valence": valence,
    "duration_min": duration_min,
    "energy_loudness": energy_loudness,
    "dance_valence": dance_valence,
    "speech_energy": speech_energy,
    "minor_mode": minor_mode,
    "high_acoustic": high_acoustic,
    "high_speech": high_speech
}])

# ==========================================
# TABS
# ==========================================

tab1, tab2 = st.tabs(["Flop Prediction", "Song Segmentation"])

# ==========================================
# TAB 1 - CLASSIFICATION
# ==========================================

with tab1:

    st.header("🎯 Flop Prediction")

    prob = rf_model.predict_proba(classification_df)[0][1]
    prediction = rf_model.predict(classification_df)[0]

    st.metric("Flop Probability", f"{prob:.2%}")

    if prediction == 1:
        st.error(" High Risk of Flop")
    else:
        st.success(" Likely to Perform Well")

    st.write(classification_df)    

# ==========================================
# TAB 2 - CLUSTERING
# ==========================================

with tab2:

    st.header("🎧 Song Segment Analysis")

    scaled = cluster_scaler.transform(cluster_df)
    pca_features = cluster_pca.transform(scaled)
    cluster_label = cluster_model.predict(pca_features)[0]

    # Real interpretation based on your results
    cluster_info = {
        0: {
            "name": "Commercial / High Energy Mainstream",
            "popularity": "Highest Avg Popularity",
            "desc": "High energy, danceable, loud tracks. Strong commercial appeal."
        },
        1: {
            "name": "Acoustic / Emotional / Indie",
            "popularity": "Medium Popularity",
            "desc": "More acoustic, softer, emotional or instrumental tracks."
        },
        2: {
            "name": "Speech / Spoken / Niche",
            "popularity": "Very Low Popularity ",
            "desc": "High speech content, niche/spoken-word style. High flop risk segment."
        }
    }

    info = cluster_info.get(cluster_label)

    st.subheader(f"Cluster {cluster_label}: {info['name']}")
    st.write(info["desc"])
    st.write("📊 Popularity Insight:", info["popularity"])
    

