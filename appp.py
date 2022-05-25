import streamlit as st
from get_features import get_chorus_file, get_features
from features import chroma_stft,chroma_cqt,chroma_cens,mfcc,rms,spectral_centroid,spectral_bandwidth,spectral_contrast,spectral_rolloff,tonnetz,zero_crossing_rate
import joblib
import os
#from pydub import AudioSegment
import subprocess
st.title('Songs Popularity Prediction System')
st.markdown (
    """
    <style>
    [data-testid='stSidebar'][aria-expanded='true'] > div:first-child{
       width:350px
    }
    [data-testid='stSidebar'][aria-expanded='true'] > div:first-child{
       width:350px
       margin-left: -350px
    }
    </style>


    """,
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("Upload a Mp3 File")
if uploaded_file is not None:
    st.write(uploaded_file.name)
    with open(uploaded_file.name,"wb") as f:
      f.write(uploaded_file.getbuffer())
    if os.path.exists(uploaded_file.name):
        wav_file = uploaded_file.name[:-4]+".wav"
        #subprocess.call(['ffmpeg','-i',uploaded_file.name,wav_file])
        #sound = AudioSegment.from_mp3(uploaded_file.name)
        #sound.export(uploaded_file.name[:-4]+".wav", format="wav")
        
        chorus_file = get_chorus_file(uploaded_file.name) # get path to chorus file
        feature_names = [chroma_stft,chroma_cqt,chroma_cens,mfcc,rms,spectral_centroid,spectral_bandwidth,spectral_contrast,spectral_rolloff,tonnetz,zero_crossing_rate]
        features = get_features(chorus_file, feature_names)
        scaler = joblib.load("scaler.sav")
        final_input = scaler.transform(features)
        model = joblib.load("model.sav")
        prediction = model.predict(final_input)
        result = "The Song is Hit" if prediction > 0.8 else " The Song is not Hit"
        #st.write(prediction)
        st.write(result.upper())
    else:
        st.write("The file didnot get uploaded")

else:
    pass
st.write("Waiting for the file")
