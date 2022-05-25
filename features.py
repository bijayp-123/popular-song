
import librosa

def chroma_stft(chorus_file):
  data, sample_rate = librosa.load(chorus_file)
  stft_features = librosa.feature.chroma_stft(y=data, sr=sample_rate)
  return stft_features

def chroma_cqt(chorus_file):
  data, sample_rate = librosa.load(chorus_file)
  cqt_features = librosa.feature.chroma_cqt(y=data, sr=sample_rate)
  return cqt_features

def chroma_cens(chorus_file):
  data, sample_rate = librosa.load(chorus_file)
  cens_features = librosa.feature.chroma_cens(y=data, sr=sample_rate)
  return cens_features

def mfcc(chorus_file):
  data, sample_rate = librosa.load(chorus_file)
  mfcc_features = librosa.feature.mfcc(y=data, sr=sample_rate)
  return mfcc_features

def rms(chorus_file):
  data, sample_rate = librosa.load(chorus_file)
  rms_features = librosa.feature.rms(y=data)
  return rms_features

def spectral_centroid(chorus_file):
  data, sample_rate = librosa.load(chorus_file)
  centroid_features = librosa.feature.spectral_centroid(y=data, sr=sample_rate)
  return centroid_features

def spectral_bandwidth(chorus_file):
  data, sample_rate = librosa.load(chorus_file)
  bandwidth_features = librosa.feature.spectral_bandwidth(y=data, sr=sample_rate)
  return bandwidth_features

def spectral_contrast(chorus_file):
  data, sample_rate = librosa.load(chorus_file)
  contrast_features = librosa.feature.spectral_contrast(y=data, sr=sample_rate)
  return contrast_features

def spectral_rolloff(chorus_file):
  data, sample_rate = librosa.load(chorus_file)
  rolloff_features = librosa.feature.spectral_rolloff(y=data, sr=sample_rate)
  return rolloff_features

def tonnetz(chorus_file):
  data, sample_rate = librosa.load(chorus_file)
  tonnetz_features = librosa.feature.tonnetz(y=data, sr=sample_rate)
  return tonnetz_features

def zero_crossing_rate(chorus_file):
  data, sample_rate = librosa.load(chorus_file)
  zcr_features = librosa.feature.zero_crossing_rate(y=data)
  return zcr_features
