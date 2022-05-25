
from pychorus import find_and_output_chorus
from scipy.stats import kurtosis, skew, tstd
import numpy as np


def get_chorus_file(file_name):
  chorus = find_and_output_chorus(file_name, "chorus_file.wav", 15)
  return "chorus_file.wav"

def get_features(file_name,feature_names):
  features = []
  for name in feature_names:
    feature = name(file_name)
    features.append(np.max(feature.T, axis=0))
    features.append(np.min(feature.T, axis=0))
    features.append(np.mean(feature.T, axis=0))
    features.append(np.median(feature.T, axis=0))
    features.append(tstd(feature.T, axis=0))
    features.append(skew(feature.T, axis=0))
    features.append(kurtosis(feature.T, axis=0))
  long_list = []
  for i in range(np.array(features).shape[0]):
    feature = features[i]
    for j in range(len(feature)):
      element = feature[j]
      long_list.append(element)

  return np.array(long_list).reshape((1,518))
  

