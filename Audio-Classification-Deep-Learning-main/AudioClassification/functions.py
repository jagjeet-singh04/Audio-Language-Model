import librosa, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from django.conf import settings

assets_dir = settings.ASSETS_ROOT
warnings.filterwarnings("ignore")

# Load models without compiling to avoid legacy training-config issues
Model1_ANN = load_model(assets_dir + "Model1.h5", compile=False)
Model2_CNN1D = load_model(assets_dir + "Model2.h5", compile=False)
Model3_CNN2D = load_model(assets_dir + "Model3.h5", compile=False)

# Prepare label encoder
le = LabelEncoder()
try:
    final = pd.read_pickle(assets_dir + "extracted_df.pkl")
    y = np.array(final["class"].tolist())
    le.fit(y)
except Exception:
    # Fallback: if class list is unavailable, infer number of classes from model output
    try:
        num_classes = int(Model1_ANN.output_shape[-1])
    except Exception:
        # default to 10 classes if output_shape is unavailable
        num_classes = 10
    le.fit(np.arange(num_classes))


def extract_feature(audio_path):
    # Use SOXR resampler to avoid optional resampy dependency
    audio_data, sample_rate = librosa.load(audio_path, sr=22050, res_type="soxr_hq")
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=128)
    feature_scaled = np.mean(feature.T, axis=0)
    return np.array([feature_scaled])


def ANN_print_prediction(audio_path):
    prediction_feature = extract_feature(audio_path)
    predicted_vector = np.argmax(Model1_ANN.predict(prediction_feature), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    return predicted_class[0]


def CNN1D_print_prediction(audio_path):
    tmp = extract_feature(audio_path)
    prediction_feature = np.expand_dims(tmp, axis=2)
    predicted_vector = np.argmax(Model2_CNN1D.predict(prediction_feature), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    return predicted_class[0]


def CNN2D_print_prediction(audio_path):
    tmp2 = extract_feature(audio_path)
    prediction_feature = tmp2.reshape(tmp2.shape[0], 16, 8, 1)
    predicted_vector = np.argmax(Model3_CNN2D.predict(prediction_feature), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    return predicted_class[0]
