from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TimeDistributed, Bidirectional, LSTM, Dense, Dropout

MODEL_PATH = "weights/violence_detector.keras"

def load_violence_model():
    model = load_model(
        MODEL_PATH,
        custom_objects={
            "TimeDistributed": TimeDistributed,
            "Bidirectional": Bidirectional,
            "LSTM": LSTM,
            "Dense": Dense,
            "Dropout": Dropout
        }
    )
    return model
