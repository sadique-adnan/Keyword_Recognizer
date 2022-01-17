from dataloader import DataGenerator
from tensorflow.keras.models import load_model
import numpy as np
from utils import get_label_list


def speech_recog(file):
    test_dataloader = DataGenerator([file], batch_size=1, shuffle=False, to_fit=False)
    model = load_model('model_spectogram.h5')
    predictions = np.argmax(model.predict(test_dataloader), axis =1)
    pred_label = get_label_list()[predictions[0]]
    print('predictions',pred_label)
    return pred_label