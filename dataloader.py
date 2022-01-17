import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import os
from utils import get_label_list

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, filenames, batch_size=64, shuffle=True, to_fit = True):
        'Initialization'
        if not isinstance(filenames, list):
            raise ValueError('"filenames" must be a list')
            
        self.filenames = filenames
        self.batch_size = batch_size
        self.labels = get_label_list()
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_x = self.filenames[index * self.batch_size:(index + 1) * self.batch_size]
        # read your data here using the batch lists, batch_x and batch_y
        data = [self.get_spectrogram(file_id) for file_id in batch_x]
        if self.to_fit:
            gt = [self.get_label(file_id) for file_id in batch_x] 
            return np.array(data), np.array(gt)
        else:
            return np.array(data)    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def decode_audio(self, audio_binary):
        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)
       
    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        return label_id
        
    def get_spectrogram(self, file_path):
        audio_binary = tf.io.read_file(file_path)
        waveform = self.decode_audio(audio_binary)
      # Padding for files with less than 16000 samples
        zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

        # Concatenate audio with padding so that all audio clips will be of the 
        # same length
        waveform = tf.cast(waveform, tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)
        spectrogram = tf.signal.stft(
          equal_length, frame_length=255, frame_step=128)

        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, -1)

        return spectrogram
