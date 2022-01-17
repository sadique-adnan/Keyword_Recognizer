from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.layers.experimental.preprocessing import Normalization,Resizing




class SpeechModel(Model):
    def __init__(self, num_classes = 8):
        super(SpeechModel, self).__init__()
        self.resize = Resizing(32,32)
        self.norm = Normalization()
        self.conv1 = Conv2D(32,3, activation='relu')
        self.conv2 = Conv2D(64,3, activation='relu')
        self.pool1 = MaxPool2D(pool_size=(2, 2))
        self.dropout1 = Dropout(0.25)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dropout2 = Dropout(0.5)
        self.output(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.resize(inputs)
        x = self.norm.adapt(x)
        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        
        return self.output(x)



