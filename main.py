from dataloader import DataGenerator
from model_func import create_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import os
import wandb
from wandb.keras import WandbCallback
import argparse
from utils import get_label_list, load_train_val_files


parser = argparse.ArgumentParser(description='Keyword Recognizer')
parser.add_argument('--bs', type=int, default=64,
                    help='enter the batch size')
parser.add_argument('--epochs', type=int, default=10,
                    help='Enter the epochs to be trained')

parser.add_argument('--save_path', type=str, default='experiments', help='Enter the path to save the model')
parser.add_argument('--save_model', type=str, default='spectogram_model.h5', help='Enter the name of the  model')

#parser.add_argument('--labeltype', type=str, default='Trajectories', help='Enter the type either Trajectories or Videos')

args = parser.parse_args()
print(args)
wandb.init(project="", entity="")

experiment_path = os.path.join(args.save_path)
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)


def get_data_loader(x_train, x_val, batch_size):

    train_dataloader = DataGenerator(x_train, batch_size)
    val_dataloader = DataGenerator(x_val, batch_size)

    return train_dataloader, val_dataloader

def get_model(): 
    model = create_model(input_shape= (124,129,1))
    model.compile(optimizer=Adam(), loss= SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'],)
    print(model.summary())
    return model

def _fit(train_ds, val_ds):
    model = get_model()
    history = model.fit(
    train_ds, 
    validation_data=val_ds,  
    epochs=args.epochs,
    callbacks=[EarlyStopping(verbose=1, patience=5),WandbCallback()])
    model.save(experiment_path+args.save_model)
    return history



if __name__=='__main__':
    X_train, X_val = load_train_val_files()
    train_datagen, val_datagen = get_data_loader(X_train, X_val, args.bs)  

    history = _fit(train_datagen, val_datagen)

