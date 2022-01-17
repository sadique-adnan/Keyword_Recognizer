import pickle

def load_train_val_files():
    train_files = open("/home/dfki.uni-bremen.de/ssiddiqui/Desktop/Kimmi/data/train_files.pkl", "rb")
    val_files = open('/home/dfki.uni-bremen.de/ssiddiqui/Desktop/Kimmi//data/val_files.pkl', 'rb')
    X_train  = pickle.load(train_files)
    X_val = pickle.load(val_files)
    return X_train, X_val    

def get_label_list():
    labels_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    return labels_list

