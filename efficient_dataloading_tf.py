# import tensorflow as tf
# from tensorflow.keras.datasets import cifar10
# import tensorflow.keras.losses as losses
# import tensorflow.keras.optimizers as optimizers  
import numpy as np
import pandas as pd
import random
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,precision_recall_fscore_support,auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4"

# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)

config = dict(
    lr = 0.001,
    EPOCHS = 5,
    BATCH_SIZE = 32,
    IMAGE_SIZE = 32,
    SEED = 42,
    GPU_ID=0,
    val_split = 0.2,
    test_csv = "sign_mnist_test.csv",
    train_csv = "sign_mnist_train.csv"
)

class CustomDataset(object):
    def __init__(self,csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        label, data = self.data.iloc[idx,0], np.array(self.data.iloc[idx,1:])
        data = data.reshape(28,28)
        img = Image.fromarray(np.uint8(data))
        img = img.resize((config['IMAGE_SIZE'],config['IMAGE_SIZE']))
        img = np.array(img.getdata()).reshape(config['IMAGE_SIZE'],config['IMAGE_SIZE'],1)
        return (img,label)

# ------------------------ method 1 dataloading using map and zip ------------------------------------

a1 = time.perf_counter()

train_dataset = CustomDataset(config['train_csv'])
test_dataset = CustomDataset(config['test_csv'])

training_data = list(map(lambda x: [x[0],x[1]],train_dataset))
testing_data = list(map(lambda x: [x[0],x[1]],test_dataset))
# print(len(train_data))
# print(len(train_labels))
x_train,y_train = zip(*training_data)
x_test,y_test = zip(*testing_data)

y_train = np.array(y_train).reshape(len(y_train),1)
y_test = np.array(y_test).reshape(len(y_test),1)

x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype("float32") / 255

print(x_train.shape,y_train.shape)


a2 = time.perf_counter()

print(f'{(a2-a1)/60:.2f} Mins')

#------------------------------- second method using generator -------------------------------------
a3 = time.perf_counter()

train_dataset = CustomDataset(config['train_csv'])
test_dataset = CustomDataset(config['test_csv'])


def train_generator(train_dataset):
    len_Train = len(train_dataset)
    for i in range(len_Train):
        yield train_dataset[i]

traing = train_generator(train_dataset)
testg = train_generator(test_dataset)

x_train,y_train,x_test,y_test = [],[],[],[]

for idx,(img,label) in enumerate(traing):
    x_train.append(img)
    y_train.append(label)

for idx,(img,label) in enumerate(testg):
    x_test.append(img)
    y_test.append(label)

x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype("float32") / 255

print(x_train.shape,y_train.shape)

a4 = time.perf_counter()

print(f'{(a4-a3)/60:.2f} Mins')

#------------------------------- third method using generator -------------------------------------
a5 = time.perf_counter()

train_dataset = CustomDataset(config['train_csv'])
test_dataset = CustomDataset(config['test_csv'])


def train_generator(train_dataset):
    len_Train = len(train_dataset)
    for i in range(len_Train):
        yield train_dataset[i]

traing = iter(train_generator(train_dataset))
testg = iter(train_generator(test_dataset))

x_train,y_train,x_test,y_test = [],[],[],[]

training_data = list(map(lambda x: [x[0],x[1]],traing))
testing_data = list(map(lambda x:[x[0],x[1]],testg))

x_train,y_train = zip(*training_data)
x_test,y_test = zip(*testing_data)

x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype("float32") / 255

print(x_train.shape,y_train.shape)

a6 = time.perf_counter()

print(f'{(a6-a5)/60:.2f} Mins')

