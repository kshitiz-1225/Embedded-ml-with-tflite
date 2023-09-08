import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers  
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

gpus = tf.config.list_physical_devices('GPU')
print(gpus)

config = dict(
    lr = 0.001,
    EPOCHS = 5,
    BATCH_SIZE = 32,
    IMAGE_SIZE = 28,
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
# print(y_test)

# print(x_train.shape)
# exit(0)
print(x_train.shape,y_train.shape)

# exit(0)

# print(len(train_data),len(test_data))
train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(config['BATCH_SIZE'])
test_data = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(config['BATCH_SIZE'])

# for i in range(5):
#     print((test_data[i][0].shape, test_data[i][1]))

class CNNLayer(tf.keras.layers.Layer):
    def __init__(self,out_channels,kernel_size=3,input_shape=None):
        super(CNNLayer,self).__init__()
        # if (input_shape != None):
        #     self.conv1 = tf.keras.layers.Conv2D(out_channels,kernel_size=kernel_size,padding='valid',input_shape=input_shape)
        # else:
        self.conv1 = tf.keras.layers.Conv2D(out_channels,kernel_size=kernel_size,padding='valid')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation(tf.nn.relu)

    def call(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CNNarchitecture(tf.keras.Model):
    def __init__(self,n_class=25):
        super(CNNarchitecture,self).__init__()
        self.cn1_a = CNNLayer(32,3)
        self.cn1_b = CNNLayer(32,3)
        self.cn2 = CNNLayer(64,3)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024)
        self.relu2 = tf.keras.layers.Activation(tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(n_class)

    def call(self,x):
        # x = self.inp(x);
        x1 = self.cn1_a(x)
        x2 = self.cn1_b(x)
        x = tf.concat([x1,x2],axis=3)
        x = self.cn2(x)
        # x = tf.reshape(x,(x.shape[0],-1))
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu2(x)
        x = self.dense2(x)
        return x 
        

model = CNNarchitecture()
model.compile(
    optimizer = tf.keras.optimizers.Adam(lr=config['lr']),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
history = model.fit(x_train,y_train,epochs=config['EPOCHS'],batch_size=config['BATCH_SIZE'],verbose=1,validation_split=config['val_split'])
print(model.summary())
#,batch_size=config['BATCH_SIZE']
try:
    model.save('gesture_recogition_model')
except Exception as e:
    print(e)

print(model.evaluate(test_data,verbose=1))
