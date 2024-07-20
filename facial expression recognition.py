# -*- coding: utf-8 -*-
import tensorflow as tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, AveragePooling2D, Activation, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
print('Tensorflow Version:',tensorflow.__version__)

"""## <b> Load the  Dataset"""

# Loading the Dataset

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
names=['emotion','pixels','usage']
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Emotion Detection/fer2013.csv',names=names, na_filter=False)
im=df['pixels']
df.head()

"""## **Prepare the dataset for the model**"""

# Loading the Dataset

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
names=['emotion','pixels','usage']
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Emotion Detection/fer2013.csv',names=names, na_filter=False)
im=df['pixels']
df.head()

filname = '/content/drive/MyDrive/Colab Notebooks/Emotion Detection/fer2013.csv'

# Images are of 48x48 pixels
# No. of Images = 35887

def getData(filname):
    Y = []
    X = []
    first = True
    for line in open(filname):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y

X, Y = getData(filname)
num_class = len(set(Y))
print(num_class)

N, D = X.shape
X = X.reshape(N, 48, 48, 1) #Reshaping the images present in the dataset

#Splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)

"""###**Develop CNN model for recognizing facial expression of the images**"""

def my_model():
    model = Sequential()
    input_shape = (48,48,1)
    model.add(Conv2D(64, (5, 5), input_shape=input_shape,activation='relu', padding='same'))#64 neurons with 5*5 filter
    #This class allows to create convolutional neural network to extract feature from the images
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))#MaxPooling2D helps to reduce the size of the data

    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))#MaxPooling2D helps to reduce the size of the data

    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))#MaxPooling2D helps to reduce the size of the data

    model.add(Flatten())#Converts multi dimensional array to 1D channel
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu')) #relu activation function added to remove the negative values
    model.add(Dropout(0.2)) #Used to prevent a model from overfitting
    model.add(Dense(7)) #output layer
    model.add(Activation('softmax')) #softmax activation function

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

    return model
model=my_model()
model.summary()

"""###**Fit the basic CNN Model**"""



model=my_model() # create the model
# model.set_value(model.optimizer.lr,1e-3) # set the learning rate
# fit the model
h=model.fit(x=X_train,
            y=y_train,
            batch_size=64,
            epochs=20,
            verbose=1,
            validation_data=(X_test,y_test),
            shuffle=True)

model.save("/model_filter.h5")

#Categories of Facial Expressions

objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
y_pos = np.arange(len(objects))
print(y_pos)

#A Fucntion for visualizing the Facial Expression Recognition Results

def emotion_analysis(emotions):
    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, emotions, align='center', alpha=0.9)
    plt.title('Emotion')

plt.show()

"""###**Predict the Facial Expression of an Image**"""

from skimage import io
import warnings
warnings.filterwarnings("ignore")

img = image.load_img('/content/drive/MyDrive/Colab Notebooks/Emotion Detection/angry.jpg', grayscale=True, target_size=(48, 48))
show_img=image.load_img('/content/drive/MyDrive/Colab Notebooks/Emotion Detection/angry.jpg', grayscale=False, target_size=(200, 200))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)
 #print(custom[0])
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48]);

plt.gray()
plt.imshow(show_img)
plt.xticks([])
plt.yticks([])

plt.show()

objects = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

m=0.000000000000000000001
a=custom[0]
for i in range(0,len(a)):
    if a[i]>m:
        m=a[i]
        ind=i

print('Expression Prediction:',objects[ind])

from skimage import io
import warnings
warnings.filterwarnings("ignore")

img = image.load_img('angry.jpg', grayscale=True, target_size=(48, 48))
show_img=image.load_img('angry.jpg', grayscale=False, target_size=(200, 200))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)
 #print(custom[0])
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48]);

plt.gray()
plt.imshow(show_img)
plt.xticks([])
plt.yticks([])

plt.show()

objects = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

m=0.000000000000000000001
a=custom[0]
for i in range(0,len(a)):
    if a[i]>m:
        m=a[i]
        ind=i

print('Expression Prediction:',objects[ind])

"""##**Use OpenCV and Haar Cascade File to check the emotion in real time**

"""

fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)