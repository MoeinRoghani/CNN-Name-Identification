"""
Title: Name and age Prediction based on a person's Image using a Convolutional Neural Network

Author: Moein Roghani
Email: roghanim@mcmaster.ca

Description: In this project, our goal is to create a model that can predict a person's name and their age, 
based on their image using a Convolutional Neural Network (CNN) which we have implemented for each. 

Datasets used for training:
- lfw: The data set contains more than 13,000 images of faces (5749 people) collected from the web
  http://vis-www.cs.umass.edu/lfw/}{http://vis-www.cs.umass.edu/lfw/
- Names100: Contains 80,000 unconstrained human face images, including 100 popular names and 800 images per name
  https://exhibits.stanford.edu/data/catalog/tp945cq9122
- IMDB-WIKI: 500k+ face images with age and gender labels
  https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
"""

# -----------------------------------------------------------------------------------------------------------------------------
# Data Processing

# The dataset is organized into multiple folders, with each folder representing a label (name or age),//
# //and the images inside each folder corresponding to the data points for that name. So basically our training set is images that are each inside// 
# //their labelled  classification folder.

import os
import numpy as np
from PIL import Image
# -----------------------------------------------------------------------------------------------------------------------------


data_directory = 'path_to_dataset_directory'


#loading data from the directory and making our X_train set which is the data points, and Y_train which is their label.
def load_data(data_directory):
    X_train = [] 
    Y_train = []
    data = []
    class_labels_dic = {}
    
    for value, label in enumerate(os.listdir(data_directory)):
        #assign an integer value for each class
        class_labels_dic[label] = value
        
        #add the data with its label
        class_folder = os.path.join(data_directory, label)
        for image in os.listdir(class_folder):
            X_train.append(os.path.join(class_folder, image))
            Y_train.append(value)
            data.append((os.path.join(class_folder, image), value))
            
    return data, class_labels_dic, np.array(X_train), np.array(Y_train)

data, class_labels_dic, X_train, Y_train = load_data(data_directory)


#preprocessing the image by converting it to an array representing the image
def preprocess_image(image, size = (224, 224)):
    
    #resize the image to a specific dimension and convert it to an RGB channel
    image = Image.open(image).convert('RGB')
    image = image.resize(size)
    
    #convert image to an array of pixel values (RGB value of every pixel)
    #it is a matrix of size 224*224, where each item is an array of size 3 (RGB)
    image_array = np.array(image) / 255.0
    
    return image_array



# -----------------------------------------------------------------------------------------------------------------------------
# Convolutional Neural Network (CNN) Architecture

# Some notes on CNN:
# We use a Dropout layer to prevent overfitting
# ReLU activation is basically f(x) = max(0, x)
# Softmax activation is a smooth approximation to the max function, making it differentiable for second derivates as well in the case of using Newton's//
# //method by second Taylor series approximation 
# We need a fully-connected layer between the Convolutional outputs and the final layer to be able to make predictions based on all the labels
# We use Categorical Cross Entropy as a loss function for multi-class classification 
# We use Mean Squared Error (MSE) as a loss function for regression

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
# -----------------------------------------------------------------------------------------------------------------------------

#mini-batch gradient descent method for faster updates when we have large datasets
def data_batch(data, batch_size=32, num_classes=None):
    while True:
        np.random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            X_batch = np.array([preprocess_image(img_path) for img_path, _ in batch_data], dtype=np.float32)
            y_batch = np.array([label for _, label in batch_data])
            y_batch = tf.keras.utils.to_categorical(y_batch, num_classes=num_classes)
            yield X_batch, y_batch


#CNN Architecture for Name Prediction using Multi-Class Classification
def name_CNN_Model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape), #convolutional layer
        BatchNormalization(),
        MaxPooling2D((2, 2)), #decrease the number of parameters
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),  #decrease the number of parameters
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)), #decrease the number of parameters
        
        Flatten(),
        Dense(128, activation='relu'), #first fully connected layers
        Dropout(0.5), #for regularization
        
        #fully connected layers (Number of neurons should equal to the number of classes)
        Dense(num_classes, activation='softmax') 
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


#CNN Architecture for Age Prediction using Regression (Age is a continuous valued variable)
def age_CNN_Model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape), #Convolutional Layer
        MaxPooling2D((2, 2)), #decrease the number of parameters
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)), #decrease the number of parameters
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'), #first fully connected layers
        Dropout(0.5), #for regularization
        
        #fully connected layers (Number of neurons should equal to the number of classes)
        Dense(num_classes, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model



# -----------------------------------------------------------------------------------------------------------------------------
# Model Training 
# -----------------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)


#it is a matrix of size 224*224, where each item is an array of size 3 (RGB)
input_shape = (224, 224, 3)
num_classes = len(class_labels_dic.values())

#need more testing for finding a better epoch
#the values are taken from a similar model
batch_size = 32
epochs = 20

#create our model and our batch for training and testing sets
train_data_gen = data_batch(train_data, batch_size, num_classes)
val_data_generator = data_batch(val_data, batch_size, num_classes)


#fit model on training data for age prediction
model = age_CNN_Model(input_shape, num_classes)
history = model.fit(
    train_data_gen,
    steps_per_epoch=len(train_data) // batch_size,
    validation_data=val_data_generator,
    validation_steps=len(val_data) // batch_size,
    epochs=epochs
)


#fit model on training data for name prediction
model = name_CNN_Model(input_shape, num_classes)
history = model.fit(
    train_data_gen,
    steps_per_epoch=len(train_data) // batch_size,
    epochs=epochs
)