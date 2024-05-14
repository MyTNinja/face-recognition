from keras.src.layers import Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import pickle
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
import time


# Deep Learning CNN model to recognize face
"""This script uses a database of images and creates CNN model on top of it to test
   if the given image is recognized correctly or not"""

'''####### IMAGE PRE-PROCESSING for TRAINING and TESTING data #######'''

# Specifying the folder where images are present
# name = input('Enter your name: ')
TrainingImagePath = 'data/Final Training Images'


# Understand more about ImageDataGenerator at below link
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# Defining pre-processing transformations on raw images of training data
# These hyperparameters helps to generate slightly twisted versions
# of the original image, which leads to a better model, since it learns
# on the good and bad mix of images
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Defining pre-processing transformations on raw images of testing data
# No transformations are done on the testing images
test_datagen = ImageDataGenerator()

# Generating the Training Data
training_set = train_datagen.flow_from_directory(
    TrainingImagePath,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# Generating the Testing Data
test_set = test_datagen.flow_from_directory(
    TrainingImagePath,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# Printing class labels for each face
print(test_set.class_indices)

'''############ Creating lookup table for all faces ############'''
# class_indices have the numeric tag for each face
TrainClasses = training_set.class_indices

# Storing the face and the numeric tag for future reference
ResultMap = {}
for faceValue, faceName in zip(TrainClasses.values(), TrainClasses.keys()):
    ResultMap[faceValue] = faceName

# Saving the face map for future reference

with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)

# The model will give answer as a numeric tag
# This mapping will help to get the corresponding face name for it
print("Mapping of Face and its ID", ResultMap)

# The number of neurons for the output layer is equal to the number of faces
OutputNeurons = len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)

'''######################## Create CNN deep learning model ########################'''


'''Initializing the Convolutional Neural Network'''
classifier = Sequential()

''' STEP--1 Convolution
# Adding the first layer of CNN
# we are using the format (64,64,3) because we are using TensorFlow backend
# It means 3 matrix of size (64X64) pixels representing Red, Green and Blue components of pixels
'''
classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))

classifier.add(Flatten())

# classifier.add(Dropout(0.2))
classifier.add(Dense(64, activation='relu'))

classifier.add(Dense(OutputNeurons, activation='sigmoid'))

classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

###########################################################

# Measuring the time taken by the model to train
StartTime = time.time()

# Starting the model training
classifier.fit(
    training_set,
    epochs=25,
    validation_data=test_set,
    validation_steps=10)

EndTime = time.time()
print("###### Total Time Taken: ", round((EndTime - StartTime) / 60), 'Minutes ######')