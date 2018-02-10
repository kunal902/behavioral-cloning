
import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D, Cropping2D
from keras.models import Sequential
from keras import backend as K
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img
import cv2


def reduce_brightness(image):
    # convert the image to hsv color space so that the brightness can be adjusted
    output_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # add some random brightness to the image and also add some constant so that it is not completely dark
    random_bright = .25 + np.random.uniform()
    # apply the random brightness to the V channel of HSV color space
    output_image[::2] = output_image[::2]*random_bright
    # convert the image back to the RGB color space
    output_image = cv2.cvtColor(output_image, cv2.COLOR_HSV2RGB)
    return output_image


def augment_data(row):
    # get the steering angle
    steering_angle = row['steering']
    # randomly choose images from center, left or right camera images
    camera_image = np.random.choice(['center', 'left', 'right'])
    # adjust the steering angle for left and right camera images
    if camera_image == 'left':
        steering_angle += 0.25
    elif camera_image == 'right':
        steering_angle -= 0.25
    # load the image from the data folder
    output_image = load_img(row[camera_image].strip())
    output_image = img_to_array(output_image)
    # randomly flip the image so as to avoid the left turn bias
    flip_probability = np.random.random()
    if flip_probability > 0.5:
        steering_angle = -1 * steering_angle
        output_image = cv2.flip(output_image, 1)
    # reduce the image brightness
    output_image = reduce_brightness(output_image)
    return output_image, steering_angle


def load_data():
    # array to store the image data and steering angle measurements
    images, measurements = [], []
    # read csv using pandas library, read only the four columns i.e center, left, right images and steering angle
    data_frame = pd.read_csv('data/driving_log.csv', usecols=[0, 1, 2, 3])
    # iterate through the csv data
    for index, row in data_frame.iterrows():
        # augment the data and retrieve the new image and steering angle measurement
        image, steering_angle = augment_data(row)
        # store the result of augmentation
        images.append(image)
        measurements.append(steering_angle)
    # create a numpy array of data and labels
    x_train = np.array(images)
    y_train = np.array(measurements)
    return x_train, y_train


def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(32, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))
    model.add(Conv2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dropout(.3))
    model.add(ELU())
    model.add(Dense(512))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model


def main(_):
    # load the data
    x_train, y_train = load_data()
    print(x_train.shape, y_train.shape)
    # build the model
    model = build_model()
    # train the model with a train/test split of 0.8/0.2 along with random shuffling of the data,
    # train for 5 epochs as I found out to be decent number of epochs by experimentation
    model.fit(x_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
    # save the model
    model.save("model.h5")
    # clear the session to release the gpu memory
    K.clear_session()


if __name__ == '__main__':
    tf.app.run()
