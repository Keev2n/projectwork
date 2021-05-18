import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random

def img_preprocess(image):
    image = image[120:480, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    return image/255

def img_zoom(image):
    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image

def img_random_brightness(image):
    brigthness = iaa.Multiply((0.2, 1.2))
    image = brigthness.augment_image(image)
    return image

def img_random_flip(image, steering_angle):
    image = iaa.Fliplr(1.0)
    steering_angle = -steering_angle
    return image, steering_angle

def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.4:
        image = img_zoom(image)
    if np.random.rand() < 0.4:
        image = img_random_brightness(image)
    #if np.random.rand() < 0.4:
        #image = img_random_flip(image, steering_angle)

    return image, steering_angle

def load_test_images(image_paths, steerings):
    fig, axs = plt.subplots(5, 1, figsize=(30,60))
    fig.tight_layout()

    for i in range(5):
        random_number = random.randint(0, len(image_paths))
        image = mpimg.imread(image_paths[random_number])

        axs[i].imshow(image)
        axs[i].set_title(steerings[random_number])

    plt.show()

def load_and_change_images_test(image_path):
    image = mpimg.imread(image_path)
    cv2.imshow("Originak", image)
    zoomed_image = img_zoom(image)
    cv2.imshow("Zoomed", zoomed_image)
    random_brightness = img_random_brightness(image)
    cv2.imshow("Random brigthness", random_brightness)
    #random_flip = img_random_flip(image, 0)
    #cv2.imshow("Flipped image", random_flip)
    cv2.waitKey(0)

def remove_data(num_bins, numbers_per_bin, data):
    remove_list = []
    for j in range(num_bins):
        list_ = []
        for i in range(len(data['steering'])):
            if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
                list_.append(i)
        list_ = shuffle(list_)
        list_ = list_[numbers_per_bin:]
        remove_list.extend(list_)

    print('removed', len(remove_list))
    data.drop(data.index[remove_list], inplace=True)
    print('remaining:', len(data))

def load_img_measurements(datadir, data):
    image_path = []
    steering = []
    kmh = []
    throttle = []
    brake = []
    for i in range(len(data)):
        row = data.iloc[i]
        path = row[0]
        image_path.append(f'{datadir}/img3/{path}.png')
        steering.append(float(row[3]))
        kmh.append(row[6])
        throttle.append(row[1])
        brake.append(row[2])

    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    kmh = np.asarray(kmh)
    throttle = np.asarray(throttle)
    brake = np.asarray(brake)
    return image_paths, steerings, kmh, throttle, brake

def batch_generator(image_paths, kmh_all, steering_ang, throttle_all, brake_all, batch_size, istraining):
    while True:
        batch_img = []
        batch_steering = []
        batch_kmh = []
        batch_throttle = []
        batch_brake = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)

            if istraining:
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
                kmh = kmh_all[random_index]
                throttle = throttle_all[random_index]
                brake = brake_all[random_index]
            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]
                kmh = kmh_all[random_index]
                throttle = throttle_all[random_index]
                brake = brake_all[random_index]

            im = img_preprocess(im)
            batch_img.append(im)
            batch_steering.append(steering)
            batch_kmh.append(kmh)
            batch_throttle.append(throttle)
            batch_brake.append(brake)

        yield ([np.asarray(batch_img), np.asarray(batch_kmh)], [np.asarray(batch_steering), np.asarray(batch_throttle), np.asarray(batch_brake)])

def nvidia_model():
    numb_input = keras.Input(shape=(1,))
    image_input = keras.Input(shape=(66, 200, 3))
    x = layers.Convolution2D(24, kernel_size=(5,5), strides=(2, 2), input_shape=(66, 200, 3), activation='relu')(image_input)
    x = layers.Convolution2D(36, kernel_size=(5,5), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(48, kernel_size=(5,5), strides=(2, 2), activation='relu')(x)
    x = layers.Convolution2D(64, kernel_size=(3,3), activation='relu')(x)
    x = layers.Convolution2D(64, kernel_size=(3,3), activation='relu')(x)
    flat_layer = layers.Flatten()(x)

    merge = layers.concatenate([flat_layer,numb_input])
    merge = layers.Dense(500, activation='relu')(merge)
    merge = layers.Dropout(0.5)(merge)
    merge = layers.Dense(100, activation = 'relu')(merge)
    merge = layers.Dense(10, activation = 'relu')(merge)
    output_steering = layers.Dense(1)(merge)
    output_throttle = layers.Dense(1)(merge)
    output_brake = layers.Dense(1)(merge)

    model = keras.Model(
        inputs = [image_input, numb_input],
        outputs = [output_steering, output_throttle, output_brake]
    )
    optimizer = Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    return model

def test_model():
    numb_input = keras.Input(shape=(1,))
    image_input = keras.Input(shape=(66, 200, 3))
    x = layers.Convolution2D(60, (5, 5), activation='relu')(image_input)
    x = layers.Convolution2D(30, (5,5), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Convolution2D(15, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    flat_layer = layers.Flatten()(x)

    merge = layers.concatenate([flat_layer,numb_input])
    merge = layers.Dense(500, activation='relu')(merge)
    merge = layers.Dropout(0.5)(merge)
    output_steering = layers.Dense(1)(merge)
    output_throttle = layers.Dense(1)(merge)
    output_brake = layers.Dense(1)(merge)

    model = keras.Model(
        inputs = [image_input, numb_input],
        outputs = [output_steering, output_throttle, output_brake]
    )
    optimizer = Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    return model

datadir = 'C:/Users/User/Desktop/Projektmunka'
columns = ['name','throttle','brake','steering','xCoordination','yCoordination','kmh']
data = pd.read_csv(os.path.join(datadir, 'data4.csv'), names=columns, sep=';')
print(data)

num_bins = 25
numbers_per_bin = 2000
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1]+bins[1:])*0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (numbers_per_bin, numbers_per_bin))
plt.show()

remove_data(num_bins, numbers_per_bin, data)
hist, _ = np.histogram(data['steering'], (num_bins))
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (numbers_per_bin, numbers_per_bin))
plt.show()

image_paths, steerings, kmh, throttle, brake = load_img_measurements(datadir, data)
print(image_paths)
print(steerings)
load_test_images(image_paths, steerings)
load_and_change_images_test(image_paths[random.randint(0, len(image_paths))])

X1_train, X1_valid, X2_train, X2_valid, y1_train, y1_valid, y2_train, y2_valid, y3_train, y3_valid = train_test_split(image_paths, kmh, steerings, throttle, brake, test_size=0.1, random_state=6)
print(f'Training: X1 training: {len(X1_train)}, X1 validation: {len(X1_valid)}, X2 training: {len(X2_train)}, X2 validation: {len(X2_valid)} y1 training: {len(y1_train)}, y1 validation: {len(y1_valid)}, y2 training: {len(y2_train)}, y2 validation: {len(y2_valid)}, y3 training: {len(y3_train)}, y3 validaiton: {len(y3_valid)}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y1_train, bins=num_bins, width=0.05, color='blue')
axes[0].set_title('Training set')
axes[1].hist(y1_valid, bins=num_bins, width=0.05, color='red')
axes[1].set_title('Validation set')
plt.show()

#model = test_model()
model = nvidia_model()
print(model.summary())

history = model.fit(batch_generator(X1_train, X2_train, y1_train, y2_train, y3_train, 200, 1),
                                  steps_per_epoch=100,
                                  epochs=10,
                                  validation_data=batch_generator(X1_valid, X2_valid, y1_valid, y2_valid, y3_valid, 200, 0),
                                  validation_steps=100,
                                  verbose=1,
                                  shuffle = 1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

model.save('model_town01_test.h5')
