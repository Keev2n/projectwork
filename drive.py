import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import numpy as np
import cv2
import math
from keras.models import load_model

image_width = 640
image_height = 480

def control_vehicle(model, vehicle, image):
    image = img_preprocess(image)
    v = vehicle.get_velocity()
    kmh = int(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2))
    kmh = np.array([kmh])
    #steering_angle, throttle, brake = model.predict([image, kmh])
    print("Steering: ", steering_angle)
    #print("Throttle", throttle)
    #print("Brake", brake)
    steering_angle = model.predict(image)
    #vehicle.apply_control(carla.VehicleControl(throttle = float(throttle), steer=float(steering_angle), brake = float(brake)))
    #vehicle.apply_control(carla.VehicleControl(throttle = float(throttle), steer=float(steering_angle)))
    vehicle.apply_control(carla.VehicleControl(throttle = 0.2, steer=float(steering_angle)))


def img_preprocess(image):
    image = np.array(image.raw_data)
    image2 = image.reshape((image_height, image_width, 4))
    image3 = image2[:, :, :3]
    image = image3[120:480, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = np.array([image])
    return image

actor_list = []
try:
    model = load_model('model_town01_basic.h5')
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    world = client.load_world('Town01')

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    bp.set_attribute('color', '0,0,255')

    spawn_points = world.get_map().get_spawn_points()

    vehicle = world.spawn_actor(bp, spawn_points[4])

    actor_list.append(vehicle)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{image_width}')
    camera_bp.set_attribute('image_size_y', f'{image_height}')
    camera_bp.set_attribute('fov', '90')

    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    sensor = world.spawn_actor(camera_bp, spawn_point, attach_to=vehicle)

    actor_list.append(sensor)

    sensor.listen(lambda image: control_vehicle(model, vehicle, image))

    time.sleep(60)

finally:
    for actor in actor_list:
        actor.destroy()
