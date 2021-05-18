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
import ctypes
from datetime import datetime

image_width = 640
image_height = 480

def write_measurements(vehicle, image):
    c = vehicle.get_control()
    v = vehicle.get_velocity()
    l = vehicle.get_location()
    throttle = c.throttle
    steer = c.steer
    brake = c.brake
    xCoordinates = l.x
    yCoordinates = l.y
    kmh = int(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2))
    print("Throttle: ", throttle)
    print("Steer: ", steer)
    print("Brake:", brake)
    print("xCoordinates", xCoordinates)
    print("yCoordinates", yCoordinates)
    print("Kmh", kmh)
    write_to_text(throttle, steer, brake, xCoordinates, yCoordinates, kmh, image)
    process_img(image)

def write_to_text(throttle, steer, brake, xCoordinates, yCoordinates, kmh, image):
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
    image.save_to_disk(f'E:/projektmunka/img3/{current_time}.png')
    with open('E:\projektmunka\data4.csv', 'a') as file:
        file.write(f'{str(current_time)};{str(throttle)};{str(brake)};{str(steer)};{str(xCoordinates)};{str(yCoordinates)};{str(kmh)}\n')

def process_img(image):
    image = np.array(image.raw_data)
    image2 = image.reshape((image_height, image_width, 4))
    image3 = image2[:, :, :3]
    cv2.imshow("", image3)
    cv2.waitKey(1)
    return image3/255.0

actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.load_world('Town01')
    tm = client.get_trafficmanager(5000)
    tm_port = tm.get_port()

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    bp.set_attribute('color', '0,0,255')

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)

    actor_list.append(vehicle)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{image_width}')
    camera_bp.set_attribute('image_size_y', f'{image_height}')
    camera_bp.set_attribute('fov', '90')
    camera_bp.set_attribute('sensor_tick', '1')

    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    sensor = world.spawn_actor(camera_bp, spawn_point, attach_to=vehicle)

    actor_list.append(sensor)
    vehicle.set_autopilot(True, tm_port)
    tm.vehicle_percentage_speed_difference(vehicle, -20)
    tm.auto_lane_change(vehicle, False)

    sensor.listen(lambda image: write_measurements(vehicle, image))

    time.sleep(18000)

finally:
    for actor in actor_list:
        actor.destroy()
