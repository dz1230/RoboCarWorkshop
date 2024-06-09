import tensorflow as tf
import cv2
import time
from time import sleep
import smbus as smbus
import os
import numpy as np
from threading import Thread, Semaphore

# Load the model
modelPath = './stop_sign_model.h5'
inputSaveDir = './predictions/m9_5/'
current_img_dir = './.robo/1/'
savePrediction = False
pictureNumber = 0

bus = smbus.SMBus(1)


def stopCar(bus):
    bus.write_word_data(adress, 0xff, carStopValue)
    time.sleep(0.01)

def driveForwardConstantly(bus):
    bus.write_word_data(adress, 0xff, carForwardValue)
    time.sleep(0.01)

adress = 0x18
width = 224
height = 224
camIndex = 0

carForwardValue = 0x220A
carBackwardValue = 0x230A
carRightValue = 0x250A
carLeftValue = 0x240A
carStopValue = 0x210A

carLeftLightOnValue = 0x3601
carLeftLightOffValue = 0x3600
carRightLightOnValue = 0x3701
carRightLightOffValue = 0x3700

camera_ready = Semaphore(0)
camera_buffer_1 = None
camera_buffer_1_lock = Semaphore(1)
camera_buffer_2 = None
camera_buffer_2_lock = Semaphore(1)
camera_active_buffer = 0
camera_active_buffer_lock = Semaphore(1)

def camera_thread():
    global bus, camera_buffer_1, camera_buffer_2, camera_buffer_1_lock, camera_buffer_2_lock, camera_ready, camera_active_buffer_lock, camera_active_buffer
    camera = cv2.VideoCapture(camIndex)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # camera.set(cv2.CAP_PROP_FPS, 10) doesnt work, only 15 fps is supported
    camera_buffer_1_lock.acquire()
    _, camera_buffer_1 = camera.read()
    camera_buffer_1_lock.release()
    camera_active_buffer_lock.acquire()
    camera_active_buffer = 1
    camera_active_buffer_lock.release()
    time.sleep(10) # warm-up
    camera_ready.release()
    try:
        while True:
            camera_buffer_2_lock.acquire()
            _, camera_buffer_2 = camera.read()
            camera_buffer_2_lock.release()
            camera_active_buffer_lock.acquire()
            camera_active_buffer = 2
            camera_active_buffer_lock.release()
            time.sleep(0.01)
            camera_buffer_1_lock.acquire()
            _, camera_buffer_1 = camera.read()
            camera_buffer_1_lock.release()
            camera_active_buffer_lock.acquire()
            camera_active_buffer = 1
            camera_active_buffer_lock.release()
            time.sleep(0.01)
    except Exception:
        stopCar(bus)

def prepare_motors(bus):
    speed = 0x2
    right_motor = 0x27
    a = right_motor << 8
    i2c_value_right = a + speed
    bus.write_word_data(adress, 0xff, i2c_value_right)
    time.sleep(0.01)
    left_motor = 0x26
    b = left_motor << 8
    i2c_value_left = b + speed
    bus.write_word_data(adress, 0xff, i2c_value_left)
    time.sleep(0.01)

def crop_img(frame):
    # Mittelpunkt des Bildes bestimmen
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    
    # Start- und Endpunkte des Ausschnitts berechnen
    start_x = max(center_x - 112, 0) # 224 / 2 = 112
    end_x = start_x + 224
    start_y = max(center_y - 112, 0)
    end_y = start_y + 224
    
    # Stellen Sie sicher, dass der Ausschnitt innerhalb der Bildgrenzen bleibt
    if end_x > frame.shape[1]: 
        end_x = frame.shape[1]
        start_x = end_x - 224
    if end_y > frame.shape[0]:
        end_y = frame.shape[0]
        start_y = end_y - 224
    
    # Bild beschneiden
    cropped_image = frame[start_y:end_y, start_x:end_x]
    return cropped_image

def prediction_thread():
    global pictureNumber, savePrediction, inputSaveDir, bus, modelPath, camera_buffer_1, camera_buffer_2, camera_buffer_1_lock, camera_buffer_2_lock, camera_ready, camera_active_buffer_lock, camera_active_buffer
    prepare_motors(bus)
    camera_ready.acquire()
    model = tf.keras.models.load_model(modelPath)
    model.predict(np.zeros((1, 224, 224, 3)))
    firstPrediction = True
    try:
        while True:
            loop_start_time = time.time()
            camera_active_buffer_lock.acquire()
            active_buffer = camera_active_buffer
            camera_active_buffer_lock.release()
            if active_buffer == 1:
                camera_buffer_1_lock.acquire()
                img = camera_buffer_1
                camera_buffer_1_lock.release()
            elif active_buffer == 2:
                camera_buffer_2_lock.acquire()
                img = camera_buffer_2
                camera_buffer_2_lock.release()
            else:
                img = None
                raise Exception("Invalid active buffer")
            cropped_img = crop_img(img)
            _, buffer = cv2.imencode('.jpg', cropped_img)
            with open(current_img_dir + '0.jpg', 'wb') as f:
                f.write(buffer.tobytes())
            single_frame_dataset = tf.keras.utils.image_dataset_from_directory(
                './.robo/',
                image_size=(224, 224),
                batch_size=1,
                shuffle=False,
            )
            for image, label in single_frame_dataset:
                prediction = model.predict(image)
                print(prediction)
                if prediction[0][0] > 0.8:
                    stopCar(bus)
                elif firstPrediction:
                    firstPrediction = False
                else:
                    driveForwardConstantly(bus)
            if savePrediction:
                imgPath = inputSaveDir + 'img_' + str(pictureNumber) + '_pred_' + str(prediction[0][0]) + '_t_' + str(time.time()) + '.jpg'
                with open(imgPath, 'wb') as f:
                    f.write(buffer.tobytes())
                pictureNumber += 1
            loop_end_time = time.time()
            print("Prediction time: ", loop_end_time - loop_start_time)
    except Exception:
        stopCar(bus)

def main():
    global bus, inputSaveDir, current_img_dir
    os.makedirs(inputSaveDir, exist_ok=True)
    os.makedirs(current_img_dir, exist_ok=True)
    cameraThread = Thread(target=camera_thread)
    cameraThread.start()
    predictionThread = Thread(target=prediction_thread)
    predictionThread.start()
    while True:
        time.sleep(1)

if __name__ == '__main__':
    main()