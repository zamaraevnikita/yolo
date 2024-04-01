import logging
import cv2
from ultralytics import YOLO
import concurrent.futures
import numpy as np
import sys
import argparse
import json

# Отключение логирования библиотеки ultralytics
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Функция для вычисления центра тяжести
def calculate_center(keypoints):
    # Проверка наличия ключевых точек
    if len(keypoints) == 0:
        return None

    # Вычисление среднего значения по x и y координатам
    mean_x = np.mean(keypoints[:, 0])
    mean_y = np.mean(keypoints[:, 1])

    return mean_x, mean_y

# Функция для сохранения ключевых точек в JSON файл
def save_keypoints_to_json(keypoints, file_path):
    keypoints_dict = {'keypoints': keypoints.tolist()}
    with open(file_path, 'w') as json_file:
        json.dump(keypoints_dict, json_file)

# Функция для обработки изображения
def process_image(image_path, yolo_model='yolov8n-pose.pt', confidence_threshold=0.5, min_area=100):
    # Загрузка изображения
    image = cv2.imread(image_path)
    
    # Загрузка модели YOLO с весами оценки позы
    model = YOLO(yolo_model)
    
    # Выполнение обнаружения объектов с помощью YOLO
    results = model(image, conf=confidence_threshold)
    
    # Получение аннотированного изображения с предсказаниями YOLO
    annotated_image = results[0].plot()
    
    # Получение ключевых точек
    keypoints = results[0].keypoints.xy.cpu().numpy().reshape(-1, 2)
    
    # Сохранение ключевых точек в файл JSON
    keypoints_json_file = 'keypoints.json'
    save_keypoints_to_json(keypoints, keypoints_json_file)
    
    # Вычисление центра тяжести
    center = calculate_center(keypoints)
    
    # Запись центра тяжести в файл
    if center is not None:
        center_file = 'center_of_gravity.txt'
        with open(center_file, 'w') as f:
            f.write(f"Центр тяжести: {center[0]}, {center[1]}")

    return annotated_image
# Функция для обработки видео
def process_video(video_path, yolo_model='yolov8n-pose.pt', confidence_threshold=0.5):
    # Загрузка модели YOLO с весами оценки позы
    model = YOLO(yolo_model)
    
    # Открытие объекта захвата видео для видеофайла
    cap = cv2.VideoCapture(video_path)

    # Создание файла для записи keypoints
    keypoints_file = 'keypoints.txt'
    with open(keypoints_file, 'w') as f:
        # Определение функции для обработки кадра
        def process_frame(frame):
            results = model(frame, conf=confidence_threshold)
            keypoints = results[0].keypoints.xy.cpu().numpy().reshape(-1, 2)
            for keypoint in keypoints:
                f.write(' '.join(map(str, keypoint)) + '\n')
            f.write('\n')

        # Основной цикл для захвата кадров видео и выполнения обнаружения объектов YOLO
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while(cap.isOpened()):
                # Чтение кадра из захвата видео
                success, frame = cap.read()

                if success:
                    # Запуск функции process_frame в отдельном потоке для каждого кадра
                    executor.submit(process_frame, frame)
                else:
                    break

    # Освобождение объекта захвата видео
    cap.release()

# Функция для обработки видеопотока с веб-камеры
def process_webcam(yolo_model='yolov8n-pose.pt', confidence_threshold=0.5):
    # Загрузка модели YOLO с весами оценки позы
    model = YOLO(yolo_model)
    
    # Открытие объекта захвата видео для вебкамеры
    cap = cv2.VideoCapture(0)

    # Основной цикл для захвата кадров видео и выполнения обнаружения объектов YOLO
    while(cap.isOpened()):
        # Чтение кадра из захвата видео
        success, frame = cap.read()

        if success:
            # Выполнение обнаружения объектов с помощью YOLO и сохранение результатов
            results = model(frame, save=True, conf=confidence_threshold)
        
            # Получение аннотированного кадра с ограничительными рамками и метками
            annotated_frame = results[0].plot()
        
            # Отображение аннотированного кадра с предсказаниями YOLO
            cv2.imshow('Yolo', annotated_frame)

            # Прерывание цикла при нажатии клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # Освобождение объекта захвата видео
    cap.release()

if __name__ == "__main__":
    # Выбор режима работы
    mode = 'photo'  # Используем режим photo для обработки изображения

    if mode == 'photo':
        # Путь к изображению
        image_path = 'statham.jpg'
        process_image(image_path, confidence_threshold=0.5)
    elif mode == 'video':
        # Добавьте код для обработки видео
        pass
    elif mode == 'webcam':
        # Добавьте код для обработки веб-камеры
        pass