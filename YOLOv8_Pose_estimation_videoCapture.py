import logging
import cv2
from ultralytics import YOLO
import time
import concurrent.futures

# Отключение логирования библиотеки ultralytics
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Функция для обработки изображения
def process_image(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    
    # Загрузка модели YOLO с весами оценки позы
    model = YOLO('yolov8n-pose.pt')
    
    # Выполнение обнаружения объектов с помощью YOLO
    results = model(image)
    
    # Получение аннотированного изображения с предсказаниями YOLO
    annotated_image = results[0].plot()
    
    return annotated_image

# Функция для обработки видео
def process_video(video_path):
    # Загрузка модели YOLO с весами оценки позы
    model = YOLO('yolov8n-pose.pt')
    
    # Открытие объекта захвата видео для видеофайла
    cap = cv2.VideoCapture(video_path)

    # Создание файла для записи keypoints
    with open('keypoints.txt', 'w') as f:
        # Определение функции для обработки кадра
        def process_frame(frame):
            results = model(frame)
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
def process_webcam():
    # Загрузка модели YOLO с весами оценки позы
    model = YOLO('yolov8n-pose.pt')
    
    # Открытие объекта захвата видео для вебкамеры
    cap = cv2.VideoCapture(0)

    # Основной цикл для захвата кадров видео и выполнения обнаружения объектов YOLO
    while(cap.isOpened()):
        # Чтение кадра из захвата видео
        success, frame = cap.read()

        if success:
            # Выполнение обнаружения объектов с помощью YOLO и сохранение результатов
            results = model(frame, save=True)
        
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

# Выберите режим: 'photo', 'video' или 'webcam'
mode = 'photo'

# Обработка в зависимости от выбранного режима
if mode == 'photo':
    annotated_image = process_image('photo.jpg')
    cv2.imshow('Yolo', annotated_image)
    cv2.imwrite('annotated_image.jpg', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif mode == 'video':
    process_video('gta5.mp4')
elif mode == 'webcam':
    process_webcam()
