from flask import Flask, request, jsonify
import logging
import cv2
from ultralytics import YOLO
import concurrent.futures
import numpy as np
import sys
import argparse
import json
import os

app = Flask(__name__)

@app.route('/api/keypoints', methods=['POST'])
def get_keypoints():
    # Проверяем, что POST запрос содержит файл
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Проверяем, что файл имеет допустимое расширение
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        # Сохраняем файл во временную директорию
        filepath = os.path.join(os.getcwd(), file.filename)
        file.save(filepath)
        def convert_numpy_to_list(num_array):
            return num_array.tolist()
        def process_video(filepath, yolo_model='yolov8n-pose.pt', confidence_threshold=0.5):
            model = YOLO(yolo_model)
            cap = cv2.VideoCapture(filepath)

            def process_frame(frame):
                results = model(frame, conf=confidence_threshold)
                keypoints = {
                    '0_nose': [],
                    'left_eye': [],
                    'right_eye': [],
                    'left_ear': [],
                    'right_ear': [],
                    'left_shoulder': [],
                    'right_shoulder': [],
                    'left_elbow': [],
                    'right_elbow': [],
                    'left_wrist': [],
                    'right_wrist': [],
                    'left_hip': [],
                    'right_hip': [],
                    'left_knee': [],
                    'right_knee': [],
                    'left_ankle': [],
                    'right_ankle': []
                }
                for person in results[0].boxes.data.cpu().numpy():
                    for idx, kp in enumerate(results[0].keypoints[0].xy.cpu().numpy()):
                        keypoints[f'{idx // 3}_{["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"][idx % 3]}'].append(kp)
                return keypoints

            keypoints_list = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                while(cap.isOpened()):
                    success, frame = cap.read()
                    if success:
                        keypoints_list.append(executor.submit(process_frame, frame).result())
                    else:
                        break

            cap.release()

            def convert_numpy_to_list(num_array):
                if isinstance(num_array, np.ndarray):
                    return num_array.tolist()
                return num_array

            keypoints_list = [[convert_numpy_to_list(kp) for kp in keypoints.values()] for keypoints in keypoints_list]

            # Преобразование вложенных списков numpy в списки Python
            def convert_nested_numpy_to_list(data):
                if isinstance(data, np.ndarray):
                    return data.tolist()
                elif isinstance(data, list):
                    return [convert_nested_numpy_to_list(item) for item in data]
                return data

            # Преобразование всех вложенных numpy массивов в списки
            keypoints_list = convert_nested_numpy_to_list(keypoints_list)

            return json.dumps(keypoints_list)

        result = process_video(filepath)
        return jsonify(result)
    else:
        return jsonify({'error': 'File type not allowed'})

# Функция для проверки разрешенных расширений файлов
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}  # Допустимые расширения файлов
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)