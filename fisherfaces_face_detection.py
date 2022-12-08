import sys
import os
import numpy as np
import cv2
import matplotlib as pyplot

training_data_folder_path = 'dataset/training-data'
test_data_folder_path = 'dataset/test-data'

haarcascade_frontalface = 'opencv_xml_files/ haarcascade_frontalface.xml'

def detect_face(input_img):
    try:
        image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    except:
        convert_image = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(convert_image, cv2.COLOR_BGR2GRAY)
                             
    face_cascade = cv2.CascadeClassifier('opencv_xml_files/haarcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return -1, -1
    (x, y, w, h) = faces[0]
    return image[y:y+w, x:x+h], faces[0]

def prepare_training_data(training_data_folder_path):
    detected_faces = []
    face_labels = []
    traning_image_dirs = os.listdir(training_data_folder_path)
    for dir_name in traning_image_dirs:
        label = int(dir_name)
        training_image_path = training_data_folder_path + "/" + dir_name
        training_images_names = os.listdir(training_image_path)
        
        for image_name in training_images_names:
            image_path = training_image_path  + "/" + image_name
            image = cv2.imread(image_path)
            face, rect = detect_face(image)
            if face is not -1:
                resized_face = cv2.resize(face, (121,121), interpolation = cv2.INTER_AREA)
                detected_faces.append(resized_face)
                face_labels.append(label)

    return detected_faces, face_labels

detected_faces, face_labels = prepare_training_data("dataset/training-data")

print("Total faces: ", len(detected_faces))
print("Total labels: ", len(face_labels))

fisherfaces_recognizer = cv2.face.FisherFaceRecognizer_create()

modified_face_labels = np.array(face_labels)
print(modified_face_labels)

fisherfaces_recognizer.train(detected_faces, np.array(face_labels))

def draw_rectangle(test_image, rect):
    (x, y, w, h) = rect
    cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(test_image, label_text, x, y):
    cv2.putText(test_image, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
def predict(test_image):
    detected_face, rect = detect_face(test_image)
    resized_test_image = cv2.resize(detected_face, (121,121), interpolation = cv2.INTER_AREA)
    label=fisherfaces_recognizer.predict(resized_test_image)
    label_text = tags[label[0]]
    draw_rectangle(test_image, rect)
    draw_text(test_image, label_text, rect[0], rect[1]-5)
    print(label)
    return test_image, label_text

tags = ['0', '1', '2', '3', '4', '5', '6']

cap = cv2.VideoCapture('test_recording.avi')
count = 0

while cap.isOpened():
    _, frame = cap.read()
    face, rect = detect_face(frame)
    if face is not -1:
        predicted_image, label = predict(face)
        count +=1
        if label == '0':
            count +=1
            draw_rectangle(frame, rect)
            cv2.imwrite("detected_faces/1/"+str(count)+ ".jpg", face)
cap.release()