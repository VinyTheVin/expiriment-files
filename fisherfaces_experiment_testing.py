import numpy as np
import cv2
import os

haarcascade_frontalface = 'opencv_xml_files/haarcascade_frontalface.xml'

def detect_face(input_img):
    """
    detect faces from an input image
    return: detected face and its postions that is x,y,w,h
    """
    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(haarcascade_frontalface)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return -1, -1
    (x, y, w, h) = faces[0]
    return image[y:y+w, x:x+h], faces[0]

def draw_rectangle(img, rect):
    """
    draws rectangular bounding box around detected face
    """
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    """
    put label above the box
    """
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

detected_faces = []
face_labels = []

def prepare_training_data(training_data_folder_path):
    """
    read images from folder and prepare training dataset
    return list of detected face and labels
    """
    traning_image_dirs = os.listdir(training_data_folder_path)
    for dir_name in traning_image_dirs:
        if dir_name != '.DS_Store':
            label = int(dir_name)
            training_image_path = training_data_folder_path + "/" + dir_name
            training_images_names = os.listdir(training_image_path)
            for image_name in training_images_names:
                image_path = training_image_path  + "/" + image_name
                image = cv2.imread(image_path, 0)
                detected_faces.append(image)
                face_labels.append(label)

    return detected_faces, face_labels

detected_faces, face_labels = prepare_training_data('detected_faces')    

print("Total faces: ", len(detected_faces))
print("Total labels: ", len(face_labels))

eigenfaces_recognizer = cv2.face. EigenFaceRecognizer_create()
eigenfaces_recognizer.train(detected_faces, np.array(face_labels))

count = 0
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    face, rect = detect_face(frame)
    resized_face = cv2.resize(face, (308, 308), interpolation = cv2.INTER_AREA)
    if face is not -1:
        label = eigenfaces_recognizer.predict(resized_face)
        print(label)
        if label[1]<12500:
            label_text = str(label[0])
        else:
            label_text = 'unknown'
        draw_rectangle(frame, rect)
        if label[1]<12500:
            count +=1
            draw_text(frame, label_text, rect[0], rect[1]-5)
            cv2.imwrite("fisherfaces/1/"+str(count)+ ".jpg", face)
        else:
            draw_text(frame, label_text, rect[0], rect[1]-5)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
