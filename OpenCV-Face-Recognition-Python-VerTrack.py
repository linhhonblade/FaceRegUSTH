import cv2
import os
import sys
import numpy as np

subjects = ["unknown","Hung", "Mai", "Dung"]

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('classifier/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)
    if(len(faces)==0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;
        label = int(dir_name.replace("s",""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            face, rect = detect_face(image)
            if face is not None:
                faces.append(cv2.resize(face, (90, 100)))
                labels.append(label)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
#def predict(test_img):
    #face, rect = detect_face(test_img)
    #if face is not None:
        #label, confidence = face_recognizer.predict(cv2.resize(face,(90,100)))
    #lse:
        #label = 0
    #label_text = subjects[label]
    #return label_text
def predict(face):
    label, confidence = face_recognizer.predict(cv2.resize(face, (90, 100)))
    label_text = subjects[label]
    return label_text

def toIntTuple(floatTuple):
    intList = []
    for number in floatTuple:
        intNumber = int(number)
        intList.append(intNumber)
    intTuple = tuple(intList)
    return intTuple


##################### Train #################################
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared.")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

########### Create recognizer ###############################
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

############### Create tracker ##############################
tracker = cv2.TrackerMOSSE_create()

################### Main ####################################
video = cv2.VideoCapture("test-data/NguyenTrungDung.mp4")
ret, frame = video.read();
face, rect = detect_face(frame)
while True:
    if face is None:
        ret, frame = video.read()
        face, rect = detect_face(frame)
        imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video.release()
            cv2.destroyAllWindows() 
        continue
    if face is not None:
        predicted_name = predict(face)
        rect = tuple(rect)
        ok = tracker.init(frame, rect)
        while True:
            ret, frame = video.read()
            ok, rect = tracker.update(frame)
            if ok:
                rect = toIntTuple(rect)
                draw_rectangle(frame, rect)
                (x, y, w, h) = rect
                draw_text(frame, predicted_name, x, y)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    video.release()
                    cv2.destroyAllWindows()
            if not ok:
                face = None
                break
cv2.waitKey(0)
cv2.destroyAllWindows()















