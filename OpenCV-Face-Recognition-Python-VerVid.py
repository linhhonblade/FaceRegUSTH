import math
import cv2
import os
import numpy as np
import time
subjects = ["", "Hung", "Mai Dien", "NguyenTrungDung"]
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('classifier/lbpcascade_frontalface.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighborfaces = face_cascade.detectMultiScale(gray, 1.3, 5)s=5);

    #if no faces are detected then return original img |modify
    if (len(faces) == 0):
        return None, None, 0

    #under the assumption that there will be only one face,|modify
    #extract the face area
    (x, y, w, h) = faces[0]

    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0], 1

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    #list to hold labels for all subjects
    labels = []
    #let's go through each directory and read images within it
    for dir_name in dirs:

        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;

        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        #build path of directory containin images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        #------STEP-3--------
        #go through each image name, read image,
        #detect face and add face to list of faces
        for image_name in subject_images_names:

            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)

            #display an image window to show the image
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)

            #detect face |modify
            face, rect,check = detect_face(image)

            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(cv2.resize(face, (90, 100)))
                #add label for this face
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels



def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#function to draw text on give image starting from
#passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face,rect,check = detect_face(img)

    #predict the image using our face recognizer  |HIEP modify
    if check==1:
        label, confidence = face_recognizer.predict(cv2.resize(face, (90, 100)))
    else:
        label=0
    #get name of respective label returned by face recognizer
    label_text = subjects[label]

    return label_text


############## main #####################

print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

print("Predicting images...")
face_cascade = cv2.CascadeClassifier('classifier/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('video.mp4')
#frameRate = cap.get(cv2.CAP_PROP_FPS)
#pre = 0
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

while(cap.isOpened()):
    #frameId = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #now = time.time()
    #print (1/(now-pre))
    #ret, frame = cap.read()
    #pre = now
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #predicted_name=''
    #if (frameId % (3*math.floor(frameRate))==0):
    predicted_name = predict(frame)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        draw_text(frame,predicted_name,x,y)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
