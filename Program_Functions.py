import face_recognition
import cv2
import numpy as np
import csv
import os 
import time
from datetime import datetime

def encode(name):
        name_image = face_recognition.load_image_file(f".Photos/{name}")
        name_encoding = face_recognition.face_encodings(name_image)[0]
        return name_encoding

def face_rec():
    video_capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_locations = []
    face_encodings = []
    s=True
    mkr=1

    current_date =datetime.now().strftime("%Y-%m-%d")
    
    try:
        f = open(current_date+'.csv','a',newline = '')
        lnwriter = csv.writer(f)
    except:
        f = open(current_date+'.csv','w+',newline = '')
        lnwriter = csv.writer(f)

    known_face_encoding=[]
    known_faces_names=[]
    names_list=os.listdir('.\Photos')

    for name in names_list:
        enc=encode(name)
        known_face_encoding.append(enc)

    for name in names_list:
        a=name[-5::-1]
        b=a[::-1]
        known_faces_names.append(b)

    while True:
        if len(known_faces_names)==0 and mkr==1:
            print("!! The Registration is Empty, Please Enter '1' to Register !!")
            break
        elif len(known_faces_names)==0 and mkr==0:
            print("Everyone is present! Thank You for the Attendence..")
            break

        _,frame = video_capture.read()
        scale_percent=40
        width=int(frame.shape[1]*scale_percent/100)
        height=int(frame.shape[0]*scale_percent/100)
        dim=(width,height)
        small_frame=cv2.resize(frame,dim,fx=0.1,fy=0.1)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        if s:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
                face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]
                    current_date =datetime.now().strftime("%Y-%m-%d")
                    present_time =datetime.now().strftime("%H:%M:%S")

                if name in known_faces_names:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10,100)
                    fontScale              = 1.5
                    fontColor              = (255,0,0)
                    thickness              = 3
                    lineType               = 2
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame,name+' Present', 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
                        
                    print(f"{name} is present\t\t",current_date,present_time)
                    lnwriter.writerow([name,current_date,present_time])
                    index=known_faces_names.index(name)
                    known_face_encoding.pop(index)
                    known_faces_names.remove(name)
                    mkr=0
                    
        cv2.imshow("Facial Recognition for Attendance System",frame)            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    f.close()
    return 1

def Input():
    try:
        n1=int(input("Enter the Number of Person: "))
        cam = cv2.VideoCapture(0)
        for i in range(n1):
            inp = input("\nEnter person's name : ")
            no_of_take=3
            while(1): 
                    result,image = cam.read()                        
                    cv2.imshow(inp,image)
                    if cv2.waitKey(1):
                        path = '.\Photos'
                        cv2.imwrite(os.path.join(path ,inp+".jpg"), image)
                        time.sleep(2)
                        no_of_take-=1                       
                    if no_of_take==0:
                        print(f"{inp} is Successfully Registered\n")
                        break
        cam.release()
        cv2.destroyAllWindows()
        return 1
    except:
        print('#### Enter the correct input ####\n')
        return Input()