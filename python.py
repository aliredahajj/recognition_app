import cv2
import face_recognition
import glob
import os

known_faces = []
known_names = []
known_faces_paths = []

data_faces_path = 'data/'
for name in os.listdir(data_faces_path):
    images_mask = '%s%s/*.jpg' % (data_faces_path, name)
    images_paths = glob.glob(images_mask)
    known_faces_paths += images_paths
    known_names += [name for x in images_paths]


def get_encodings(img_path):
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)
    return encoding[0]
    
known_faces = [get_encodings(img_path) for img_path in known_faces_paths]




cap = cv2.VideoCapture(0)

cap.set(3, 800)
cap.set(4, 600)
cap.set(5, 60)
fps = int(cap.get(5))
print(fps)
count = 0

show_fps = False
show_face = False
show_object = False

while True:
  
    ret, frame = cap.read()
    if not ret:
        break
    if show_fps:
        fps_prt = "FPS : %0.1f" % fps
        cv2.putText(frame, fps_prt, (20,80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    if show_face:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(frame_rgb)
        for face in faces:
            top, right, bottom, left = face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
            encoding = face_recognition.face_encodings(frame_rgb, [face])[0]
            results = face_recognition.compare_faces(known_faces, encoding)
            if any(results):
                name = known_names[results.index(True)]
            else:
                name = "Unknown"


            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    
    
    key = cv2.waitKey(1)
    if ord("q") == key:
        print("The app has been closed")
        break
    if ord("g") == key:
        show_fps = not show_fps
        msg =  "The FPS feature has been enabled" if show_fps == True else "The FPS feature has been disabled"
        print(msg)
    if ord("f") == key:
        show_face = not show_face
        msg =  "The face recognition feature has been enabled" if show_fps == True else "The face recognition feature has been disabled"
        print(msg)
    if ord("o") == key:
        show_object = not show_object
        msg =  "The object recognition feature has been enabled" if show_fps == True else "The object recognition feature has been disabled"
        print(msg)
    cv2.imshow("Recognition App", frame)
if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()

