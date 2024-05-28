import cv2 as cv

face_ref = cv.CascadeClassifier("face_ref.xml")
camera = cv.VideoCapture(0)

def face_detector(frame):
   optimazed_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
   faces = face_ref.detectMultiScale(optimazed_frame, scaleFactor=1.1, minSize=(300, 300), minNeighbors=5)
   return faces

def drawer_box(frame):
   for x, y, w, h in face_detector(frame):
      cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)

def close_window():
   camera.release()
   cv.destroyAllWindows()
   exit()

def main():
   while True:
      _, frame = camera.read()
      drawer_box(frame)
      cv.imshow('Face Recognition', frame)

      if cv.waitKey(1) & 0xFF == ord('q'):
         close_window()

if __name__ == '__main__':
   main()