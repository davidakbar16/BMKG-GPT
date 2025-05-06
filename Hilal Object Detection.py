#Install python 3.7 atau lebih dari itu

import cv2 # install cv = pip install opencv-python
from ultralytics import YOLO # install ultalytics = pip install ultralytrics


model = YOLO('D:\\Project hilal\\Project hilal\\best.pt') #path atau folder model ganti path dimana lokasi model berada


class_names = {
    0: "Hilal", # nilai 0 merupakan label pengenal citra hilal dan "Hilal" nama pengenal untuk label 
}

video_path = 'D:\\Project hilal\\Project hilal\\NTT2.mp4' #path untuk input video #ganti path sesuai dengan jenis input yang diinginkan Input di khususkan untuk video atau real time stream dari kamera atau webcam
video_cam = 0  #path untuk webcam / camera yang terhubung ke komputer  /// NILAI BISA 0, 1, 2,3 CEK nilai kamera terlebih dahulu
cap = cv2.VideoCapture(video_path) #ganti paTH VIDEO_CAM / VIDEO_PATH

if not cap.isOpened():
    print("error")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 
    frame_resized = cv2.resize(frame, (640, 480))  #atur frame sesuai kebutuhan dengan mengubah nilainya
    results = model(frame_resized)

  
    annotated_frame = frame_resized.copy()

   
    for result in results:
        boxes = result.boxes 
        for box in boxes:
            
            if box.conf[0] >= 0.3: # untuk menentukan ambang akurasi yang diinginkan dalam menampilkan citra
                x1, y1, x2, y2 = box.xyxy[0]  
                class_index = int(box.cls[0])  
                class_name = class_names.get(class_index, "Unknown") 
                confidence = box.conf[0] #digunakan untuk mendapatkan nilai akurasinya

  
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(annotated_frame, f'{class_name}',
                #cv2.putText(annotated_frame, f'{class_name} {confidence:.2f}', 
                            (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    

    cv2.imshow('Hilal Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
