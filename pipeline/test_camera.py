import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import json

# Cargar clases
with open("../models/classes.json") as f:
    class_indices = json.load(f)

labels = {v:k for k,v in class_indices.items()}

# Modelo
model = tf.keras.models.load_model("../models/mobilenet_letters.h5")

# MediaPipe
mp_hands = mp.solutions.hands.Hands()

# Cámara
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)

    if results.multi_hand_landmarks:

        h, w, _ = frame.shape

        for hand_landmarks in results.multi_hand_landmarks:

            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]

            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))

            # margen
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            hand_crop = frame[y_min:y_max, x_min:x_max]

            if hand_crop.size == 0:
                continue

            # Preprocesamiento
            img = cv2.resize(hand_crop, (224,224))
            img = img / 255.0
            img = np.expand_dims(img, 0)

            # Predicción
            prediction = model.predict(img, verbose=0)
            class_id = np.argmax(prediction)
            letter = labels[class_id]

            # Mostrar
            cv2.putText(frame, letter, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.rectangle(frame, (x_min,y_min),(x_max,y_max),(0,255,0),2)

    cv2.imshow("Detector", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()