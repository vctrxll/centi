import cv2
import mediapipe as mp
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
isFullscreen = True

def line_intersection(p1, p2, p3, p4):
    """
    Calcula el punto de intersección entre dos líneas definidas por (p1, p2) y (p3, p4).
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # No hay intersección (líneas paralelas)

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    return int(px), int(py)

cap = cv2.VideoCapture(0)

with mp_pose.Pose(static_image_mode=False) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Centro de la cámara
        center_cam = (width // 2, height // 2)
        cv2.circle(frame, center_cam, 5, (0, 0, 255), -1)  # Punto rojo en el centro

        if results.pose_landmarks is not None:
            landmarks = results.pose_landmarks.landmark

            # Obtener coordenadas de puntos clave
            p12 = (int(landmarks[12].x * width), int(landmarks[12].y * height))
            p23 = (int(landmarks[23].x * width), int(landmarks[23].y * height))
            p11 = (int(landmarks[11].x * width), int(landmarks[11].y * height))
            p24 = (int(landmarks[24].x * width), int(landmarks[24].y * height))


            # Dibujar líneas entre puntos (para referencia)
            cv2.line(frame, p12, p23, (0, 255, 0), 2)  # Línea verde de 12 a 23
            cv2.line(frame, p11, p24, (0, 0, 255), 2)  # Línea roja de 11 a 24

            # Calcular el centro del cuerpo (intersección de las líneas)
            intersection = line_intersection(p12, p23, p11, p24)

            if intersection:
                # Dibujar el punto de intersección (centro del cuerpo)
                cv2.circle(frame, intersection, 5, (255, 255, 0), -1)  # Punto amarillo

                # Crear un punto con x en el centro de la pantalla y y igual al del centro del cuerpo
                new_point = (center_cam[0], intersection[1])
                cv2.circle(frame, new_point, 5, (0, 255, 255), -1)  # Punto cian para referencia

                # Dibujar los catetos en azul:
                # 1. Del centro del cuerpo (intersection) al nuevo punto (horizontal)
                cv2.line(frame, intersection, new_point, (255, 0, 0), 2)
                # 2. Del nuevo punto al centro de la cámara (vertical)
                cv2.line(frame, new_point, center_cam, (255, 0, 0), 2)
                
                # Dibujar una línea rosa entre la intersección y el centro de la cámara
                cv2.line(frame, intersection, center_cam, (255, 0, 255), 2)

                # Mostrar coordenadas en la imagen
                cv2.putText(frame, f"{intersection}", intersection, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


cap.release()
cv2.destroyAllWindows()
