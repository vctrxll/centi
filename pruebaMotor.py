import cv2
import mediapipe as mp
import RPi.GPIO as GPIO
import time

# =======================
# Configuración del Motor (DRV8825)
# =======================
GPIO.setmode(GPIO.BOARD)  # Usamos numeración física
PIN_DIR = 24  # Pin de dirección
PIN_STEP = 26  # Pin de paso

GPIO.setup(PIN_DIR, GPIO.OUT)
GPIO.setup(PIN_STEP, GPIO.OUT)


def move_motor(steps, direction, pause=0.005):
    """
    Mueve el motor paso a paso.

    Parámetros:
      steps (int): Cantidad de pasos a mover.
      direction (int): Dirección del giro (por ejemplo, 0 para un sentido, 1 para el opuesto).
      pause (float): Tiempo de espera (en segundos) entre pulsos, controla la velocidad.
    """
    GPIO.output(PIN_DIR, direction)
    for _ in range(steps):
        GPIO.output(PIN_STEP, True)
        time.sleep(pause)
        GPIO.output(PIN_STEP, False)
        time.sleep(pause)


# =======================
# Configuración de MediaPipe y OpenCV
# =======================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def line_intersection(p1, p2, p3, p4):
    """
    Calcula el punto de intersección entre dos líneas definidas por los puntos p1-p2 y p3-p4.
    Retorna (px, py) si existe intersección, o None si son paralelas.
    """
    x1, y1 = p1;
    x2, y2 = p2
    x3, y3 = p3;
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return int(px), int(py)


# Inicializa la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# =======================
# Parámetros para el ajuste del motor
# =======================
threshold_px = 20  # Error mínimo en píxeles para activar el movimiento
calibration_factor = 100.0  # Número de píxeles de error equivalentes a 1 paso (valor a calibrar)

# =======================
# Proceso de detección de pose y control del motor
# =======================
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Invierte la imagen para efecto espejo y obtiene dimensiones
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            center_cam = (width // 2, height // 2)
            cv2.circle(frame, center_cam, 5, (0, 0, 255), -1)  # Marca el centro de la cámara

            # Convierte la imagen a RGB para MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks is not None:
                landmarks = results.pose_landmarks.landmark

                # Se eligen los puntos para formar las líneas de referencia.
                # Por ejemplo:
                p12 = (int(landmarks[12].x * width), int(landmarks[12].y * height))
                p23 = (int(landmarks[23].x * width), int(landmarks[23].y * height))
                p11 = (int(landmarks[11].x * width), int(landmarks[11].y * height))
                p24 = (int(landmarks[24].x * width), int(landmarks[24].y * height))

                # Se calcula la intersección de las dos líneas.
                intersection = line_intersection(p12, p23, p11, p24)
                if intersection:
                    cv2.circle(frame, intersection, 5, (255, 255, 0), -1)

                    # Se crea un nuevo punto que comparte la coordenada x con el centro de la cámara
                    # y la coordenada y del punto de intersección.
                    new_point = (center_cam[0], intersection[1])
                    cv2.circle(frame, new_point, 5, (0, 255, 255), -1)

                    # Líneas de referencia (opcional)
                    cv2.line(frame, intersection, new_point, (255, 0, 0), 2)
                    cv2.line(frame, new_point, center_cam, (255, 0, 0), 2)
                    cv2.line(frame, intersection, center_cam, (255, 0, 255), 2)

                    # Calcula la diferencia horizontal (error) entre el centro del cuerpo y el centro de la cámara
                    # Dado que new_point tiene la x centrada, la diferencia se puede calcular directamente sobre el eje x.
                    # Aquí error_px será 0 si el centro del cuerpo (proyectado) coincide con el centro horizontal.
                    error_px = new_point[0] - center_cam[0]

                    # Muestra el valor del error en la imagen
                    cv2.putText(frame, f"Error: {error_px}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Si el error es mayor al umbral, se mueve el motor
                    if abs(error_px) > threshold_px:
                        # Se determina el número de pasos a mover (proporcional al error)
                        steps = int(abs(error_px) / calibration_factor)
                        if steps < 1:
                            steps = 1

                        # La dirección dependerá del signo del error:
                        # Si el error es positivo, significa que el centro proyectado está a la derecha del centro de la cámara;
                        # así que se puede definir, por ejemplo, que el motor gire hacia la izquierda (ajusta según tu mecánica).
                        if error_px > 0:
                            motor_direction = 0  # Sentido horario (ajusta según el montaje)
                        else:
                            motor_direction = 1  # Sentido antihorario

                        print(f"Moviendo motor {steps} pasos, dirección: {motor_direction}")
                        move_motor(steps, motor_direction)

            # Muestra el frame resultante
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Presiona Esc para salir
                break

    except KeyboardInterrupt:
        print("Programa interrumpido por el usuario.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()
