import cv2  # Importa la librería OpenCV para procesamiento de video y manejo de imágenes
import mediapipe as mp  # Importa la librería MediaPipe para detección y seguimiento de poses

# Inicializa las utilidades de dibujo de MediaPipe y la solución de pose
mp_drawing = mp.solutions.drawing_utils  # Utilidad para dibujar anotaciones sobre la imagen
mp_pose = mp.solutions.pose  # Módulo para la detección de poses
isFullscreen = True  # Variable que puede usarse para definir si la ventana se mostrará a pantalla completa (no se usa en este fragmento)


def line_intersection(p1, p2, p3, p4):
    """
    Calcula el punto de intersección de dos líneas, cada una definida por dos puntos.
    p1, p2: Coordenadas (x, y) de la primera línea.
    p3, p4: Coordenadas (x, y) de la segunda línea.

    Retorna el punto de intersección como una tupla (px, py) si existe, o None en caso de líneas paralelas.
    """
    # Desempaqueta las coordenadas de cada punto
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Calcula el denominador de las fórmulas de intersección
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denom == 0:
        # Si el denominador es 0, las líneas son paralelas y no tienen intersección
        return None

    # Calcula la coordenada x del punto de intersección usando la fórmula de intersección
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    # Calcula la coordenada y del punto de intersección usando la fórmula de intersección
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    # Retorna las coordenadas convertidas a enteros para poder dibujarlas en la imagen
    return int(px), int(py)


# Inicializa la captura de video desde la cámara por defecto (generalmente la webcam)
cap = cv2.VideoCapture(0)

# Configura la resolución del video a 1920x1080 (Full HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Establece el ancho del frame a 1920 píxeles
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Establece el alto del frame a 1080 píxeles

# Inicializa la solución de pose de MediaPipe en modo de video en tiempo real
with mp_pose.Pose(static_image_mode=False) as pose:
    # Bucle principal que se ejecuta mientras la cámara esté abierta
    while cap.isOpened():
        ret, frame = cap.read()  # Lee un frame del video; 'ret' indica si la lectura fue exitosa y 'frame' contiene la imagen capturada
        if not ret:
            # Si no se pudo leer el frame, se sale del bucle
            break

        # Invierte horizontalmente la imagen para que el video sea como un espejo
        frame = cv2.flip(frame, 1)

        # Obtiene las dimensiones del frame (alto, ancho y canales de color)
        height, width, _ = frame.shape

        # Convierte la imagen de BGR (formato por defecto de OpenCV) a RGB (formato que utiliza MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesa el frame para detectar la pose utilizando MediaPipe
        results = pose.process(frame_rgb)

        # Calcula el punto central de la cámara (centro del frame)
        center_cam = (width // 2, height // 2)
        # Dibuja un círculo rojo en el centro del frame para marcar el punto central
        cv2.circle(frame, center_cam, 5, (0, 0, 255), -1)

        # Verifica si se han detectado landmarks (puntos clave) de la pose
        if results.pose_landmarks is not None:
            # Extrae la lista de landmarks detectados
            landmarks = results.pose_landmarks.landmark

            # Calcula las coordenadas de los puntos de interés redimensionándolas al tamaño del frame
            # Landmarks 12 y 23 representan puntos específicos (por ejemplo, puntos del hombro o cadera)
            p12 = (int(landmarks[12].x * width), int(landmarks[12].y * height))
            p23 = (int(landmarks[23].x * width), int(landmarks[23].y * height))
            # Landmarks 11 y 24 representan otros puntos clave correspondientes a la pose
            p11 = (int(landmarks[11].x * width), int(landmarks[11].y * height))
            p24 = (int(landmarks[24].x * width), int(landmarks[24].y * height))

            # Dibuja una línea verde entre los puntos 12 y 23
            cv2.line(frame, p12, p23, (0, 255, 0), 2)
            # Dibuja una línea roja entre los puntos 11 y 24
            cv2.line(frame, p11, p24, (0, 0, 255), 2)

            # Calcula el punto de intersección entre las dos líneas dibujadas
            intersection = line_intersection(p12, p23, p11, p24)
            if intersection:
                # Si se obtuvo una intersección, dibuja un círculo amarillo en ese punto
                cv2.circle(frame, intersection, 5, (255, 255, 0), -1)

                # Crea un nuevo punto que tiene la misma coordenada x que el centro de la cámara
                # y la coordenada y del punto de intersección, para formar un triángulo
                new_point = (center_cam[0], intersection[1])
                # Dibuja un círculo cian en el nuevo punto
                cv2.circle(frame, new_point, 5, (0, 255, 255), -1)

                # Dibuja una línea azul horizontal desde el punto de intersección hasta el nuevo punto
                cv2.line(frame, intersection, new_point, (255, 0, 0), 2)
                # Dibuja una línea azul vertical desde el nuevo punto hasta el centro de la cámara
                cv2.line(frame, new_point, center_cam, (255, 0, 0), 2)
                # Dibuja una línea rosa entre el punto de intersección y el centro de la cámara
                cv2.line(frame, intersection, center_cam, (255, 0, 255), 2)

                # Muestra las coordenadas del punto de intersección en la imagen, usando una fuente sencilla
                cv2.putText(frame, f"{intersection}", intersection, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Muestra el frame procesado en una ventana titulada "Frame"
        cv2.imshow("Frame", frame)
        # Espera 1 ms a que se presione la tecla 'Esc' (código ASCII 27) para salir del bucle
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Libera la captura de video y cierra todas las ventanas abiertas para limpiar los recursos
cap.release()
cv2.destroyAllWindows()
