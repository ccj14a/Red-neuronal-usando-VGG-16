import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Parámetros de la carga de imágenes
img_height, img_width = 150, 150

# Definir el mapeo de clases
class_names = [
    "Straight",  # recto
    "Wavy",  # ondulado suave
    "curly",  # ondulado mas definido(rizos)
    "dreadlocks",  # rastas
    "kinky",  # afro
]  # Clases en tu dataset

# Cargar el modelo entrenado
model = load_model(r"modelo_pelo.h5")
x = r"imag_pelo"

# Obtener la lista de imágenes
images = [f for f in os.listdir(x) if f.endswith((".png", ".jpg", ".jpeg"))]
print(f"Total de imágenes a procesar: {len(images)}")
for image_file in images:
    # Cargar la imagen
    img_path = os.path.join(x, image_file)
    img = cv2.imread(img_path)

    # Preprocesamiento de la imagen para la predicción
    img_resized = cv2.resize(img, (img_height, img_width))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Realizar la predicción
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    # Mostrar la clase en la consola
    print(f"Imagen: {image_file} - Predicción: {class_names[predicted_class[0]]}")
