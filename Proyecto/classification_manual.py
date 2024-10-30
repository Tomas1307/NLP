import os
import tkinter as tk
from tkinter import Label, Button, Scale
from PIL import Image, ImageTk
import pandas as pd
import re

# Ruta de las imágenes
image_dir = "data/memes_conjuntos"
csv_path = "data/textos/chistes_clasificados/clasificacion_memes.csv"

# Cargar progreso
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    reviewed_ids = set(df['id'].astype(str).tolist())  # Convertimos a string para coincidir con el formato del ID de imagen
else:
    df = pd.DataFrame(columns=["id", "descripcion", "label", "nivel_risa"])
    reviewed_ids = set()

# Obtener y ordenar lista de imágenes numéricamente
def extract_number(filename):
    match = re.search(r'\d+', filename)  # Extrae solo el número de la imagen
    return int(match.group()) if match else float('inf')

image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files = sorted(image_files, key=extract_number)  # Ordena numéricamente

# Filtra las imágenes que no han sido revisadas
image_files = [f for f in image_files if os.path.splitext(f)[0] not in reviewed_ids]

# Configurar ventana
root = tk.Tk()
root.title("Clasificación de Memes")

# Variables para controlar la interfaz
image_label = Label(root)
image_label.pack()
status_label = Label(root, text="")
status_label.pack()

# Control deslizante para el nivel de risa
laugh_scale = Scale(root, from_=1, to=5, orient="horizontal", label="Nivel de risa")
laugh_scale.pack()

# Función para guardar y mostrar siguiente imagen
def save_and_next(label):
    global current_image_index
    # Guardar en CSV
    image_file = image_files[current_image_index]
    image_id = os.path.splitext(image_file)[0]  # Obtener el ID sin la extensión
    nivel_risa = laugh_scale.get() if label else 0
    df.loc[len(df)] = [image_id, image_file, label, nivel_risa]
    df.to_csv(csv_path, index=False)
    
    # Mostrar siguiente imagen
    current_image_index += 1
    if current_image_index < len(image_files):
        show_image()
    else:
        status_label.config(text="¡Clasificación completa!")
        yes_button.config(state="disabled")
        no_button.config(state="disabled")

# Función para mostrar la imagen actual
def show_image():
    image_file = image_files[current_image_index]
    img_path = os.path.join(image_dir, image_file)
    img = Image.open(img_path)
    img.thumbnail((400, 400))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img
    status_label.config(text=f"Mostrando: {image_file} ({current_image_index + 1}/{len(image_files)})")

# Botones de clasificación
yes_button = Button(root, text="Gracioso", command=lambda: save_and_next(True))
yes_button.pack(side="left")
no_button = Button(root, text="No gracioso", command=lambda: save_and_next(False))
no_button.pack(side="right")

# Iniciar
current_image_index = 0
show_image()

root.mainloop()
