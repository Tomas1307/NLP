import os
import tkinter as tk
from tkinter import Label, Button, Scale
import pandas as pd

# Ruta del archivo CSV
csv_path = "data/textos/complete_text_data.csv"
output_csv_path = "data/textos/chistes_clasificados/clasificacion_chistes.csv"

# Cargar dataset de chistes
df_texto = pd.read_csv(csv_path)
df_texto["id"] = range(2007, 2007 + len(df_texto))

# Cargar progreso anterior si existe
if os.path.exists(output_csv_path):
    df_result = pd.read_csv(output_csv_path)
    reviewed_texts = set(df_result['id'].tolist())
else:
    df_result = pd.DataFrame(columns=["id", "Chistes", "label", "nivel_risa"])
    reviewed_texts = set()

# Filtrar los chistes que aún no han sido revisados
df_texto = df_texto[~df_texto["id"].isin(reviewed_texts)]
texts = df_texto[["id", "Chistes"]].values.tolist()

# Configurar ventana
root = tk.Tk()
root.title("Clasificación de Chistes")

# Variables para controlar la interfaz
chiste_label = Label(root, wraplength=400, justify="left")
chiste_label.pack(pady=20)
status_label = Label(root, text="")
status_label.pack()

# Control deslizante para el nivel de risa
laugh_scale = Scale(root, from_=1, to=5, orient="horizontal", label="Nivel de risa")
laugh_scale.pack()

# Función para guardar y mostrar el siguiente chiste
def save_and_next(label):
    global current_text_index
    # Guardar en CSV
    chiste_id, chiste_text = texts[current_text_index]
    nivel_risa = laugh_scale.get() if label else 0
    df_result.loc[len(df_result)] = [chiste_id, chiste_text, label, nivel_risa]
    df_result.to_csv(output_csv_path, index=False)
    
    # Mostrar el siguiente chiste
    current_text_index += 1
    if current_text_index < len(texts):
        show_text()
    else:
        status_label.config(text="¡Clasificación completa!")
        yes_button.config(state="disabled")
        no_button.config(state="disabled")

# Función para mostrar el chiste actual
def show_text():
    chiste_id, chiste_text = texts[current_text_index]
    chiste_label.config(text=chiste_text)
    status_label.config(text=f"Chiste {chiste_id} ({current_text_index + 1}/{len(texts)})")

# Botones de clasificación
yes_button = Button(root, text="Gracioso", command=lambda: save_and_next(True))
yes_button.pack(side="left", padx=20, pady=20)
no_button = Button(root, text="No gracioso", command=lambda: save_and_next(False))
no_button.pack(side="right", padx=20, pady=20)

# Iniciar
current_text_index = 0
show_text()

root.mainloop()
