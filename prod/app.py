import numpy as np
import torch
import streamlit as st
from model import VAE
from utils import load_model, load_dataset, model_interp
import os
from PIL import Image
import torchvision.transforms as T

# Configurar el título de la aplicación
st.title("Celebrity Maker")
st.write("¿Nunca te preguntaste como sería el hijo entre dos famosos? ¡Selecciona 2 y averigualo!")

# Cargar el modelo y los datos
model = VAE().to("cpu")
model = load_model(model)

# Cargar el dataset
dataset = load_dataset()

# Obtener nombres de actores desde las carpetas en el dataset
actor_names = sorted(dataset.classes)

# Limpiar los nombres de los actores eliminando los guiones bajos
clean_actor_names = [name.replace('_', ' ') for name in actor_names]

# Función para mostrar una imagen específica de un actor seleccionado
def display_specific_actor_image(actor_name, image_number):
    actor_folder = os.path.join(dataset.root, actor_name.replace(' ', '_'))
    image_files = sorted(os.listdir(actor_folder))
    image_index = min(image_number - 1, len(image_files) - 1)
    image_path = os.path.join(actor_folder, image_files[image_index])
    img = Image.open(image_path)
    st.image(img, caption=f"Imagen seleccionada de {actor_name}", use_container_width=True)
    return image_path

# Crear tres columnas: Actor 1, Fusión, Actor 2
col1, col2, col3 = st.columns([1.5, 1.5, 1.5])

# Columna para el Actor 1
with col1:
    st.header("Famoso nº 1")
    actor1 = st.selectbox("Selecciona el famoso nº 1", clean_actor_names)
    image_number1 = st.selectbox("Selecciona la imagen", [f"Imagen {i+1}" for i in range(10)])
    actor1_image = display_specific_actor_image(actor1, int(image_number1.split()[1]))

# Columna para el Actor 2
with col3:
    st.header("Famoso nº 2")
    actor2 = st.selectbox("Selecciona el famoso nº 2", clean_actor_names)
    image_number2 = st.selectbox("Selecciona la imagen", [f"Imagen {i+1}" for i in range(10)], key="img_select_actor2")
    actor2_image = display_specific_actor_image(actor2, int(image_number2.split()[1]))

# Columna central para la intensidad de fusión y el botón de interpolación
with col2:
    st.header("¡Fusión!")
    intensity = st.slider("Intensidad de la fusión", min_value=1, max_value=10, value=5) - 1
    to_pil = T.ToPILImage()

    # Botón para generar la interpolación
    if st.button("Fusionar"):
        if actor1_image and actor2_image:
            with st.spinner("Fusionando..."):
                start_idx = actor_names.index(actor1.replace(' ', '_')) * 10 + int(image_number1.split()[1]) - 1
                end_idx = actor_names.index(actor2.replace(' ', '_')) * 10 + int(image_number2.split()[1]) - 1
                interpolated_images = model_interp(model, dataset, start_idx, end_idx, size=10)
                
                # Seleccionar y mostrar la imagen interpolada según la intensidad
                selected_image_tensor = interpolated_images[intensity]
                selected_image_pil = to_pil(selected_image_tensor)
                st.image(selected_image_pil, caption=f"Interpolación con intensidad {intensity + 1}", use_container_width=True)
        else:
            st.warning("Por favor, selecciona una imagen para ambos famosos antes de generar la fusión.")
