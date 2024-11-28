# CelebrityMaker 🎭 - Deep Neural Network for Celebrity Face Fusion

[![Página Web](https://img.shields.io/badge/P%C3%A1gina_Web-Celebrity%20Maker-blue)](https://celebrity-maker.streamlit.app/)
[![Licencia](https://img.shields.io/github/license/JuliDir/CelebrityMaker?color=blue)](https://github.com/JuliDir/CelebrityMaker/blob/main/LICENSE)

¿Alguna vez quisiste combinar los rostros de tus celebridades favoritas en una sola imagen? **CelebrityMaker** te permite hacerlo realidad gracias al poder de las redes neuronales profundas. Este proyecto utiliza un Autoencoder Variacional (VAE) para generar imágenes realistas que combinan características de dos celebridades seleccionadas por el usuario.

<p align="center">
  <img src="./data/celebrity_fusion_example.png" alt="Ejemplo de fusión de rostros de celebridades" width="600">
</p>

**¿Listo para experimentar con combinaciones únicas?** 👉 [Prueba CelebrityMaker aquí](https://celebritymaker.streamlit.app/)

> Este proyecto fue desarrollado como parte del curso de Redes Neuronales Profundas, dirigido por el profesor Ing. Pablo Marinozi. Es un testimonio de creatividad, aprendizaje y tecnología de vanguardia. ❤️

---

## 📚 Tabla de Contenidos

1. [Características](#características)
2. [Organización del Repositorio](#-organización-del-repositorio)
3. [Detalles del Modelo VAE](#detalles-del-modelo-vae)
4. [Instalación y Ejecución](#-instalación-y-ejecución)

---

## ✨ Características

- **Selección de Celebridades**: Escoge entre una lista de celebridades populares.
- **Fusión de Rostros**: Genera mezclas visualmente atractivas utilizando un Autoencoder Variacional.
- **Interfaz Intuitiva**: Ajusta el nivel de mezcla y personaliza los resultados a tu gusto.
- **Modelos Avanzados**: Entrenado con PyTorch para garantizar imágenes fluidas y realistas.

---

## 🗂️ Organización del Repositorio

El repositorio está estructurado para facilitar el desarrollo y uso de la aplicación:

### 1. `data/`
- **celebrity_dataset.zip**: Contiene el conjunto de datos utilizado para el entrenamiento del modelo.
- **data_exploration.ipynb**: Análisis y preprocesamiento del conjunto de datos.

### 2. `dev/`
- **src/**: Código fuente del Autoencoder Variacional (VAE).
- **model_training.ipynb**: Notebook de Jupyter donde se entrena el modelo.
- **utils.py**: Funciones auxiliares para preprocesamiento y carga del modelo.

### 3. `prod/`
- **app.py**: Archivo principal que ejecuta la aplicación web con Streamlit.
- **models/vae.pt**: Pesos del modelo entrenado.
- **requirements.txt**: Lista de dependencias necesarias.

---

## 🤖 Detalles del Modelo VAE

El Autoencoder Variacional (VAE) es el núcleo del proyecto. Este modelo genera imágenes realistas a partir de combinaciones de rostros utilizando un espacio latente.

### Principales Componentes del Modelo:

1. **Codificador (Encoder)**:
   - Comprime las imágenes originales en un espacio latente reducido.

2. **Decodificador (Decoder)**:
   - Reconstruye imágenes a partir de representaciones latentes.

3. **BetaLoss**:
   - Combina pérdida de reconstrucción y regularización latente, controlada por un factor $\beta$.

---

## ⚙️ Instalación y Ejecución

Sigue estos pasos para ejecutar el proyecto en tu máquina local:
### 1. Clonar el repositorio
bash
$ git clone https://github.com/JuliDir/CelebrityMaker.git
### 2. Instalar dependencias
bash
$ pip install -r prod/requirements.txt
### 3. Ejecutar la aplicación
bash
$ streamlit run prod/app.py
