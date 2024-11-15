# CelebrityMaker - Deep Neural Network for Celebrity Face Fusion

CelebrityMaker es una aplicación de fusión de rostros de celebridades, impulsada por redes neuronales profundas. La app permite a los usuarios seleccionar entre dos celebridades y generar una mezcla única de sus rostros mediante el uso de un Autoencoder Variacional (VAE) para crear imágenes realistas y personalizadas.

## Características

- **Selección de Celebridades**: Elige entre una lista de celebridades populares.
- **Fusión de Rostros**: Utilizando un modelo de Autoencoder VAE, la app genera mezclas realistas entre los rostros seleccionados.
- **Interfaz Intuitiva**: Ajusta el nivel de fusión y personaliza el resultado.
- **Modelos de Redes Neuronales Profundas**: Entrenado con un Autoencoder VAE implementado en PyTorch para lograr una fusión fluida y realista.

## Tecnologías Utilizadas

- **Autoencoder Variacional (VAE)**: Un modelo de aprendizaje profundo diseñado para la generación de imágenes con alta calidad y realismo.
- **PyTorch**: Framework utilizado para la implementación y entrenamiento del modelo de Autoencoder VAE.
- **Procesamiento de Imágenes**: Transformaciones previas para el alineamiento y la normalización de rostros.

## Ejecución Local

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tuusuario/CelebrityMaker.git
   cd CelebrityMaker

## Uso

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/pabloMarinozi/ejemploGlobalRNP.git
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecutar app de streamlit:
   ```bash
   streamlit run prod/app.py
   ```