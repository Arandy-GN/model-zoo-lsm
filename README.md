## Descripción

Este proyecto implementa un sistema de reconocimiento de letras del lenguaje de señas mexicanas utilizando visión por computadora y aprendizaje profundo. Se desarrolló un pipeline completo que abarca desde el preprocesamiento de imágenes hasta la predicción en tiempo real mediante el uso de una cámara.

El sistema emplea un modelo basado en MobileNetV2 para la clasificación de imágenes y MediaPipe para la detección de la mano, lo que permite mejorar la robustez del reconocimiento en entornos reales.

## Tecnologías utilizadas

Python
TensorFlow / Keras
OpenCV
MediaPipe

## Estructura del proyecto

```
model_zoo_lsm/
│
├── pipeline/
│   └── test_camera.py
│
├── training/
│   └── train_mobilenet.py
│
├── models/
├── datasets/
│
└── .gitignore
```

## Instalación

```
pip install -r requirements.txt
```

## Uso

Entrenamiento del modelo:

```
python training/train_mobilenet.py
```

Ejecución en tiempo real:

```
python pipeline/test_camera.py
```

## Dataset

Se utilizó un dataset del alfabeto para el entrenamiento del modelo. El dataset no se incluye en este repositorio debido a su tamaño.

## Resultados

El sistema permite detectar la mano en tiempo real y clasificar letras del alfabeto ASL utilizando una cámara web. El uso de un modelo preentrenado facilita la generalización en diferentes condiciones.

## Limitaciones

El modelo puede presentar dificultades al distinguir letras visualmente similares. Además, el desempeño puede verse afectado por condiciones extremas de iluminación o variaciones significativas en la posición de la mano.

## Trabajo futuro

Se propone mejorar la precisión mediante técnicas de aumento de datos, explorar el uso de keypoints en lugar de imágenes completas y extender el sistema hacia el reconocimiento de palabras completas.

## Autor

Arandy García Navarro
