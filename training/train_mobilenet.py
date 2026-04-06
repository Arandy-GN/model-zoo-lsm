import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
import json

IMG_SIZE = 224
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    "../datasets/asl_alphabet_train",
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="training"
)

val_data = datagen.flow_from_directory(
    "../datasets/asl_alphabet_train",
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation"
)

# 🔥 IMPORTANTE: detectar número de clases automáticamente
num_classes = train_data.num_classes
print("Número de clases:", num_classes)
print("Clases:", train_data.class_indices)

# Guardar clases para usar en cámara
with open("../models/classes.json", "w") as f:
    json.dump(train_data.class_indices, f)

# Modelo base
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE,IMG_SIZE,3)
)

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128,activation="relu")(x)

# 🔥 AQUÍ ESTÁ EL CAMBIO
output = layers.Dense(num_classes,activation="softmax")(x)

model = tf.keras.Model(base_model.input,output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
    
)

model.save("../models/mobilenet_letters.h5")