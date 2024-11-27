import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("test.keras")
path = r'C:\Users\Anurath\Desktop\Projects\Intelligent Systems\LAB4\HandwrittenDigits'
images = [cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
image = images[0]
image = cv2.resize(image, (28, 28))
image = np.expand_dims(image, axis=0)
pred = model.predict(image)
print(f"Guess: {np.argmax(pred)}")
plt.imshow(images[0], cmap='gray')
plt.show()

#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img_resized = cv2.resize(img, (28, 28))
#     img_normalized = img_inverted / 255.0
#     img_expanded = np.expand_dims(img_normalized, axis=0)