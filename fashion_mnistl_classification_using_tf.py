import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# to prevent the tf errot of avx2
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#

data = keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels) = data.load_data()
# print(len(train_images))

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# print(train_labels[0])
# plt.imshow(train_images[3], cmap=plt.cm.binary)
# plt.show()
# plt.imshow(train_images[3])
# plt.show()
train_images = train_images/255.0
test_images = test_images/255.0
# print(train_images[0])
# print(len(train_images[0][0]))

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")
	])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)

# test_loss, test_acc = model.evaluate(test_images, test_labels)

# print("Tested Acc:",test_acc)

prediction = model.predict(test_images)

for i in range(5):
	plt.grid(False)
	plt.imshow(test_images[i], cmap=plt.cm.binary)
	plt.xlabel("Actual: "+ class_names[test_labels[i]])
	plt.title("prediction: "+ class_names[np.argmax(prediction[i])])
	plt.show()
# print(class_names[np.argmax(prediction[0])])

