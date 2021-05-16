from keras.preprocessing.image_dataset import load_image
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from tensorflow.keras import utils
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


img = image.load_img("test8.jpg", target_size=(28, 28), color_mode="grayscale")
arr = image.img_to_array(img)
# Меняем форму массива в плоский вектор
arr = arr.reshape(1, 784)
# Инвертируем изображение
arr = 255 - arr
# Нормализуем изображение
arr = arr/255


# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Преобразование размерности изображений
# Полносвязная нейронная сеть не может работать с двумерными данными, поэтому
# изображения преобразуем в плоский вектор


# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)

# Делим интенсивность каждого пикселя на 255, поэтому данные для входа в нейронную
# сеть будут в диапазоне [0,1] (удобно для алгоритмов оптимизации, которые обучают
# нейронку)


# x_train = x_train/255
# x_test = x_test/255

# y_train = utils.to_categorical(y_train, 10)
# y_test = utils.to_categorical(y_test, 10)

# 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'сандалии', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

# создание нейронки

# model = Sequential()
#
# model.add(Dense(800, input_dim=784, activation="relu"))
# model.add(Dense(600, activation="relu"))
# model.add(Dense(250, activation="relu"))
# model.add(Dense(30, activation="relu"))
# model.add(Dense(10, activation="softmax"))
#
# model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
#
# print(model.summary())

# обучение
# model.fit(x_train,
#          y_train,
#          batch_size=100,
#          epochs=100,
#          verbose=1)
#
# model.save("neural.h5")

#запуск нейронки



model = load_model("neural.h5")

predictions = model.predict(arr)
predictions = np.argmax(predictions[0])

# номер класса нейросети
print(predictions)
print(classes[predictions])
#правильный ответ класса
#print(np.argmax(y_test[0]))

# сверточная

# classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'сандалии', 'рубашка', 'кроссовки', 'сумка', 'ботинки']
#
# np.random.seed(42)
#
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#
# #x_train = x_train.astype('float32')
# #x_test = x_test.astype('float32')
#
#
# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
#
# x_train = x_train/255
# x_test = x_test/255
#
# y_train = utils.to_categorical(y_train, 10)
# y_test = utils.to_categorical(y_test, 10)



# model = Sequential()
#
#
# model.add(Convolution2D(filters=28, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1),
#                         activation='relu', data_format="channels_last"))
# model.add(Convolution2D(filters=28, kernel_size=(3, 3),
#                         activation='relu', data_format="channels_last"))
# model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
# model.add(Dropout(0.25))
# model.add(Convolution2D(56, (3, 3), padding='valid', activation='relu',data_format="channels_last"))
# model.add(Convolution2D(56, (3, 3), activation='relu',data_format="channels_last"))
# model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))
#
# model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
#
# model.fit(x_train,
#           y_train,
#           batch_size=32,
#           epochs=25,
#           validation_split=0.1,
#           shuffle=True)
#
# model.save("neural2.h5")

