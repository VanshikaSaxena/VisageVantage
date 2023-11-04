import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# reading the data
with open('C:/Users/saxen/OneDrive/Documents/facial_image_recognition/fer2013.csv') as file:
    data = file.readlines()
    data_array = np.array(data)
records = data_array.size

# Creating training, validation, and test set data
num_classes = 7
x_trg, y_trg, x_test, y_test, x_val, y_val = [], [], [], [], [], []
for i in range(1, records):
    try:
        emotion, img, usage = data_array[i].split(",")
        val = img.split(" ")
        pixels = np.array(val, 'float32')
        emotion = keras.utils.to_categorical(emotion, num_classes)
        if 'Training' in usage:
            y_trg.append(emotion)
            x_trg.append(pixels)
        elif "PrivateTest" in usage:
            y_val.append(emotion)
            x_val.append(pixels)
        else:
            y_test.append(emotion)
            x_test.append(pixels)
    except:
        print("", end="")

x_trg = np.array(x_trg, 'float32')
y_trg = np.array(y_trg, 'float32')
x_val = np.array(x_val, 'float32')
y_val = np.array(y_val, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_trg /= 255
x_val /= 255
x_test /= 255

x_trg = x_trg.reshape(x_trg.shape[0], 48, 48, 1)
x_trg = x_trg.astype('float32')
x_val = x_val.reshape(x_val.shape[0], 48, 48, 1)
x_val = x_val.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

model = Sequential()
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

img_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

batch_size = 256
steps_per_epoch_trg = x_trg.shape[0] // batch_size

img_generator_trg = img_gen.flow(x_trg, y_trg, batch_size=batch_size)

model.fit_generator(img_generator_trg, steps_per_epoch=steps_per_epoch_trg, epochs=50)

model.save('trained_img_model_1.h5')

# val_score = model.evaluate(x_val, y_val, verbose=0)
# print('Loss in validation data:', val_score[0])
# print('Accuracy of validation data:', 100 * val_score[1])
# test_score = model.evaluate(x_test, y_test, verbose=0)
# print('Loss in test data:', test_score[0])
# print('Accuracy of test data:', 100 * test_score[1])
#
# y_pred = model.predict(x_test)
# y_pred_list = [np.argmax(i) for i in y_pred]
# y_test_list = [np.argmax(i) for i in y_test]
#
# print('Accuracy Score:', accuracy_score(y_test_list, y_pred_list))
# print('Classification Report:\n', classification_report(y_test_list, y_pred_list))
# print('Confusion Matrix:\n', confusion_matrix(y_test_list, y_pred_list))
