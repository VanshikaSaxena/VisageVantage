import os
from flask import Flask, render_template, request
import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

app = Flask(__name__)

def func_expression(emotions, newimage):
    bestemotion = np.argmax(emotions)
    emo_val = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
    mycolor = 'blue'  # default color
    emoji = None  # default emoji
    if bestemotion == 0:
        emoji = cv2.imread('images/angry.jpeg')
        mycolor = 'red'
    elif bestemotion == 1:
        emoji = cv2.imread('images/disgust.jpg')
        mycolor = 'orange'
    elif bestemotion == 2:
        emoji = cv2.imread('images/fear.png')
        mycolor = 'purple'
    elif bestemotion == 3:
        emoji = cv2.imread('images/happy.jpg')
        mycolor = 'green'
    elif bestemotion == 4:
        emoji = cv2.imread('images/sad.jpg')
        mycolor = 'yellow'
    elif bestemotion == 5:
        emoji = cv2.imread('images/neutral.jpeg')
        mycolor = 'cyan'

    if emoji is not None:
        plt.figure(figsize=(20, 7))
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(newimage, cv2.COLOR_BGR2RGB), interpolation='bicubic')
        y_pos = np.arange(len(emo_val))
        plt.subplot(132)
        plt.bar(y_pos, emotions, align='center', alpha=0.5, color=mycolor)
        plt.xticks(y_pos, emo_val)
        plt.xlabel('Emotions')
        plt.ylabel('Percentage')
        plt.title('Display the emotion from face')
        plt.subplot(133)
        plt.imshow(cv2.cvtColor(emoji, cv2.COLOR_BGR2RGB))
        plt.title('Emoji for the emotion')
        plt.savefig('static/result_plot.png')
        return emo_val[bestemotion], 'static/result_plot.png'
    else:
        print("No emoji found for the detected emotion.")
        return "No emotion detected", None


def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    cv2.imwrite('static/uploaded_image.png', frame)
    return 'static/uploaded_image.png'

def load_trained_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

def preprocess_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (48, 48))
    resized_img = resized_img.astype('float32')
    resized_img /= 255
    preprocessed_img = np.expand_dims(resized_img, axis=-1)
    return preprocessed_img

def predict_emotions(model, img):
    preprocessed_img = preprocess_image(img)
    emotions = model.predict(np.expand_dims(preprocessed_img, axis=0))[0]
    return emotions

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' not in request.form:
            return "No image part"

        image_data = request.form['image'].split(',')[1]
        with open('static/uploaded_image.png', "wb") as fh:
            fh.write(base64.b64decode(image_data))

        img = cv2.imread('static/uploaded_image.png')
        if img is None:
            return "Error in reading the image file."

        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        facecasc = cv2.CascadeClassifier(cascade_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            model = load_trained_model('trained_img_model_1.h5')
            pred_emotions = predict_emotions(model, img)
            predicted_emotion, plot_path = func_expression(pred_emotions, img)
            return render_template('upload.html', predicted_emotion=predicted_emotion, plot_path=plot_path)
        else:
            return render_template('upload.html', predicted_emotion="No face detected.", plot_path=None)

@app.route('/')
def index():
    return render_template('index.html', predicted_emotion=None)

@app.route('/upload')
def upload():
    return render_template('upload.html', predicted_emotion=None)

if __name__ == '__main__':
    app.run(debug=True)
