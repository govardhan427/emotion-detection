from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Load the model
model = load_model("Black_Box.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
last_emotion = "Waiting..."

# Upload folder config
UPLOAD_FOLDER = 'static/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Haar Cascade is not loaded properly")

def predict_emotion(face):
    global last_emotion
    try:
        # Check image shape
        print(f"Image shape: {face.shape}")
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(face_gray, 1.3, 5)
        
        if len(faces) == 0:
            return "No Face Detected"

        for (x, y, w, h) in faces:
            roi_gray = face_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = model.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                last_emotion = label
                return label
    except cv2.error as e:
        print(f"Error in predict_emotion: {e}")
        return "Error in face detection"
    
    return "No Face Detected"

def generate_frames():
    global last_emotion
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            emotion = predict_emotion(frame)
            cv2.putText(frame, f'{emotion}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', current_emotion=last_emotion, uploaded_result=None)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        img = cv2.imread(save_path)
        if img is None:
            print("Error: Uploaded image is not valid")
            return redirect(url_for('index'))  # or show an error message

        # Resize image (optional but helps with face detection)
        img = cv2.resize(img, (640, 480))

        label = predict_emotion(img)

        # Create the 'static/uploaded' folder if it doesn't exist
        static_upload_path = os.path.join('static', 'uploaded')
        os.makedirs(static_upload_path, exist_ok=True)

        # Move the image to static/uploaded/ folder
        os.rename(save_path, os.path.join(static_upload_path, filename))

        # Generate the URL for the uploaded image
        image_url = url_for('static', filename=f'uploaded/{filename}')

        # Pass uploaded_result (label) to the template
        return render_template('index.html', current_emotion=last_emotion, uploaded_result=label, uploaded_image=image_url)

if __name__ == '__main__':
    app.run(debug=True)