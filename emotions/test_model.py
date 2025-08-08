import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model_path = "Black_Box.h5"
model = tf.keras.models.load_model(model_path)

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the test image in grayscale
image_path = "test_image.jpg"  # Ensure this file exists
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"❌ ERROR: Cannot read image at {image_path}. Check if the file exists!")
    exit()

# Resize and preprocess the image
image = cv2.resize(image, (48, 48))  # Resize to match model input size
image = image.astype('float32') / 255.0  # Normalize pixel values
image = np.expand_dims(image, axis=0)  # Add batch dimension
image = np.expand_dims(image, axis=-1)  # Add grayscale channel

# Make a prediction
prediction = model.predict(image)

# Print all emotion probabilities
print("\n🔹 Raw Model Prediction Probabilities:")
for i, label in enumerate(emotion_labels):
    print(f"{label}: {prediction[0][i]:.2f}")

# Get the emotion with the highest probability
emotion_index = np.argmax(prediction)
confidence = prediction[0][emotion_index]
emotion = emotion_labels[emotion_index]

print(f"\n✅ Final Detected Emotion: {emotion} ({confidence:.2f})")
