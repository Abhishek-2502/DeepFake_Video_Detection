import os
import cv2
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def extract_frames(video_path, label, max_frames=10):
    video_frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        video_frames.append([np.array(frame), label])
        frame_count += 1
    cap.release()
    return video_frames

def load_data(paths, label):
    start_time = time.time()
    data = []
    for path in paths:
        print(f"Processing path: {path}")
        for video in os.listdir(path):
            video_path = os.path.join(path, video)
            data.extend(extract_frames(video_path, label))
    end_time = time.time()
    print(f"Loaded data from {paths}. Time taken: {end_time - start_time:.2f} seconds")
    return data
# Paths to your datasets

real_paths = ["D:\\Dataset_Celeb_df\\Celeb-DF\\Celeb-real",
              "D:\\Dataset_Celeb_df\\Celeb-DF\\YouTube-real",
              "D:\\Dataset_Celeb_df\\Celeb-DF-v2\\Celeb-real",
              "D:\\Dataset_Celeb_df\\Celeb-DF-v2\\YouTube-real"]
synthetic_paths = ["D:\\Dataset_Celeb_df\\Celeb-DF-v2\\Celeb-synthesis",
                   "D:\\Dataset_Celeb_df\\Celeb-DF\\Celeb-synthesis"]

# Load and label data
real_data = load_data(real_paths, 0) # 0 for real
synthetic_data = load_data(synthetic_paths, 1) # 1 for synthetic

# Combine and split data
all_data = real_data + synthetic_data
X, y = zip(*all_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Data Augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(np.array(X_train), np.array(y_train), batch_size=32)

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

my_model = build_model()
# Compute class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Train the model
def train_model(model, train_generator, X_test, y_test, epochs=10):
    start_time = time.time()
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // 32,
        epochs=epochs,
        validation_data=(np.array(X_test) / 255.0, np.array(y_test)),
        class_weight=class_weights
    )
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    return history

train_history = train_model(my_model, train_generator, X_test, y_test, epochs=10)
# Evaluate the Model
def evaluate_model(model, X_test, y_test):
    start_time = time.time()
    loss, accuracy = model.evaluate(np.array(X_test) / 255.0, np.array(y_test))
    end_time = time.time()
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")

evaluate_model(my_model, X_test, y_test)
