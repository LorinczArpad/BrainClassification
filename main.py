import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
#Konstansok
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
IMG_HEIGHT, IMG_WIDTH = TARGET_SIZE
DATASET_PATH = "./input"
TRAINING_PATH = os.path.join(DATASET_PATH, "Training")
TESTING_PATH = os.path.join(DATASET_PATH, "Testing")
CLASS_NAMES = sorted(os.listdir(TRAINING_PATH))
SAVE_DIRECTORY ="./preproccesed"
SAVE_PREPROCESSED = True
#DataSeT készítése a megadott elérésiutak alapján
def load_dataset(folder_path):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder_path)) 
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            for img_file in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_file)
                if img_file.lower().endswith(('jpg', 'png', 'jpeg')):
                    # Előfeldolgozás
                    processed_image = preprocess_image(img_path)
                    #Kimentés
                 
                    images.append(processed_image.flatten()) # Normalizálás az SVM nek
                    labels.append(class_idx)  
    return np.array(images, dtype=np.float32), np.array(labels)

def preprocess_image(filepath):


    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    img = cv2.resize(img, TARGET_SIZE)
    
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobel_x, sobel_y)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
def get_processed_data():
    train_images, train_labels = load_dataset(TRAINING_PATH)
    test_images, test_labels = load_dataset(TESTING_PATH)
    return train_images / 255, train_labels,test_images/ 255 ,test_labels
def train_svm(train_images,train_labels):
    print("Training started.")
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(train_images, train_labels)
    print("Training ended.")
    return svm
def test_svm(test_images,test_labels,svm):
    predictions = svm.predict(test_images)
    print("Accuracy:", accuracy_score(test_labels, predictions))
    print("Classification Report:\n", classification_report(test_labels, predictions))
def main():

    train_images,train_labels,test_images,test_labels  = get_processed_data()   
    svm = train_svm( train_images,train_labels)
    test_svm(test_images,test_labels,svm)

if __name__ == "__main__":
    main()