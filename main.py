import os
import joblib
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Konstansok
TARGET_SIZE = (224, 224)
IMG_HEIGHT, IMG_WIDTH = TARGET_SIZE
DATASET_PATH = "./input"
TRAINING_PATH = os.path.join(DATASET_PATH, "Training")
TESTING_PATH = os.path.join(DATASET_PATH, "Testing")
CLASS_NAMES = sorted(os.listdir(TRAINING_PATH))
SAVE_DIRECTORY ="./preproccesed"
SAVE_PREPROCESSED = True
MODEL_PATH = "svm_model.pkl"
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
                    if(class_name == 'glioma' and SAVE_PREPROCESSED and folder_path.endswith('Training')):
                        filename = os.path.basename(img_path)
                        save_path = os.path.join(SAVE_DIRECTORY, filename)
                        cv2.imwrite(save_path, processed_image)

                    images.append(processed_image.flatten()) # Normalizálás az SVM nek
                    labels.append(class_idx)  
    return np.array(images, dtype=np.float32), np.array(labels)

def preprocess_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    img = cv2.resize(img, TARGET_SIZE)
    #img = cv2.equalizeHist(img)    
    img = cv2.GaussianBlur(img, (3, 3), 0)

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobel_x, sobel_y)
    
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX) 
    return edges
    
def get_processed_data():
    train_images, train_labels = load_dataset(TRAINING_PATH)
    test_images, test_labels = load_dataset(TESTING_PATH)
    return train_images,train_labels,test_images,test_labels
def train_svm(train_images,train_labels):
    print("Training started.")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(train_images, train_labels)
    print("Training ended.")
    joblib.dump(svm, MODEL_PATH,compress=1)
    return svm
def test_svm(test_images,test_labels,svm,save_path="evaluation.png"):
    predictions = svm.predict(test_images)
    
    accuracy = accuracy_score(test_labels, predictions)
    print("Accuracy:", accuracy)
    
    report = classification_report(test_labels, predictions, output_dict=True)
    print("Classification Report:\n", classification_report(test_labels, predictions))
    
    cm = confusion_matrix(test_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f"{save_path.split('.')[0]}_confusion_matrix.png")
    plt.close()


    categories = list(report.keys())[:-3]  
    precision = [report[cat]['precision'] for cat in categories]
    recall = [report[cat]['recall'] for cat in categories]
    f1_score = [report[cat]['f1-score'] for cat in categories]

    x = np.arange(len(categories))  
    width = 0.25  

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label='Precision', color='skyblue')
    plt.bar(x, recall, width, label='Recall', color='salmon')
    plt.bar(x + width, f1_score, width, label='F1 Score', color='limegreen')
    
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.title('Classification Report')
    plt.xlabel('Categories')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"{save_path.split('.')[0]}_classification_report.png")
    plt.close()
    
    print("Accuracy:", accuracy_score(test_labels, predictions))
    print("Classification Report:\n", classification_report(test_labels, predictions))
def validate_svm_data(images, labels, data_name="Training"):
    if len(images.shape) != 2:
        raise ValueError(f"{data_name} images should be 2D with shape (n_samples, n_features). Current shape: {images.shape}")
    
    if not np.issubdtype(images.dtype, np.number):
        raise ValueError(f"{data_name} images should contain numerical values. Current dtype: {images.dtype}")
    
    if len(labels.shape) != 1:
        raise ValueError(f"{data_name} labels should be 1D with shape (n_samples,). Current shape: {labels.shape}")
    
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"Number of samples in {data_name} images and labels should match. "
                         f"Images samples: {images.shape[0]}, Labels samples: {labels.shape[0]}")
    
    print(f"{data_name} data validation successful. Shape: {images.shape}, Labels: {labels.shape}")
def main():

    train_images,train_labels,test_images,test_labels  = get_processed_data()

    validate_svm_data(train_images, train_labels, "Training")
    validate_svm_data(test_images, test_labels, "Tesiting")
    
    svm = train_svm( train_images,train_labels)
    test_svm(test_images,test_labels,svm)

if __name__ == "__main__":
    main()