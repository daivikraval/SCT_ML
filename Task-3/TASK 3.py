import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set parameters
IMAGE_SIZE = (64, 64)  
DATASET_DIR = 'train' 

X = []
y = []
categories = ['cat', 'dog']

for label, category in enumerate(categories):
    folder = os.path.join(DATASET_DIR, category)
    for filename in os.listdir(folder)[:1000]:  
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = hog(img_gray, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM
clf = svm.SVC(kernel='rbf', gamma='scale')
clf.fit(X_train, y_train)

# Predict & evaluate
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=categories))
