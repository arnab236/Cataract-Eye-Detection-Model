import os
import cv2
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import classification_report

from feature_extraction import extract_features


dataset_path = "./dataset"

classes = ["normal","cataract"]

X = []
y = []

# ---------- LOAD DATASET ----------

for label in classes:

    folder = os.path.join(dataset_path,label)

    for file in os.listdir(folder):

        if not file.lower().endswith((".jpg",".jpeg",".png",".bmp")):
            continue

        path = os.path.join(folder,file)

        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(img,(224,224))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = extract_features(gray)

        X.append(features)

        y.append(label)


X = np.array(X)

print("Total samples:", len(X))


# ---------- SPLIT ----------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------- SVM ----------

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True))
])

# ---------- RANDOM FOREST ----------

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

# ---------- EXTRA TREES ----------

et_model = ExtraTreesClassifier(
    n_estimators=300,
    random_state=42
)

# ---------- ENSEMBLE ----------

ensemble_model = VotingClassifier(
    estimators=[
        ("svm", svm_model),
        ("rf", rf_model),
        ("et", et_model)
    ],
    voting="soft"
)

print("Training ensemble model...")

ensemble_model.fit(X_train,y_train)

# ---------- EVALUATION ----------

pred = ensemble_model.predict(X_test)

print("\nDetection Accuracy:", ensemble_model.score(X_test,y_test))
print("\nClassification Report:\n")
print(classification_report(y_test,pred))


# ---------- SAVE MODEL ----------

os.makedirs("../models",exist_ok=True)

joblib.dump(ensemble_model,"../models/cataract_detection.pkl")

print("Detection model saved!")