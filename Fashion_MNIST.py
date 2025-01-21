
import kagglehub
import os
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

"""DOWNLOAD Fashion-MNIST DATASET OF ZALANDO'S ARTICLE IMAGES FROM KAGGLE"""
# Download latest version
path = kagglehub.dataset_download("zalando-research/fashionmnist")
print("Path to dataset files:", path)

# show the files in the path
for file in os.listdir(path):
    print(file)

# Load images and labels
train_images_path = f"{path}/train-images-idx3-ubyte"
train_labels_path = f"{path}/train-labels-idx1-ubyte"
test_images_path = f"{path}/t10k-images-idx3-ubyte"
test_labels_path = f"{path}/t10k-labels-idx1-ubyte"

# Convert them into numpy
X_train = idx2numpy.convert_from_file(train_images_path)
y_train = idx2numpy.convert_from_file(train_labels_path)
X_test = idx2numpy.convert_from_file(test_images_path)
y_test = idx2numpy.convert_from_file(test_labels_path)

# Verify dimensions
print("Dimensions X_train:", X_train.shape)  # should be (60000, 28, 28)
print("Dimensions y_train:", y_train.shape)  # should be (60000,)
print("Dimensions X_test:", X_test.shape)    # should be (10000, 28, 28)
print("Dimensions y_test:", y_test.shape)    # should be (10000,)

"""EXPLORE THE DATA"""
# Initial Inspection of data
# verify unique labels
print("Unique classes in y_train:", np.unique(y_train))  # should be from 0 to 9
print("Unique classes in y_test:", np.unique(y_test))

"""
The labels are:
    0: T-shirt/top
    1: Trouser
    2: Pullover
    3: Dress
    4: Coat
    5: Sandal
    6: Shirt
    7: Sneaker
    8: Bag
    9: Ankle boot
"""

# check if images are correctly linked with labels
plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Etiqueta: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# verify value of the pixels
print("Minumun value of pixel:", X_train.min())  # should be 0
print("Maximun value of pixel:", X_train.max())  # should be 255

"""NORMALIZATION"""

# scale the pixels from 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0
print("New minumun value mínimo of pixel:", X_train.min())  # should be 0.0
print("New minumun value mínimo of pixel:", X_train.max())  # should be 1.0

# Flatten images
X_train = X_train.reshape(X_train.shape[0], -1)  # From (60000, 28, 28) to (60000, 784)
X_test = X_test.reshape(X_test.shape[0], -1)    # From (10000, 28, 28) to (10000, 784)
print("New dimensions of X_train:", X_train.shape)
print("New dimensions of X_test:", X_test.shape)

"""DIVISION OF DATASET"""
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print("Dimensions of X_train:", X_train.shape)
print("Dimensions of X_val:", X_val.shape)
print("Dimensions of y_train:", y_train.shape)
print("Dimensions of y_val:", y_val.shape)
print("Dimensions of X_test:", X_test.shape)
print("Dimensions of y_test:", y_test.shape)

"""SELECT A MODEL"""
"""Logistic Regression Model"""
# create a model with initial parameters
model = LogisticRegression(max_iter=1000, solver='lbfgs')
# max_iter in 1000 iteration forr the moment
# solver in Limited-memory Broyden-Fletcher-Goldfarb-Shanno for the moment, eficient with multi-characteristic problems

"""TRAIN MODEL"""
model.fit(X_train, y_train)
print("Training completed.")

"""EVALUATE MODEL"""
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Accuracy in validation: {val_accuracy}")

print("Report of classification in validation:")
print(classification_report(y_val, y_val_pred))

conf_matrix = confusion_matrix(y_val, y_val_pred)
# Visualize it
plt.figure(figsize=(8, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='PuRd', xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Prediction")
plt.ylabel("Real Class")
plt.title("Confusion Matrix")
plt.show()

"""ADJUST MODEL"""
# Cross Validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print("Accuracy in each partition:", cv_scores)
print("Average accuracy:", cv_scores.mean())

# GridSearchCV
# Combinations of hiperparameters
param_grid = {
    'C': [0.01, 0.1, 1, 10],
}
# Set up GridSearchCV
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
# Adjust the model
grid_search.fit(X_train, y_train)
# Results
print("Best hiperparameters:", grid_search.best_params_)
print("Best average accuracy:", grid_search.best_score_)

# Best model found by GridSearchCV
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)
print("Accuracy in validation of the best model:", accuracy_score(y_val, y_val_pred))

disp = ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred, normalize="true", values_format=".0%", cmap="PuRd")
# Adjust size
fig = disp.figure_
fig.set_figwidth(20)
fig.set_figheight(4)
plt.title("Normalized Confusion Matrix")
plt.show()

# Identify wrong classified examples
class_a, class_b = 4, 6
misclassified_a_b = X_val[(y_val == class_a) & (y_val_pred == class_b)]
misclassified_b_a = X_val[(y_val == class_b) & (y_val_pred == class_a)]


def plot_images(images, title):
    if len(images) == 0:
        print(f"No hay ejemplos para {title}")
        return
    fig, axes = plt.subplots(1, min(len(images), 5), figsize=(12, 3))
    fig.suptitle(title)
    for img, ax in zip(images[:5], axes):
        ax.imshow(img.reshape(28, 28), cmap='gray')
        ax.axis('off')
    plt.show()

# Visualizar imágenes mal clasificadas
plot_images(misclassified_a_b, f"Clase {class_a} mal clasificada como {class_b}")
plot_images(misclassified_b_a, f"Clase {class_b} mal clasificada como {class_a}")

"""Random Forest Model"""
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)

"""TRAIN MODEL"""
rf_model.fit(X_train, y_train)
print("Training completed.")

"""EVALUATE MODEL"""
y_val_pred_rf = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred_rf)
print(f"Accuracy in validation: {val_accuracy}")

"""FINAL EVALUATION WITH TEST SET"""
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy of test set: {test_accuracy:.4f}")

print("Classification Report on Test Set:")
print(classification_report(y_test, y_test_pred))

disp1 = ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, normalize="true", values_format=".0%", cmap="PuRd")
# adjust size
fig = disp1.figure_
fig.set_figwidth(20)
fig.set_figheight(4)
plt.title("Normalized Confusion Matrix (Test Set)")
plt.show()
