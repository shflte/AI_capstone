import torch
from torchvision import transforms
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from PIL import Image
import os

def load_dataset(root, label_file):
    images = [f"{i}.png" for i in range(1, len(open(os.path.join(root, label_file), 'r').readlines()) + 1)]
    labels = []
    with open(os.path.join(root, label_file), 'r') as f:
        for line in f:
            label = int(line.strip())
            labels.append(label)
    return images, labels

transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

def main():
    TRAIN_PATH = '.'
    LABEL_FILE = 'label_svm.txt'
    images, labels = load_dataset(TRAIN_PATH, LABEL_FILE)

    X = []
    for img_file in images:
        img_path = os.path.join(TRAIN_PATH, 'captcha_img', img_file)
        image = Image.open(img_path)
        image = transform(image)
        X.append(image.numpy().flatten())
    X = np.array(X)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = svm.SVC(gamma='auto')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
