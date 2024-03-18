import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from PIL import Image
import os
from torchvision import transforms
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from PIL import Image
import os
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

def load_images(root, num_images):
    images = []
    for i in range(1, num_images + 1):
        img_path = os.path.join(root, 'captcha_img', f"{i}.png")
        image = Image.open(img_path)
        image = transform(image)
        images.append(image.numpy().flatten())
    return np.array(images)

def perform_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    
    silhouette_avg = silhouette_score(X, labels)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")
    
    return labels, kmeans.cluster_centers_

def main():
    TRAIN_PATH = '.'
    num_images = 600
    X = load_images(TRAIN_PATH, num_images)

    n_clusters = range(2, 20)
    silhouette_scores = []
    for n in n_clusters:
        labels, centers = perform_kmeans(X, n)
        silhouette_scores.append(silhouette_score(X, labels))

    plt.plot(n_clusters, silhouette_scores)
    plt.savefig('silhouette_scores.png')

if __name__ == '__main__':
    main()
