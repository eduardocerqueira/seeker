#date: 2024-08-16T16:46:23Z
#url: https://api.github.com/gists/996d18cecf3ca2076d1cbd640f02420a
#owner: https://api.github.com/users/AkshathRaghav

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import seaborn as sns
from scipy.spatial import ConvexHull
import umap
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mpl_toolkits.mplot3d import Axes3D
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

def generate_ind_data_3d(num_samples=1000):
    mean = [0, 0, 0]
    cov = np.eye(3)  
    ind_data = np.random.multivariate_normal(mean, cov, num_samples)
    return ind_data

def generate_ood_data_3d(num_samples=1000):
    ood_data = np.random.uniform(low=-5, high=5, size=(num_samples, 3))
    return ood_data

ind_data_3d = generate_ind_data_3d()
ood_data_3d = generate_ood_data_3d()

def plot_convex_hull_2d(points, ax, color='b', alpha=0.1):
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], color)
    ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], color)
    ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=alpha)

fig = plt.figure(figsize=(16, 8))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(ind_data_3d[:, 0], ind_data_3d[:, 1], ind_data_3d[:, 2], label='IND', alpha=0.5, c='b')
ax1.scatter(ood_data_3d[:, 0], ood_data_3d[:, 1], ood_data_3d[:, 2], label='OOD', alpha=0.5, c='r')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

ind_data_2d = ind_data_3d[:, :2]
ood_data_2d = ood_data_3d[:, :2]

ax2 = fig.add_subplot(122)
ax2.scatter(ind_data_2d[:, 0], ind_data_2d[:, 1], label='IND', alpha=0.5, c='b')
ax2.scatter(ood_data_2d[:, 0], ood_data_2d[:, 1], label='OOD', alpha=0.5, c='r')
plot_convex_hull_2d(ind_data_2d, ax2, color='b', alpha=0.2)
plot_convex_hull_2d(ood_data_2d, ax2, color='r', alpha=0.2)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.legend()

plt.tight_layout()
plt.savefig('./assets/img/tldr/etran/initial.png')

def extract_features(data, n_components=3):
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(data)
    return features

ind_features_3d = extract_features(ind_data_3d)
ood_features_3d = extract_features(ood_data_3d)

def calculate_energy(features):
    energy = np.sum(features**2, axis=1)
    return energy

ind_energy_3d = calculate_energy(ind_features_3d)
ood_energy_3d = calculate_energy(ood_features_3d)

def normalize_energy(energy):
    return (energy - np.min(energy)) / (np.max(energy) - np.min(energy))

ind_energy_normalized = normalize_energy(ind_energy_3d)
ood_energy_normalized = normalize_energy(ood_energy_3d)

def calculate_probability_density(energy):
    exp_neg_energy = np.exp(-energy)
    partition_function = np.sum(exp_neg_energy)
    probability_density = exp_neg_energy / partition_function
    return probability_density

ind_probabilities = calculate_probability_density(ind_energy_normalized)
ood_probabilities = calculate_probability_density(ood_energy_normalized)


def apply_umap(features, n_neighbors=15, min_dist=0.1, n_components=3):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    embedding = reducer.fit_transform(features)
    return embedding

ind_umap_3d = apply_umap(ind_features_3d)
ood_umap_3d = apply_umap(ood_features_3d)

ind_umap_2d = ind_umap_3d[:, :2]
ood_umap_2d = ood_umap_3d[:, :2]

fig = plt.figure(figsize=(16, 8))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(ind_umap_3d[:, 0], ind_umap_3d[:, 1], ind_umap_3d[:, 2], label='IND', alpha=0.5, c='b')
ax1.scatter(ood_umap_3d[:, 0], ood_umap_3d[:, 1], ood_umap_3d[:, 2], label='OOD', alpha=0.5, c='r')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

ax2 = fig.add_subplot(122)
ax2.scatter(ind_umap_2d[:, 0], ind_umap_2d[:, 1], label='IND', alpha=0.5, c='b')
ax2.scatter(ood_umap_2d[:, 0], ood_umap_2d[:, 1], label='OOD', alpha=0.5, c='r')
plot_convex_hull_2d(ind_umap_2d, ax2, color='b', alpha=0.2)
plot_convex_hull_2d(ood_umap_2d, ax2, color='r', alpha=0.2)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.legend()

plt.tight_layout()
plt.savefig('./assets/img/tldr/etran/post.png')


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.kdeplot(ind_probabilities, label='IND', fill=True, alpha=0.5, ax=axes[0])
sns.kdeplot(ood_probabilities, label='OOD', fill=True, alpha=0.5, ax=axes[0])
axes[0].set_xlabel(r'Probability Density', fontsize=12)
axes[0].set_ylabel(r'Density', fontsize=12)
axes[0].set_title('KDE Plot of Probability Densities', fontsize=14)
axes[0].legend()
axes[0].grid(True)

sns.kdeplot(ind_energy_normalized, label='IND', fill=True, alpha=0.5, ax=axes[1])
sns.kdeplot(ood_energy_normalized, label='OOD', fill=True, alpha=0.5, ax=axes[1])
axes[1].set_xlabel(r'Energy Score', fontsize=12)
axes[1].set_ylabel(r'Density', fontsize=12)
axes[1].set_title('KDE Plot of Energy Scores', fontsize=14)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('./assets/img/tldr/etran/kde.png')

### 


dim = 3
num_samples = 1000 // 3

gaussian_embeddings = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, dim))
gaussian_labels = np.zeros(num_samples)

uniform_embeddings = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, dim))
uniform_labels = np.ones(num_samples)

exponential_embeddings = np.random.exponential(scale=1.0, size=(num_samples, dim))
exponential_labels = np.full(num_samples, 2)

embeddings = np.vstack((gaussian_embeddings, uniform_embeddings, exponential_embeddings))
labels = np.concatenate((gaussian_labels, uniform_labels, exponential_labels))

embeddings = torch.tensor(embeddings, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(121, projection='3d')
scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=labels, cmap='viridis', alpha=0.5)
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
ax.set_title('Pre-LDA Features')

lda = LDA(n_components=2)
features_lda = lda.fit(embeddings, labels).transform(embeddings)

ax2 = fig.add_subplot(122)
scatter2 = ax2.scatter(features_lda[:, 0], features_lda[:, 1], c=labels, cmap='viridis', alpha=0.5)

legend2 = ax2.legend(*scatter2.legend_elements(), title="Classes")
ax2.add_artist(legend2)
ax2.set_title('Post-LDA Features')

plt.tight_layout()
plt.savefig('./assets/img/tldr/etran/cls_initial.png')

prob = lda.predict_proba(embeddings)
prob_labels = np.argmax(prob, axis=1)

fig, ax3 = plt.subplots(figsize=(10, 6))
for i in range(prob.shape[1]):
    sns.kdeplot(prob[labels == i][:, i], label=f'Class {i}', fill=True, alpha=0.5)
ax3.set_xlabel('Probability')
ax3.set_ylabel('Density')
ax3.set_title('KDE Plot of Class Probabilities')
ax3.legend()
plt.savefig('./assets/img/tldr/etran/cls_kde.png')

### 

width, height = 500, 500
fig, ax = plt.subplots(figsize=(5, 5))

ax.set_facecolor('white')

circle_radius = 50
circle_center = (width - circle_radius - 20, circle_radius + 20)
circle = patches.Circle(circle_center, circle_radius, color='red')
ax.add_patch(circle)
square_size = 100
square_top_left = (20, height - square_size - 20)
square = patches.Rectangle(square_top_left, square_size, square_size, color='blue')
ax.add_patch(square)

triangle = patches.Polygon([(width - 120, height - 20), (width - 20, height - 20), (width - 70, height - 120)], color='green')
ax.add_patch(triangle)

ellipse = patches.Ellipse((100, 100), 150, 80, color='yellow')
ax.add_patch(ellipse)

star = patches.RegularPolygon((width // 2, height // 2), numVertices=5, radius=50, color='purple')
ax.add_patch(star)

ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_aspect('equal', 'box')
ax.axis('off')
plt.savefig('./plot.png')

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image = Image.open('./plot.png')

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

bounding_boxes = torch.tensor([
    [70, 65, 160, 155],  
], dtype=torch.float32)

targets = bounding_boxes
features = image_features
U, S, Vh = torch.linalg.svd(features, full_matrices=False)

energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
rank = torch.searchsorted(energy, 0.8).item()

S = S[:rank]
U = U[:, :rank]
Vh = Vh[:rank, :]

S_inv = torch.diag(1.0 / S)
features_pseudo_inv = Vh.T @ S_inv @ U.T
bboxes_approximated = features @ features_pseudo_inv @ targets.float()

regression_score = -torch.sum((targets - bboxes_approximated) ** 2) * (1/(targets.shape[0] * 4))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(image)

ax[0].add_patch(plt.Rectangle((70, 65), 90, 90, edgecolor='red', facecolor='none', lw=2))
ax[0].set_title('Original Bounding Boxes')
ax[0].axis('off')
projected_bboxes = bboxes_approximated.detach().numpy()
print(projected_bboxes)

ax[1].imshow(image)
ax[1].add_patch(plt.Rectangle((projected_bboxes[0][0], projected_bboxes[0][1]), projected_bboxes[0][2] - projected_bboxes[0][0], projected_bboxes[0][3] - projected_bboxes[0][1], edgecolor='red', facecolor='none', lw=2))
ax[1].set_title('Reconstructed Bounding Boxes')
ax[1].axis('off')
plt.savefig('./assets/img/tldr/etran/bounding_fail.png')