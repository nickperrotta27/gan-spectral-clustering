# GAN & Spectral Clustering Project

This project implements two machine learning experiments:
1. **Generative Adversarial Network (GAN)** for 2D distribution learning
2. **Spectral Clustering** vs traditional K-Means comparison

---

## 📊 Datasets

### Dataset 1: `gan_target_distribution.pkl`
- **Purpose**: Target distribution for GAN training
- **Format**: 2D point cloud (N × 2)
- **Usage**: Training data for the discriminator to learn the real distribution

### Dataset 2: `spectral_clustering_data.pkl`
- **Purpose**: 2D clustering dataset
- **Format**: 2D point cloud (N × 2)
- **Usage**: Comparing spectral clustering with traditional K-Means

---

## 🚀 Features

### Part 1: Generative Adversarial Network (GAN)

#### Architecture
- **Generator**:
  - Input: 2D noise vector (Gaussian)
  - Architecture: `2 → 100 → 100 → 2`
  - Activation: LeakyReLU (α=0.2)
  
- **Discriminator**:
  - Input: 2D point (real or generated)
  - Architecture: `2 → 100 → 100 → 1`
  - Activation: LeakyReLU (α=0.2) + Sigmoid output

#### Training Details
- **Epochs**: 5000
- **Batch Size**: 128
- **Optimizer**: Adam (lr=1e-4)
- **Loss**: Binary Cross-Entropy
- **Training Ratio**: 10 discriminator steps : 1 generator step

#### Visualizations
1. **Discriminator Heatmap**: Shows discriminator output over a 2D grid (0-8 range)
2. **Input vs Output**: Comparison of noise distribution and generated samples

---

### Part 2: Spectral Clustering

#### Algorithm Pipeline
1. **Affinity Matrix Construction**:
   - RBF kernel: `W_ij = exp(-||x_i - x_j||² / 2σ²)`
   - σ = median of pairwise distances (automatic tuning)

2. **Normalized Laplacian**:
   - `L = I - D^(-1/2) W D^(-1/2)`
   - Where D is the degree matrix

3. **Eigendecomposition**:
   - Compute eigenvectors of L
   - Use eigenvectors 1-3 (k=3 clusters)
   - Row-normalize the embedding

4. **K-Means on Embedding**:
   - Cluster the normalized eigenvectors
   - 20 random initializations for stability

#### Comparison
- **Spectral Clustering**: Handles non-convex clusters
- **Traditional K-Means**: Works best on spherical clusters
- **Diagnostic Output**: Label counts and difference verification

#### Visualizations
1. **Spectral Clustering Results**: Color-coded clusters using Set2 colormap
2. **K-Means Results**: Color-coded clusters using Set1 colormap

---

## 📁 Project Structure

```
.
├── gan_target_distribution.pkl        # GAN training data
├── spectral_clustering_data.pkl       # Clustering dataset
├── notebook.ipynb                     # Complete implementation
└── README.md                          # This file
```

---

## 📦 Dependencies

### Core Libraries
```bash
pip install torch torchvision matplotlib numpy scikit-learn
```

### Detailed Requirements
- `torch` - Neural network framework
- `matplotlib` - Visualization
- `numpy` - Numerical operations
- `scikit-learn` - K-Means & distance metrics
- `pickle` - Data loading

---

## 🧠 How to Run

### 1. Open in Google Colab
Upload the notebook to Colab for GPU acceleration.

### 2. Upload Datasets
First cell will prompt for file upload:
```python
from google.colab import files
uploaded = files.upload()
```
Upload both:
- `gan_target_distribution.pkl`
- `spectral_clustering_data.pkl`

### 3. Run All Cells
Execute sequentially to:
- Train the GAN (takes ~5-10 minutes on GPU)
- Visualize discriminator output
- Compare generated vs real distributions
- Perform spectral clustering
- Compare with K-Means

---

## 📈 Expected Results

### GAN Training
- **D_loss**: Should stabilize around 0.5-1.5
- **G_loss**: Should decrease and stabilize around 0.5-2.0
- **Generated samples**: Should match the target distribution shape

### Clustering Comparison
- **Spectral Clustering**: Better at capturing complex, non-linear cluster boundaries
- **K-Means**: May struggle with non-convex or overlapping clusters
- **Label counts**: Different cluster sizes indicate different segmentation strategies

---

## 🎯 Key Insights

### GAN Learning
- **10:1 training ratio** stabilizes GAN training
- **Discriminator heatmap** shows learned decision boundaries
- **Generator** learns to map Gaussian noise to target distribution

### Spectral Clustering Advantages
- Uses manifold structure of the data
- Captures non-linear relationships via RBF kernel
- Automatic σ tuning using median heuristic
- Better for complex cluster shapes

---

## 🔬 Theoretical Background

### GAN Objective
```
min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

### Spectral Clustering Steps
1. Graph construction (RBF similarity)
2. Normalized cut via eigenvectors
3. K-Means in reduced spectral space

---

## 📌 Hyperparameters

### GAN
| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| Batch Size | 128 |
| Epochs | 5000 |
| Hidden Dim | 100 |
| LeakyReLU α | 0.2 |

### Spectral Clustering
| Parameter | Value |
|-----------|-------|
| k (clusters) | 3 |
| σ | Auto (median) |
| K-Means init | 20 |
| Eigenvectors used | 1-3 |

---

## 🗺️ Purpose of This Repository

This project demonstrates:
- ✅ GAN training and stabilization techniques
- ✅ Discriminator visualization for understanding learned representations
- ✅ Spectral clustering implementation from scratch
- ✅ Comparison of linear vs non-linear clustering methods
- ✅ Automatic hyperparameter tuning (σ selection)
- ✅ Normalized Laplacian eigendecomposition

---

## 🐛 Troubleshooting

### GAN Issues
- **Mode collapse**: Increase discriminator training steps or adjust learning rates
- **Training instability**: Reduce learning rate or add gradient penalty
- **Poor generation**: Train for more epochs

### Clustering Issues
- **Poor separation**: Adjust σ manually or try different k values
- **Numerical instability**: Check for zero/negative eigenvalues

---

## 📄 License

This project is open source and available for educational purposes.

---

## 🤝 Contributing

Feel free to:
- Add more GAN architectures (WGAN, DCGAN)
- Implement different clustering algorithms
- Improve visualizations
- Add evaluation metrics

---

## 📚 References

- Goodfellow et al. (2014) - Generative Adversarial Networks
- Ng et al. (2001) - On Spectral Clustering
- von Luxburg (2007) - A Tutorial on Spectral Clustering

---

## 📧 Contact

For questions or suggestions, please open an issue in this repository.
