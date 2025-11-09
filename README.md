# 🧠 Adversarial Attacks and Model Robustness on CIFAR-10  
### *Advanced Visualization and Storytelling *  
**Author:** Anchal Gupta  
**Institution:** Vidyashilp University   
**Date:** November 2025  

---

## 📘 Project Overview  

This repository presents a comprehensive exploration of **adversarial robustness** and **internal learning dynamics** of deep CNNs trained on the **CIFAR-10** dataset.  
It consists of two complementary studies:  

1. **Adversarial Attacks and Spectral Normalization:**  
   - Investigates how CNNs misclassify random noise with high confidence.  
   - Introduces **Spectral Normalization** to improve robustness.  

2. **Weight/Bias Evolution and Empirical Spectral Density (ESD) Analysis:**  
   - Tracks internal parameter dynamics (weights/biases) across epochs.  
   - Analyzes **spectral properties** using the WeightWatcher library to interpret model generalization.  

---

## 🎯 Objectives  

### **Part 1 – Adversarial Robustness**
- Build and train a CNN for CIFAR-10 image classification.  
- Conduct brute-force noise-based adversarial attacks.  
- Apply **Spectral Normalization (SN)** for robustness enhancement.  
- Compare performance and adversarial vulnerability before and after SN.  
- Visualize confidence distributions, class bias, and adversarial success rates.  

### **Part 2 – Weight/Bias Evolution and Spectral Analysis**
- Track weight and bias distributions across epochs.  
- Examine **Empirical Spectral Density (ESD)** using WeightWatcher.  
- Detect transitions such as underfitting, generalization, and overfitting.  
- Visualize layer-wise evolution and interpret spectral trends.  

---

## 🧠 Dataset  

**Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)  
- 60,000 color images (32×32×3) in 10 classes.  
- 50,000 training, 10,000 test samples.  
- Classes: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck.  

---

## ⚙️ Methodology  

### 🧩 CNN Architecture  
| Component | Description |
|------------|-------------|
| Convolutions | 8 × Conv2D (3→64→128→256→512) with BatchNorm + ReLU |
| Pooling | MaxPool2d(2×2) after each block |
| Fully Connected Layers | 6 (2048→1024→512→256→10) |
| Dropout | 0.4 → 0.2 (decreasing) |
| Optimizer | SGD (lr=0.01, momentum=0.9) |
| Scheduler | ReduceLROnPlateau (mode='min', patience=5) |
| Loss | CrossEntropyLoss |

---

## 🧮 Part 1- Adversarial Attack – Random Noise Injection  

**Noise Generation:**
'''python'''
noise_image = torch.rand(1, 3, 32, 32)
noise_image = (noise_image.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

**Success Criterion:**
Model confidence ≥ 90% on a random noise input.

**Baseline CNN Results:**
- **Metric	                                  Value**
Total Attempts (T)	                         10,000
Successful Adversarial Examples (N)	          703
Success Rate	                               7.03%

**Most Common Misclassified Classes:**
🐸 Frog (49.8%), 🐱 Cat (44.2%), 🚢 Ship (6.5%)

### 🧩 Spectral Normalization Implementation
Applied to all convolutional layers:
'''python'''
self.conv1 = nn.utils.spectral_norm(nn.Conv2d(3, 64, 3, padding=1))

**Results after Spectral Normalization (SN):**
- **Metric                                   Value**
Total Attempts (T)	                        10,000
Successful Adversarial Examples (N)	         165
Success Rate                              	1.65%

    ✅ SN reduced adversarial vulnerability by ~76%.
    Vulnerability persisted mainly for “deer” and “frog” classes.

### 📊 Results Summary
**Model	      Train Acc  	Val Acc	  Test Acc	 Adversarial Success Rate
CNN	         96.53%	   88.66%     88.34%	    7.03%
Spectral CNN	93.86%	   88.07%     87.96%	    1.65%

### Visualizations
📈 Training & Validation Curves
🧩 Confusion Matrices (CNN vs SN)
💥 Adversarial Examples & Class Distribution
🌀 Gradient Visualizations of Perturbations
📊 Spectral Norm Layerwise Plot

## 🧩 Part 2 – Weight & Bias Evolution + Empirical Spectral Density (ESD)

### 🔍 Motivation
Accuracy alone cannot explain how networks learn.
By tracking weights and spectral distributions across epochs, we gain insight into learning dynamics, regularization, and generalization.

### 🧠 Analysis Tools
- **PyTorch:** model training and checkpointing
- **WeightWatcher:** layer-wise spectral density and power-law exponent analysis
- **Matplotlib/Seaborn:** weight histograms and ESD visualization
- **Manim:** visual animations of learning evolution

### 📊 Observations
**🧩 Weight & Bias Evolution**
Tracked histograms across epochs (1 → 50).
Revealed learning phases:
   Early: wide uniform distributions (underfitting)
   Mid: tighter symmetric shapes (generalization)
   Late: skewed distributions (overfitting)

**⚡ Empirical Spectral Density (ESD)**
Analyzed via WeightWatcher:
Layer	      α (Power-law Exponent) 	Interpretation
Conv1	      4.71	                  Sharp decay → strong regularization, stable
FC Layer 31 2.74	                  Heavy-tailed → mild overfitting or memorization
  Low α (~2–3) → heavy-tailed, overfitting
  High α (~4–5) → well-regularized, smooth generalization

### 🧾 Discussion
- **✅ Spectral Normalization improved robustness** and reduced overconfidence on noise.
- **🔍 Spectral analysis revealed** that different layers contribute differently to generalization.
- **⚠️ Remaining vulnerabilities** show that SN alone isn’t sufficient—class-specific weaknesses persist.
- **🧠 ESD analysis** provides interpretable insights into internal stability and overfitting behavior.

### 🏁 Conclusion
This project demonstrates that:
- **Spectral Normalization** significantly enhances CNN robustness to adversarial noise.
- The adversarial success rate dropped from 7.03% → 1.65%.
- **ESD analysis** exposes internal dynamics and highlights layer-wise stability.
- Combining adversarial evaluation with spectral analysis provides a complete picture of model robustness and generalization.

**Future Work:**
Adversarial training or certified defenses (e.g., randomized smoothing).
Layer-wise SN optimization or hybrid normalization.
Early stopping based on spectral cues.

### 📂 Repository Structure
'''objectivec'''
Adversarial-Robustness-CIFAR10/
│
├── Notebooks/
│   ├── Part1-CIFAR10.ipynb
│   ├── Part2-CIFAR10.ipynb
│
├── Results/
│   ├── Part1_loss_accuracy_plot.png
│   ├── Part1_confusion_matrix.png
│   ├── Part2_cnn_confusion_matrix.png
│   ├── Part2_cnn_spectral_confusion_matrix.png
│   ├── CNN_accuracy.png
│   ├── CNN_loss.png
│   ├── CNN_Spectral_loss.png
│   ├──CNN_Spectral_accuracy.png
│
├── Videos/
│   ├── ESD PLOTS.mp4
│   ├── global_biases.mp4
|   ├── global_weights.mp4
│   ├── Weight_Bias Video.mp4
│
├── requirements.txt
└── README.md

### ⚙️ Environment Setup
**1️⃣ Clone Repository**
'''bash'''
git clone https://github.com/<your-username> Adversarial-Robustness-CIFAR10.git
cd Adversarial-Robustness-CIFAR10

**2️⃣ Install Dependencies**
'''bash'''
pip install -r requirements.txt

**3️⃣ Run Notebooks**
'''bash'''
jupyter notebook Notebooks/CIFAR10_CNN.ipynb

### 👩‍💻 Author
**Name:** Anchal Gupta
**Course:** Advanced Visualization and Storytelling (ADVST)
**Institution:** Vidyashilp University

### 🧾 References
- Miyato et al., Spectral Normalization for Generative Adversarial Networks, ICLR 2018
- Krizhevsky, A., Learning Multiple Layers of Features from Tiny Images (CIFAR-10 Dataset)
- Martin & Mahoney, WeightWatcher: ESD and Heavy-Tailed Behavior in Deep Neural Networks
- PyTorch Documentation – https://pytorch.org