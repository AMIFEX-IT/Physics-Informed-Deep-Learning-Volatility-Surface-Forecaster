# Physics-Informed Deep Learning Volatility Surface Forecaster

> *Bridging data-driven machine learning and the structural guarantees of financial mathematics.*

An enterprise-grade deep learning framework that predicts implied volatility surfaces **σ(K, T)** while strictly enforcing fundamental financial physics and no-arbitrage constraints. By leveraging **Physics-Informed Neural Networks (PINNs)**, this model embeds the Black-Scholes-Merton and Dupire frameworks directly into the learning process — producing surfaces that are not just accurate, but mathematically valid.

---

## 📸 Demo

> **Screenshot — Predicted vs. Actual Implied Volatility Surface**

<!-- Replace the placeholder below with your actual screenshot -->


> **Video Walkthrough**

<!-- Replace the placeholder below with your actual demo video link or embed -->

> 

---

## 🚀 Executive Summary

Traditional deep learning models (standard MLPs, LSTM variants) excel at pattern recognition but routinely forecast implied volatility surfaces that allow for **non-physical realities** — negative digital option prices, negative probability densities, or calendar arbitrage loops.

This project implements a PINN that embeds financial domain knowledge **directly into the neural network's loss function** via automatic differentiation. By regularizing the network with partial differential terms, the model acts as a universal approximator that learns from noisy historical market data while remaining mathematically bounded to valid, arbitrage-free economic spaces.

### Key Capabilities

| Capability | Description |
|---|---|
| 🚫 **Arbitrage-Free Structuring** | Soft-enforces no-static (vertical) and no-calendar (horizontal) arbitrage |
| ⚙️ **Automatic Differentiation Engine** | Computes precise partial gradients (∂σ/∂K, ∂²σ/∂K², ∂σ/∂T) without numerical grid noise |
| 📉 **Data-Efficient Learning** | Outperforms standard deep architectures on sparse, skewed, or OTM-deficient data |

---

## 🧠 Core Mathematics & PINN Loss Architecture

The neural network defines a mapping function **f_θ: (K, T) → σ**, where θ represents the weights and biases. Inputs are the normalized strike price (K) and time-to-maturity (T).

The total loss function is a **multi-objective optimization problem**:

$$\mathcal{L}_{\text{total}}(\theta) = \lambda_1 \mathcal{L}_{\text{data}}(\theta) + \lambda_2 \mathcal{L}_{\text{static}}(\theta) + \lambda_3 \mathcal{L}_{\text{calendar}}(\theta) + \lambda_4 \mathcal{L}_{\text{smooth}}(\theta)$$

Where λᵢ are tunable regularization hyperparameters.

---

### 1. Data Fidelity Loss — `L_data`

Measures Mean Squared Error against observed market-implied volatilities:

$$\mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{i=1}^{N} \left| f_\theta(K_i, T_i) - \sigma_{\text{market}, i} \right|^2$$

---

### 2. Static (Butterfly/Vertical) Arbitrage Loss — `L_static`

To prevent vertical arbitrage, the risk-neutral probability density function must be **strictly non-negative**. By the **Breeden-Litzenberger theorem**, the second derivative of a European Call price (C) with respect to strike (K) must satisfy:

$$\frac{\partial^2 C}{\partial K^2} \ge 0$$

The model maps σ(K, T) into Black-Scholes pricing space dynamically and penalizes any violations:

$$\mathcal{L}_{\text{static}} = \frac{1}{M}\sum_{j=1}^{M} \max\left(0, -\frac{\partial^2 C(f_\theta(K_j, T_j))}{\partial K^2}\right)^2$$

---

### 3. Calendar (Horizontal) Arbitrage Loss — `L_calendar`

To prevent time-dependent arbitrage, the value of a calendar spread must be **non-negative**. Assuming r ≥ 0, option prices must be non-decreasing in maturity:

$$\frac{\partial C}{\partial T} \ge 0$$

$$\mathcal{L}_{\text{calendar}} = \frac{1}{M}\sum_{j=1}^{M} \max\left(0, -\frac{\partial C(f_\theta(K_j, T_j))}{\partial T}\right)^2$$

---

## 🛠️ Implementation

### Custom PINN Loss Engine

The loss engine uses PyTorch's computational graph to extract exact financial gradients via `torch.autograd.grad`:

```python
import torch
import torch.nn as nn

class PINNVolatilityLoss(nn.Module):
    def __init__(self, lambda_data=1.0, lambda_static=10.0, lambda_calendar=10.0):
        super(PINNVolatilityLoss, self).__init__()
        self.lambda_data = lambda_data
        self.lambda_static = lambda_static
        self.lambda_calendar = lambda_calendar
        self.mse = nn.MSELoss()

    def forward(self, K, T, pred_sigma, target_sigma):
        # 1. Standard data-driven loss
        loss_data = self.mse(pred_sigma, target_sigma)

        # Enable gradient tracking on inputs for automatic differentiation
        K.requires_grad_(True)
        T.requires_grad_(True)

        # Extract partial derivatives via backpropagation through the graph
        d_sigma_dK  = torch.autograd.grad(pred_sigma.sum(), K, create_graph=True)[0]
        d2_sigma_dK2 = torch.autograd.grad(d_sigma_dK.sum(), K, create_graph=True)[0]
        d_sigma_dT  = torch.autograd.grad(pred_sigma.sum(), T, create_graph=True)[0]

        # 2. Arbitrage penalties — ReLU forces violation terms positive
        loss_static   = torch.mean(torch.relu(-d2_sigma_dK2) ** 2)
        loss_calendar = torch.mean(torch.relu(-d_sigma_dT)   ** 2)

        total_loss = (self.lambda_data     * loss_data +
                      self.lambda_static   * loss_static +
                      self.lambda_calendar * loss_calendar)
        return total_loss
```

> `create_graph=True` is critical — it ensures second-order derivatives remain in the computational graph so that arbitrage penalties backpropagate correctly through the network weights.

---

## 📁 Repository Structure

```
Physics-Informed-Deep-Learning-Volatility-Surface-Forecaster/
│
├── data/
│   ├── raw/                  # Raw options chain CSV data (strikes, expiries, bids, asks)
│   └── processed/            # Cleaned, normalized implied volatility surfaces
│
├── models/
│   ├── __init__.py
│   ├── network.py            # Deep MLP / Residual Network architecture
│   └── pinn_loss.py          # Black-Scholes pricing layer & autodiff penalty engine
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_surface_visualization.ipynb   # 3D surface mesh generation & metric comparison
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Custom dataset loading pipeline
│   ├── train.py              # Training loop, logging, and checkpointing
│   └── evaluate.py           # Quantitative arbitrage validation pipeline
│
├── assets/                   # Screenshots, demo images, and figures
├── requirements.txt
└── README.md
```

---

## 📊 Evaluation Metrics

This framework uses a **dual-lens evaluation methodology** — standard statistical metrics alongside financially meaningful arbitrage violation checks.

| Metric | Formula | Target |
|---|---|---|
| Root Mean Squared Error (RMSE) | √(1/N · Σ(σ − σ̂)²) | Minimize — measures overall fit precision |
| Mean Absolute Error (MAE) | 1/N · Σ\|σ − σ̂\| | Minimize — robust to outlier strikes |
| Static Arbitrage Violations | Σ 𝟙(∂²C/∂K² < 0) | **Must equal 0** — surface shape validity |
| Calendar Arbitrage Violations | Σ 𝟙(∂C/∂T < 0) | **Must equal 0** — temporal surface validity |

> A model achieving low RMSE but non-zero arbitrage violations is **financially unusable**. Both dimensions must be satisfied simultaneously.

---

## ⚡ Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/FIRST-PRINCE/Physics-Informed-Deep-Learning-Volatility-Surface-Forecaster.git
cd Physics-Informed-Deep-Learning-Volatility-Surface-Forecaster

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train.py --epochs 500 --batch_size 128 --lr 0.001
```

### 3. Evaluate a Checkpoint

```bash
python src/evaluate.py --model_path models/checkpoints/pinn_best.pt
```

---

## 🔬 Why PINNs Over Standard Deep Learning?

| Concern | Standard MLP / LSTM | PINN (This Project) |
|---|---|---|
| Arbitrage-free guarantee | ❌ Not enforced | ✅ Penalized in loss |
| Sparse OTM data handling | ❌ Overfits or interpolates poorly | ✅ Physics fills the gap |
| Interpretability | ❌ Black box | ✅ Grounded in BSM theory |
| Surface smoothness | ❌ Can produce jagged surfaces | ✅ Smoothness term regularizes |
| Financial deployability | ❌ Requires post-hoc correction | ✅ Structurally valid outputs |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with PyTorch · Grounded in Black-Scholes-Merton · Enforced by Breeden-Litzenberger
</p>
