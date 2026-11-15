# AIGC-5500 Lab 7 - Latent Spaces

**Student:** Daniel Alejandro Castillo Martin  
**Student ID:** N10019284  
**Course:** Advanced Deep Learning (AIGC-5500-0NA)  
**Professor:** Hossein Pourmodheji

---

Lab 07 submission for **Advanced Deep Learning (AIGC-5500)** in the Artificial Intelligence with Machine Learning postgraduate program at Humber Polytechnic.

This project implements and verifies a variational autoencoder (VAE) in PyTorch using 8 × 8 digit images. It compares a plain autoencoder and a VAE on reconstruction and prior sampling, evaluates how the KL weight affects latent activity, and examines interpolation, class conditioning, and generative metrics.

## 1. Setup

**Requirements:** Python 3.14.6, PyTorch, NumPy, SciPy, scikit-learn, Matplotlib, and a Jupyter-compatible notebook environment.

Clone the repository:

```powershell
git clone https://github.com/dancmartin11/aigc-5500-lab-07-latent-spaces.git
cd aigc-5500-lab-07-latent-spaces
```

Create and activate a virtual environment in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For Linux/macOS, create and activate the environment using:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Project Structure

```text
aigc-5500-lab-07-latent-spaces/
├── scripts/
│   └── lab07_starter.py
├── src/
│   ├── __init__.py
│   └── vae.py
├── daniel_alejandro_castillo_martin_LAB7.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

The `src/` package contains the VAE architecture and loss. The notebook documents the methodology and runs the verification checks and experiments. The original `lab07_starter.py` is preserved without modification.

## 3. Run

From the repository root, run the starter check:

```powershell
python scripts/lab07_starter.py
```

Open `daniel_alejandro_castillo_martin_LAB7.ipynb` in VS Code or Jupyter, select the `.venv` kernel, and execute the sections in order.

## 4. Hardware and Reproducibility

The notebook uses Python 3.14.6 on Windows. The experiments are designed to run on CPU.

**Experimental settings:**

| Parameter | Value |
|---|---|
| Dataset | scikit-learn Digits |
| Training images | 1,438 |
| Test images | 359 |
| Data split seed | 0 |
| Training seeds | 0, 1, 2 |
| Hidden width | 64 |
| Latent dimensions | 8 |
| Epochs | 200 |
| Batch size | 64 |
| Learning rate | 2e-3 |
| Optimizer | Adam |
| KL weights | 0.1, 0.5, 1, 2, 4 |

The lab specifies the training budget and learning rate. The autoencoder comparison and KL-weight sweep use the same data split, hidden width, latent size, epochs, batch size, learning rate, and training seeds. The VAE reconstruction term is summed over the 64 pixels, the KL term is summed over the 8 latent dimensions, and the weighted loss is averaged over the batch.

## 5. Prediction (Written Before the KL-Weight Sweep)

### Hypothesis 1 — Reconstruction and Posterior Collapse

I expect the KL divergence to decrease and reconstruction to worsen as the KL weight increases. At `beta = 4`, I expect the KL divergence to approach zero and few or no latent dimensions to remain active. Together, these findings would indicate posterior collapse because the encoder would produce distributions close to the prior across inputs.

### Hypothesis 2 — Quality of Prior Samples

I expect `beta = 0.5` to give the lowest mean Fréchet distance across the three seeds. At `beta = 0.1`, the weak KL penalty may allow the encoder to use regions that are rarely sampled from the prior. I expect `beta = 0.5` to improve alignment with the prior while retaining enough information to reconstruct the digits.

## 6. Results

### 6.1 VAE Implementation and Verification

| Verification | Result |
|---|---|
| Closed-form KL vs. Monte Carlo | All five estimates were within four Monte Carlo standard errors. The largest absolute difference was 0.0247. |
| Reconstruction gradient at the encoder mean head | Gradient norm of 0.689904 with reparameterization and 0.000000 after detaching the latent sample. |
| VAE loss vs. negative ELBO at `beta = 1` | Absolute difference of 3.81 × 10⁻⁶ with the same latent draws. |