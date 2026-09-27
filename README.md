# DL Plant Disease CNN

**An image-based plant leaf disease classification application built with PyTorch and Streamlit.**

DL Plant Disease CNN, presented in the application as **The Leaf Ledger**, allows users to upload a leaf image and view a predicted crop condition, confidence score, alternative predictions, and care notes.

> **Disclaimer:** This is an educational classification tool, not a substitute for advice from an agricultural expert. Predictions may be less reliable on field photographs or images outside the model's supported classes.

## Project Features

- Upload leaf images in JPG, JPEG, or PNG format.
- Classify images across 38 supported plant-condition classes.
- Display the leading prediction and confidence score.
- Show alternative candidate predictions.
- Present crop-condition information and care notes.
- Browse the species index and per-class evaluation information.
- Run the application through a Streamlit web interface.

## Model Overview

The project uses a custom **Convolutional Neural Network (CNN)** implemented with PyTorch.

| Setting | Value |
|---|---|
| Framework | PyTorch / Torchvision |
| Architecture | Custom CNN |
| Input image size | 128 × 128 pixels |
| Input channels | RGB (3) |
| Number of classes | 38 |
| Normalization mean | `[0.5, 0.5, 0.5]` |
| Normalization standard deviation | `[0.5, 0.5, 0.5]` |
| Training augmentation | Horizontal flips and rotations |
| Saved model weights | `plant_disease_cnn.pth` |

The model architecture, class-name mapping, and helper functions are defined in `model.py`. The Streamlit interface and prediction workflow are in `app.py`.

## Training and Evaluation Results

The training notebook reports that training and evaluation completed, with the following dataset sizes and results:

| Metric | Result |
|---|---:|
| Number of classes | 38 |
| Training images | 56,236 |
| Validation images | 14,059 |
| Test images | 17,572 |
| Final training accuracy | 88.26% |
| Final validation accuracy | 94.66% |
| Test accuracy | 94.53% |

Images were resized to **128 × 128**. The training pipeline used **horizontal flips and rotations** for augmentation, and normalized each RGB channel using a mean and standard deviation of **0.5**.

The reported test accuracy is **94.53%** under the notebook's evaluation setup. This figure does not guarantee the same performance on every user-uploaded image or on photographs captured in real-world field conditions.

## Project Structure

```text
DL_plant_disease_cnn/
├── app.py                    # Streamlit application
├── model.py                  # CNN architecture, classes, and helpers
├── requirements.txt          # Python dependencies
├── plant_disease_cnn.pth     # Trained weights, if included
├── README.md
└── .gitignore
```

Keep `.venv/`, `venv/`, and `__pycache__/` out of the Git repository.

## Requirements

- Python
- pip
- Git (for cloning the repository)
- Trained model weights (`plant_disease_cnn.pth`) for real predictions

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/TayibaAbdulRasool/DL_plant_disease_cnn.git
cd DL_plant_disease_cnn
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the trained model weights

Place `plant_disease_cnn.pth` in the project folder, alongside `app.py` and `model.py`.

If the weights are not included in this repository, obtain them from the training notebook and place them in the project folder. Without the weights, the application may display its interface in demo mode; demo predictions are simulated and must not be treated as actual model outputs.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal. Upload a clear leaf image and select **Diagnose specimen**.

## How It Works

1. The user uploads a leaf photograph.
2. The application converts it to RGB, resizes it to 128 × 128, and applies the model's normalization.
3. The CNN produces scores for the supported classes.
4. The application converts the scores to probabilities and displays the top prediction and candidate classes.
5. The interface shows related crop-condition information and care notes.

## Supported Classes

The supported class labels are defined by `CLASS_NAMES` in `model.py`. The model selects among these known classes; it does not independently confirm that an uploaded image is a leaf or that the plant belongs to a supported crop.

## Limitations

- **Closed-set classification:** Unsupported plants and unrelated images may still be assigned a known class.
- **Confidence is not certainty:** A high probability score does not prove that the prediction is correct.
- **Image quality matters:** Lighting, blur, background, camera quality, leaf angle, and disease stage can affect results.
- **Dataset-to-field gap:** Performance on held-out data may differ from performance on outdoor or farm photographs.
- **Expert confirmation:** Consult a qualified agricultural specialist before making important crop-treatment decisions.

For clearer inputs, use a sharp, well-lit image with one visible leaf and minimal obstruction.

## Technologies

- Python
- PyTorch
- Torchvision
- Streamlit
- Pillow
- NumPy

## Author

**Tayiba Abdul Rasool**


