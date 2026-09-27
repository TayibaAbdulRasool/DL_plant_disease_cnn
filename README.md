# 🌿 DL Plant Disease CNN

**An image-based plant leaf disease classification app built with
PyTorch and Streamlit.**

DL Plant Disease CNN, presented in the app as **The Leaf Ledger**, lets
users upload a leaf image and view a predicted crop condition,
confidence score, alternative predictions, and care notes.

> **Disclaimer:** This is an educational classification tool, not a
> substitute for advice from an agricultural expert. Predictions may be
> less reliable on field photographs or images outside the model's
> supported classes.

## ✨ Features

-   Upload leaf images in JPG, JPEG, or PNG format.
-   Classify images across 38 supported plant-condition classes.
-   Display the leading prediction and confidence score.
-   Show alternative candidate predictions.
-   Present crop-condition information and care notes.
-   Browse the species index and per-class evaluation information.
-   Run through a Streamlit web interface.

## 🧠 Model

The project uses a custom **Convolutional Neural Network (CNN)**
implemented in PyTorch.

  -----------------------------------------------------------------------
  Setting                             Value
  ----------------------------------- -----------------------------------
  Framework                           PyTorch / Torchvision

  Architecture                        Custom CNN

  Input                               RGB image, resized to 128 × 128
                                      pixels

  Output                              38 classes

  Normalization                       Mean `[0.5, 0.5, 0.5]`; standard
                                      deviation `[0.5, 0.5, 0.5]`

  Weights                             `plant_disease_cnn.pth`
  -----------------------------------------------------------------------

The architecture, supported class names, and helper functions are in
`model.py`. The user interface and prediction workflow are in `app.py`.

### Evaluation

The training notebook reports **94.53% test accuracy**. This figure
reflects the notebook's evaluation setup and does not guarantee the same
performance on every user-uploaded image or in real-world field
conditions.

## 🗂️ Project Structure

``` text
DL_plant_disease_cnn/
├── app.py                    # Streamlit application
├── model.py                  # CNN architecture, classes, and helpers
├── requirements.txt          # Python dependencies
├── plant_disease_cnn.pth     # Trained weights, if included
├── README.md
└── .gitignore
```

Keep `.venv/`, `venv/`, and `__pycache__/` out of the Git repository.

## ⚙️ Setup

### 1. Clone the repository

``` bash
git clone https://github.com/TayibaAbdulRasool/DL_plant_disease_cnn.git
cd DL_plant_disease_cnn
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation in the current session, run:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the trained model weights

Place `plant_disease_cnn.pth` in the project folder, alongside `app.py`
and `model.py`.

If the weights are not included in this repository, obtain them from the
training notebook and place them in the project folder. Without the
weights, the app can show its interface in demo mode; demo predictions
are simulated and must not be treated as real model outputs.

### 5. Launch the app

``` bash
streamlit run app.py
```

Open the local URL shown in the terminal. Upload a clear leaf image and
click **Diagnose specimen**.

## 🔍 How It Works

1.  The user uploads a leaf photograph.
2.  The app converts it to RGB, resizes it to 128 × 128, and applies the
    model's normalization.
3.  The CNN produces scores for the supported classes.
4.  The app converts scores to probabilities and displays the top
    prediction and candidate classes.
5.  The interface shows related crop-condition information and care
    notes.

## 🌱 Supported Classes

The supported class labels are defined by `CLASS_NAMES` in `model.py`.
The model selects among these known classes; it does not independently
confirm that an uploaded image is a leaf or that the plant belongs to a
supported crop.

## ⚠️ Limitations

-   **Closed-set classification:** unsupported plants and unrelated
    images may still be assigned a known class.
-   **Confidence is not certainty:** a high probability score does not
    prove the prediction is correct.
-   **Image quality matters:** lighting, blur, background, camera
    quality, leaf angle, and disease stage can affect results.
-   **Dataset-to-field gap:** performance on held-out data may differ
    from performance on outdoor or farm photographs.
-   **Expert confirmation:** consult a qualified agricultural specialist
    before making important crop-treatment decisions.

For clearer inputs, use a sharp, well-lit image with one visible leaf
and minimal obstruction.

## 🛠️ Technologies

-   Python
-   PyTorch
-   Torchvision
-   Streamlit
-   Pillow
-   NumPy

## 👩‍💻 Author

**Tayiba Abdul Rasool**\

------------------------------------------------------------------------

**Repository:** `DL_plant_disease_cnn`\
**Application:** The Leaf Ledger
