"""
Model definition and dataset metadata for the Plant Disease CNN.

This mirrors the architecture and class list trained in the source notebook
(project1.ipynb) so a saved state_dict from that notebook loads directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Class list (38 classes, in the exact order ImageFolder produced them)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)

# Per-class test-set accuracy, taken directly from the notebook's evaluation
# run (Section 20, confusion-matrix diagonal / row sums). Used only to show
# the model's known reliability for whichever class it predicts.
PER_CLASS_ACCURACY = {
    "Apple___Apple_scab": 86.71,
    "Apple___Black_rot": 98.79,
    "Apple___Cedar_apple_rust": 95.68,
    "Apple___healthy": 95.62,
    "Blueberry___healthy": 98.24,
    "Cherry_(including_sour)___Powdery_mildew": 95.72,
    "Cherry_(including_sour)___healthy": 99.56,
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 86.10,
    "Corn_(maize)___Common_rust_": 99.58,
    "Corn_(maize)___Northern_Leaf_Blight": 94.55,
    "Corn_(maize)___healthy": 99.57,
    "Grape___Black_rot": 92.16,
    "Grape___Esca_(Black_Measles)": 97.92,
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": 99.30,
    "Grape___healthy": 92.67,
    "Orange___Haunglongbing_(Citrus_greening)": 98.61,
    "Peach___Bacterial_spot": 91.72,
    "Peach___healthy": 98.38,
    "Pepper,_bell___Bacterial_spot": 92.68,
    "Pepper,_bell___healthy": 91.55,
    "Potato___Early_blight": 97.53,
    "Potato___Late_blight": 88.25,
    "Potato___healthy": 95.39,
    "Raspberry___healthy": 97.75,
    "Soybean___healthy": 97.03,
    "Squash___Powdery_mildew": 99.54,
    "Strawberry___Leaf_scorch": 97.97,
    "Strawberry___healthy": 98.68,
    "Tomato___Bacterial_spot": 95.76,
    "Tomato___Early_blight": 75.42,
    "Tomato___Late_blight": 88.12,
    "Tomato___Leaf_Mold": 95.53,
    "Tomato___Septoria_leaf_spot": 84.40,
    "Tomato___Spider_mites Two-spotted_spider_mite": 95.40,
    "Tomato___Target_Spot": 87.96,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 95.51,
    "Tomato___Tomato_mosaic_virus": 98.88,
    "Tomato___healthy": 98.34,
}

# Training log, straight from the notebook's 10-epoch run (Section 14 output).
TRAINING_LOG = [
    {"epoch": 1, "train_loss": 1.7521, "train_acc": 47.31, "val_loss": 0.7320, "val_acc": 77.99},
    {"epoch": 2, "train_loss": 1.0095, "train_acc": 68.34, "val_loss": 0.4653, "val_acc": 85.48},
    {"epoch": 3, "train_loss": 0.7822, "train_acc": 75.17, "val_loss": 0.3535, "val_acc": 89.45},
    {"epoch": 4, "train_loss": 0.6589, "train_acc": 79.14, "val_loss": 0.3292, "val_acc": 89.62},
    {"epoch": 5, "train_loss": 0.5784, "train_acc": 81.48, "val_loss": 0.3651, "val_acc": 88.17},
    {"epoch": 6, "train_loss": 0.5177, "train_acc": 83.28, "val_loss": 0.2665, "val_acc": 91.45},
    {"epoch": 7, "train_loss": 0.4737, "train_acc": 84.98, "val_loss": 0.2002, "val_acc": 93.51},
    {"epoch": 8, "train_loss": 0.4309, "train_acc": 86.09, "val_loss": 0.2086, "val_acc": 93.21},
    {"epoch": 9, "train_loss": 0.3908, "train_acc": 87.31, "val_loss": 0.1579, "val_acc": 94.81},
    {"epoch": 10, "train_loss": 0.3696, "train_acc": 88.26, "val_loss": 0.1605, "val_acc": 94.66},
]

TEST_ACCURACY = 94.53  # overall held-out test accuracy from the notebook

IMG_SIZE = 128
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]


class PlantDiseaseCNN(nn.Module):
    """Exact architecture from the source notebook (Section 10)."""

    def __init__(self, num_classes=NUM_CLASSES):
        super(PlantDiseaseCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 128x128 -> 64x64 -> 32x32 -> 16x16
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_model(weights_path, device="cpu"):
    """Loads PlantDiseaseCNN with trained weights. Raises if the file is missing/bad."""
    model = PlantDiseaseCNN(NUM_CLASSES)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def parse_class_name(raw_name):
    """Splits an ImageFolder class name like 'Tomato___Late_blight' into
    (crop, condition, is_healthy)."""
    if "___" in raw_name:
        crop, condition = raw_name.split("___", 1)
    else:
        crop, condition = raw_name, ""
    crop = crop.replace("_", " ").replace(",", ",").strip()
    condition = condition.replace("_", " ").strip()
    is_healthy = condition.lower() == "healthy"
    return crop, condition, is_healthy


def care_notes(raw_name):
    """General, non-prescriptive care pointers keyed off keywords in the
    predicted class. Deliberately generic — this is not agronomic advice,
    just a starting point for the grower to investigate further."""
    name = raw_name.lower()
    if "healthy" in name:
        return [
            "No signs of the diseases this model was trained to recognize.",
            "Keep up current watering and spacing habits.",
            "Recheck periodically, especially after weather changes.",
        ]
    if "rust" in name:
        return [
            "Rust fungi favor humid, still air — improve airflow around plants.",
            "Remove and dispose of heavily infected leaves rather than composting them.",
            "A local extension office can confirm the strain and suggest fungicide timing.",
        ]
    if "blight" in name:
        return [
            "Blights spread fast in wet foliage — water at the base, not overhead.",
            "Prune out affected stems and leaves promptly.",
            "Rotate crops next season if this is a recurring field.",
        ]
    if "mildew" in name:
        return [
            "Powdery mildew thrives in dry air but shaded, crowded canopies.",
            "Thin nearby foliage to increase light and airflow.",
            "Avoid high-nitrogen feeding until the outbreak clears.",
        ]
    if "spot" in name or "scab" in name:
        return [
            "Leaf-spot diseases usually enter through splashed soil or water.",
            "Mulch beneath plants to reduce soil splash-back onto leaves.",
            "Remove fallen infected leaves at season's end.",
        ]
    if "rot" in name or "measles" in name or "canker" in name:
        return [
            "Rots and cankers often follow wounds or waterlogged roots.",
            "Check drainage and avoid overwatering.",
            "Cut out affected wood well below the visible damage, sterilizing tools between cuts.",
        ]
    if "virus" in name or "greening" in name or "curl" in name:
        return [
            "Viral and bacterial systemic diseases are usually spread by insects.",
            "Inspect for aphids, whiteflies, or psyllids nearby.",
            "There is no cure once infected — focus on removing the source and controlling vectors.",
        ]
    if "mite" in name:
        return [
            "Spider mites build up fast in hot, dry conditions.",
            "A strong water spray on leaf undersides can knock back light infestations.",
            "Watch for fine webbing as a sign of a heavier infestation.",
        ]
    return [
        "Isolate the affected plant from healthy neighbors if possible.",
        "Photograph progression over the next few days.",
        "A local extension office can confirm diagnosis and treatment.",
    ]