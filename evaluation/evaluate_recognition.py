
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# -----------------------------
# 1. Configuration
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "face_database.npz"
TEST_DIR = PROJECT_ROOT / "data" / "test_images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

THRESHOLDS = np.arange(0.30, 0.81, 0.05)

# -----------------------------
# 2. Load registered identities
# -----------------------------

database = np.load(DATABASE_PATH)

names = database["names"]
embeddings = database["embeddings"]

print("Registered identities:", names)

# -----------------------------
# 3. Initialize InsightFace
# -----------------------------

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=-1, det_size=(640, 640))

# -----------------------------
# 4. Extract test predictions
# -----------------------------

results = []

for person_folder in TEST_DIR.iterdir():

    if not person_folder.is_dir():
        continue

    true_identity = person_folder.name

    for image_path in person_folder.iterdir():

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not read:", image_path)
            continue

        faces = app.get(image)

        if len(faces) == 0:
            print("No face detected:", image_path.name)
            continue

        if len(faces) > 1:
            print("Multiple faces detected:", image_path.name)
            continue

        face_embedding = faces[0].embedding
        face_embedding = face_embedding / np.linalg.norm(face_embedding)

        similarities = embeddings @ face_embedding

        best_index = np.argmax(similarities)
        best_similarity = float(similarities[best_index])

        predicted_identity = str(names[best_index])

        results.append({
            "image": image_path.name,
            "true_identity": true_identity,
            "predicted_identity": predicted_identity,
            "similarity": best_similarity
        })

# -----------------------------
# 5. Evaluate thresholds
# -----------------------------

df = pd.DataFrame(results)

if df.empty:
    raise ValueError("No valid test faces were processed.")

evaluation = []

for threshold in THRESHOLDS:

    predictions = np.where(
        df["similarity"] >= threshold,
        df["predicted_identity"],
        "Unknown"
    )

    true = df["true_identity"].values

    known_mask = true != "Unknown"
    unknown_mask = true == "Unknown"

    correct_known = (
        (predictions == true) & known_mask
    ).sum()

    false_rejects = (
        (predictions == "Unknown") & known_mask
    ).sum()

    false_identifications = (
        (predictions != true) &
        (predictions != "Unknown") &
        known_mask
    ).sum()

    false_accepts = (
        (predictions != "Unknown") & unknown_mask
    ).sum()

    far = (
        false_accepts / unknown_mask.sum()
        if unknown_mask.sum() else np.nan
    )

    frr = (
        (false_rejects + false_identifications) /
        known_mask.sum()
        if known_mask.sum() else np.nan
    )

    evaluation.append({
        "threshold": round(float(threshold), 2),
        "correct_known": int(correct_known),
        "false_rejects": int(false_rejects),
        "false_identifications": int(false_identifications),
        "false_accepts": int(false_accepts),
        "FAR": far,
        "FRR": frr
    })

evaluation_df = pd.DataFrame(evaluation)

# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

SELECTED_THRESHOLD = 0.70

# Apply the selected threshold
predictions = np.where(
    df["similarity"] >= SELECTED_THRESHOLD,
    df["predicted_identity"],
    "Unknown"
)

true_labels = df["true_identity"].astype(str)

class_names = sorted(names.astype(str).tolist()) + ["Unknown"]

cm = confusion_matrix(
    true_labels,
    predictions,
    labels=class_names
)

fig, ax = plt.subplots(figsize=(8, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title(
    f"MemoLens Confusion Matrix (threshold={SELECTED_THRESHOLD})"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrix saved.")


# ==========================================
# 8. FAR / FRR CURVE
# ==========================================

plt.figure(figsize=(9, 6))

plt.plot(
    evaluation_df["threshold"],
    evaluation_df["FAR"],
    marker="o",
    label="False Acceptance Rate (FAR)"
)

plt.plot(
    evaluation_df["threshold"],
    evaluation_df["FRR"],
    marker="s",
    label="False Rejection Rate (FRR)"
)

plt.xlabel("Similarity threshold")
plt.ylabel("Error rate")
plt.title("MemoLens - FAR vs FRR")

plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "far_frr_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("FAR/FRR curve saved.")

print("\nVisualization files generated:")
print(OUTPUT_DIR / "confusion_matrix.png")
print(OUTPUT_DIR / "far_frr_curve.png")

# -----------------------------
# 6. Save and display results
# -----------------------------
print("\nTest images per identity:")
print(df["true_identity"].value_counts())

known_count = (df["true_identity"] != "Unknown").sum()
unknown_count = (df["true_identity"] == "Unknown").sum()

print("\nKnown test images:", known_count)
print("Unknown test images:", unknown_count)

if known_count == 0:
    raise ValueError(
        "No registered identities found in test_images. "
        "Add separate test photos for Sarah and Ahmed."
    )

if unknown_count == 0:
    raise ValueError(
        "No unknown test images found. "
        "Add people who are not registered."
    )
OUTPUT_DIR.mkdir(exist_ok=True)

df.to_csv(
    OUTPUT_DIR / "recognition_predictions.csv",
    index=False
)

evaluation_df.to_csv(
    OUTPUT_DIR / "threshold_evaluation.csv",
    index=False
)

print("\nIndividual predictions:")
print(df.to_string(index=False))

print("\nThreshold evaluation:")
print(evaluation_df.to_string(index=False))

print("\nResults saved in:", OUTPUT_DIR)
