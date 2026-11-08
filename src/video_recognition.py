
import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

# ==========================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "data" / "face_database.npz"

# Provisional threshold based on our evaluation
MATCH_THRESHOLD = 0.70

# ==========================================
# 2. LOAD THE REGISTERED FACE DATABASE
# ==========================================

if not DATABASE_PATH.exists():
    raise FileNotFoundError(
        f"Face database not found: {DATABASE_PATH}"
    )

database = np.load(DATABASE_PATH)

names = database["names"]
embeddings = database["embeddings"].astype(np.float32)

# Normalize database embeddings
embeddings = embeddings / np.linalg.norm(
    embeddings, axis=1, keepdims=True
)

print("Registered identities:", names.tolist())

# ==========================================
# 3. INITIALIZE INSIGHTFACE
# ==========================================

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=-1, det_size=(640, 640))

# ==========================================
# 4. OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("MemoLens live recognition started.")
print("Press Q to quit.")

# ==========================================
# 5. RECOGNITION LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    faces = app.get(frame)

    for face in faces:

        # Extract and normalize the face embedding
        face_embedding = face.embedding.astype(np.float32)

        norm = np.linalg.norm(face_embedding)

        if norm == 0:
            continue

        face_embedding = face_embedding / norm

        # Calculate cosine similarities
        similarities = embeddings @ face_embedding

        # Find the closest registered identity
        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])

        # Apply recognition threshold
        if best_score >= MATCH_THRESHOLD:
            identity = str(names[best_index])
            color = (0, 255, 0)
            status = "RECOGNIZED"
        else:
            identity = "Unknown"
            color = (0, 0, 255)
            status = "NOT RECOGNIZED"

        # Draw bounding box
        x1, y1, x2, y2 = map(int, face.bbox)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Display identity and similarity
        label = f"{identity} | {best_score:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 25, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.putText(
            frame,
            status,
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    cv2.imshow("MemoLens - Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# 6. RELEASE RESOURCES
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("MemoLens recognition stopped.")
