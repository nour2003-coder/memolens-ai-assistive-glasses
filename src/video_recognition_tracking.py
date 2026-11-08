
import cv2
import numpy as np
import queue
import threading
import pyttsx3
import json
from pathlib import Path
from insightface.app import FaceAnalysis
from face_tracker import FaceTracker


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "data" / "face_database.npz"

MATCH_THRESHOLD = 0.70
PROFILES_PATH = BASE_DIR / "data" / "family_profiles.json"

if PROFILES_PATH.exists():
    with open(PROFILES_PATH, "r", encoding="utf-8") as file:
        family_profiles = json.load(file)
else:
    family_profiles = {}
# Voice announcement settings
speech_queue = queue.Queue()


# ==========================================
# 2. TEXT-TO-SPEECH WORKER
# ==========================================

def voice_worker():

    engine = pyttsx3.init()
    engine.setProperty("rate", 155)

    while True:

        message = speech_queue.get()

        if message is None:
            break

        engine.say(message)
        engine.runAndWait()

    engine.stop()


voice_thread = threading.Thread(
    target=voice_worker,
    daemon=True
)

voice_thread.start()


# ==========================================
# 3. LOAD FACE DATABASE
# ==========================================

database = np.load(DATABASE_PATH)

names = database["names"]
embeddings = database["embeddings"].astype(np.float32)

embeddings /= np.linalg.norm(
    embeddings, axis=1, keepdims=True
)

print("Registered identities:", names.tolist())


# ==========================================
# 4. INITIALIZE INSIGHTFACE
# ==========================================

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=-1, det_size=(640, 640))


# ==========================================
# 5. INITIALIZE TRACKER
# ==========================================

tracker = FaceTracker(
    iou_threshold=0.3,
    max_age=10
)

# Track IDs already announced
announced_track_ids = set()


# ==========================================
# 6. OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("MemoLens voice recognition started.")
print("Press Q to quit.")


# ==========================================
# 7. MAIN VIDEO LOOP
# ==========================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        faces = app.get(frame)

        detections = []

        # Face recognition
        for face in faces:

            embedding = face.embedding.astype(np.float32)

            norm = np.linalg.norm(embedding)

            if norm == 0:
                continue

            embedding /= norm

            similarities = embeddings @ embedding

            best_index = int(np.argmax(similarities))
            best_score = float(similarities[best_index])

            if best_score >= MATCH_THRESHOLD:
                identity = str(names[best_index])
            else:
                identity = "Unknown"

            bbox = list(map(int, face.bbox))

            detections.append({
                "bbox": bbox,
                "identity": identity,
                "similarity": best_score
            })

        # Update tracking
        tracks = tracker.update(detections)

        # Remove announcement history for expired tracks
        active_track_ids = set(tracker.tracks.keys())

        announced_track_ids.intersection_update(
            active_track_ids
        )

        # Display and announce identities
        for track in tracks:

            track_id = track["track_id"]
            identity = track["identity"]

            x1, y1, x2, y2 = track["bbox"]

            if identity == "Unknown":

                color = (0, 0, 255)
                label = f"ID {track_id}: Unknown"

            else:

                color = (0, 255, 0)
                label = f"ID {track_id}: {identity}"

                # Announce only once per track
                if track_id not in announced_track_ids:

                    profile = family_profiles.get(identity, {})

                    relationship = profile.get(
                        "relationship",
                        "un membre de votre famille"
                    )

                    message = f"Bonjour ! Voici {identity}, {relationship}."

                    speech_queue.put(message)

                    announced_track_ids.add(track_id)

                    print("Voice announcement:", message)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        cv2.imshow(
            "MemoLens - Recognition and Voice",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    cap.release()
    cv2.destroyAllWindows()

    # Stop voice worker
    speech_queue.put(None)
    voice_thread.join(timeout=3)

    print("MemoLens stopped.")
