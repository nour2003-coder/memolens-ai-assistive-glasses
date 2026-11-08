
from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = ROOT / "data" / "face_database.npz"
IMAGE_PATH = ROOT / "data" / "test_images" / "test.jpg"
OUTPUT_PATH = ROOT / "outputs" / "recognition_result.jpg"

# Temporary experimental threshold.
# Must be calibrated on separate validation data.
MATCH_THRESHOLD = 0.45

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=-1, det_size=(640, 640))


def load_database():

    database = np.load(DATABASE_PATH)

    names = database["names"]
    embeddings = database["embeddings"]

    return names, embeddings


def cosine_similarity(a, b):

    return float(
        np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b))
    )


def recognize(image_path):

    names, database = load_database()

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Could not load image.")

    faces = app.get(image)

    for face in faces:

        query_embedding = face.embedding

        scores = [
            cosine_similarity(query_embedding, registered)
            for registered in database
        ]

        best_index = int(np.argmax(scores))
        best_score = scores[best_index]

        if best_score >= MATCH_THRESHOLD:
            identity = str(names[best_index])
            status = "POTENTIAL MATCH"
        else:
            identity = "Unknown"
            status = "UNKNOWN"

        print(
            f"Result: {identity} | "
            f"Similarity: {best_score:.4f} | "
            f"Status: {status}"
        )

        x1, y1, x2, y2 = map(int, face.bbox)

        color = (
            (0, 255, 0)
            if identity != "Unknown"
            else (0, 0, 255)
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = f"{identity}: {best_score:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(str(OUTPUT_PATH), image)

    print("\nTotal faces:", len(faces))
    print("Result saved:", OUTPUT_PATH)


if __name__ == "__main__":
    recognize(IMAGE_PATH)
