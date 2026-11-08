
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Initialize pretrained face analysis model
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)


def extract_embeddings(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Could not load image.")

    # Detect faces and extract ArcFace embeddings
    faces = app.get(image)

    results = []

    for face in faces:

        embedding = face.embedding

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        results.append({
            "embedding": embedding,
            "bbox": face.bbox.tolist(),
            "confidence": float(face.det_score)
        })

    return results


if __name__ == "__main__":

    image_path = "data/test_images/test.jpg"

    results = extract_embeddings(image_path)

    print("Detected faces:", len(results))

    for i, result in enumerate(results):

        print(f"\nFace {i + 1}")
        print("Embedding shape:", result["embedding"].shape)
        print("Embedding:", result["embedding"][:5])
        print("Detection confidence:", result["confidence"])
