
from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "data" / "registered_faces"
DATABASE_PATH = ROOT / "data" / "face_database.npz"

# Initialize ArcFace pipeline
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=-1, det_size=(640, 640))


def normalize(vector):
    norm = np.linalg.norm(vector)

    if norm == 0:
        raise ValueError("Invalid zero embedding.")

    return vector / norm


def register_person(person_dir):

    embeddings = []

    image_extensions = {".jpg", ".jpeg", ".png"}

    images = [
        p for p in person_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    for image_path in images:

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not load: {image_path.name}")
            continue

        faces = app.get(image)

        # Require exactly one face
        if len(faces) != 1:
            print(
                f"Skipping {image_path.name}: "
                f"expected 1 face, found {len(faces)}"
            )
            continue

        embedding = normalize(faces[0].embedding)

        embeddings.append(embedding)

    if not embeddings:
        print(f"No valid images for {person_dir.name}")
        return None

    # Average embeddings into a person prototype
    prototype = np.mean(embeddings, axis=0)
    prototype = normalize(prototype)

    print(
        f"{person_dir.name}: "
        f"{len(embeddings)} images registered"
    )

    return prototype


def main():

    if not DATASET_DIR.exists():
        raise FileNotFoundError(DATASET_DIR)

    identities = {}
    names = []
    vectors = []

    for person_dir in DATASET_DIR.iterdir():

        if not person_dir.is_dir():
            continue

        embedding = register_person(person_dir)

        if embedding is not None:
            names.append(person_dir.name)
            vectors.append(embedding)

    if not vectors:
        raise ValueError("No identities registered.")

    np.savez_compressed(
        DATABASE_PATH,
        names=np.array(names),
        embeddings=np.stack(vectors)
    )

    print("\nRegistration completed!")
    print("Registered identities:", names)
    print("Database:", DATABASE_PATH)


if __name__ == "__main__":
    main()
