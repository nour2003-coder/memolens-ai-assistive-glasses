
from pathlib import Path
import cv2
from retinaface import RetinaFace

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = PROJECT_ROOT / "data" / "test_images" / "test.jpg"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "detection_result.jpg"


def detect_faces(image_path, threshold=0.9):

    # Check image existence
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Could not load image.")

    # Run RetinaFace
    faces = RetinaFace.detect_faces(
        img_path=str(image_path),
        threshold=threshold
    )

    # Handle images without detected faces
    if not isinstance(faces, dict) or not faces:
        print("No faces detected.")
        return image, {}

    # Process every detected face
    for face_id, face_data in faces.items():

        x1, y1, x2, y2 = face_data["facial_area"]
        confidence = face_data["score"]

        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Display confidence
        label = f"{face_id}: {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Draw facial landmarks
        for point in face_data["landmarks"].values():

            x, y = map(int, point)

            cv2.circle(
                image,
                (x, y),
                3,
                (0, 0, 255),
                -1
            )

        print(f"Face: {face_id}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Bounding box: {[x1, y1, x2, y2]}")
        print("-------------------------")

    return image, faces


if __name__ == "__main__":

    result_image, detected_faces = detect_faces(IMAGE_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(OUTPUT_PATH), result_image)

    print(f"\nTotal detected faces: {len(detected_faces)}")
    print(f"Output saved to: {OUTPUT_PATH}")
