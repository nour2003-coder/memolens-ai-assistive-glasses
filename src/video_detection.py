
import cv2
from insightface.app import FaceAnalysis

# 1. Initialize the face detection model
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(640, 640))

# 2. Open the webcam
# Use 0 for your default webcam.
# For a video file, replace 0 with its path.
VIDEO_SOURCE = 0

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError("Could not open the webcam or video.")

print("Video started. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read the next frame.")
        break

    # 3. Detect faces in the current frame
    faces = app.get(frame)

    # 4. Draw a bounding box around every detected face
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        confidence = float(face.det_score)

        cv2.putText(
            frame,
            f"Face: {confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # 5. Display the number of detected faces
    cv2.putText(
        frame,
        f"Faces detected: {len(faces)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow("MemoLens - Live Face Detection", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Video stopped.")
