
import numpy as np


def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union between two bounding boxes.
    Each box: [x1, y1, x2, y2]
    """

    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)

    intersection = intersection_width * intersection_height

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )
    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    if union == 0:
        return 0.0

    return intersection / union


class FaceTracker:

    def __init__(self, iou_threshold=0.3, max_age=10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age

        self.tracks = {}
        self.next_track_id = 1

    def update(self, detections):
        """
        detections: list of dictionaries containing:
        {
            'bbox': [x1, y1, x2, y2],
            'identity': 'Ahmed' or 'Unknown',
            'similarity': 0.85
        }
        """

        # Increase age of existing tracks
        for track in self.tracks.values():
            track["missed"] += 1

        unmatched_tracks = set(self.tracks.keys())
        matched_detections = set()

        # Build candidate matches using IoU
        candidates = []

        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):

                iou = calculate_iou(
                    track["bbox"],
                    detection["bbox"]
                )

                if iou >= self.iou_threshold:
                    candidates.append(
                        (iou, track_id, detection_index)
                    )

        # Match highest IoU pairs first
        candidates.sort(reverse=True)

        for iou, track_id, detection_index in candidates:

            if track_id not in unmatched_tracks:
                continue

            if detection_index in matched_detections:
                continue

            detection = detections[detection_index]
            track = self.tracks[track_id]

            track["bbox"] = detection["bbox"]
            track["missed"] = 0

            # Accumulate recognized identity votes
            if detection["identity"] != "Unknown":
                track["votes"][detection["identity"]] = (
                    track["votes"].get(detection["identity"], 0) + 1
                )

            unmatched_tracks.remove(track_id)
            matched_detections.add(detection_index)

        # Create new tracks for unmatched detections
        for index, detection in enumerate(detections):

            if index in matched_detections:
                continue

            track_id = self.next_track_id
            self.next_track_id += 1

            votes = {}

            if detection["identity"] != "Unknown":
                votes[detection["identity"]] = 1

            self.tracks[track_id] = {
                "bbox": detection["bbox"],
                "missed": 0,
                "votes": votes
            }

        # Remove tracks not seen for too long
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed"] > self.max_age
        ]

        for track_id in expired:
            del self.tracks[track_id]

        # Return current tracks
        results = []

        for track_id, track in self.tracks.items():

            if track["missed"] > 0:
                continue

            if track["votes"]:
                identity = max(
                    track["votes"],
                    key=track["votes"].get
                )
            else:
                identity = "Unknown"

            results.append({
                "track_id": track_id,
                "bbox": track["bbox"],
                "identity": identity,
                "missed": track["missed"]
            })

        return results
