import os
import json
import base64
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import cv2
import numpy as np
import urllib.request

logger = logging.getLogger("visionaudit.face_service")

# Model URLs for OpenCV Zoo
YUNET_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


class FaceService:
    """
    Face Recognition & Biometric Access Control Service using OpenCV YuNet + SFace (ArcFace-based).
    Provides face detection, landmark alignment, 128-D facial embedding extraction,
    cosine similarity verification, and JSON/memory caching.
    """

    def __init__(
        self,
        models_dir: Union[str, Path] = "models",
        data_file: Union[str, Path] = "data/enrolled_faces.json"
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.yunet_path = self.models_dir / "face_detection_yunet_2023mar.onnx"
        self.sface_path = self.models_dir / "face_recognition_sface_2021dec.onnx"

        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        self.detector = None
        self.recognizer = None
        self.haar_cascade = None

        # In-memory user cache: { name: {"embedding": list, "enrolled_at": str, "thumbnail": str} }
        self.enrolled_users: Dict[str, Dict[str, Any]] = {}

        # Initialize models and load cached users
        self._init_models()
        self._load_cached_users()

    def _init_models(self):
        """Ensure model weights exist, then instantiate detector and recognizer."""
        try:
            # 1. Download YuNet if missing
            if not self.yunet_path.exists() or self.yunet_path.stat().st_size < 10000:
                logger.info(f"Downloading YuNet face detector model to {self.yunet_path}...")
                urllib.request.urlretrieve(YUNET_URL, self.yunet_path)
                logger.info("YuNet model downloaded successfully.")

            # 2. Download SFace if missing
            if not self.sface_path.exists() or self.sface_path.stat().st_size < 1000000:
                logger.info(f"Downloading SFace face recognizer model to {self.sface_path}...")
                urllib.request.urlretrieve(SFACE_URL, self.sface_path)
                logger.info("SFace model downloaded successfully.")

            # 3. Instantiate YuNet Detector
            if self.yunet_path.exists():
                self.detector = cv2.FaceDetectorYN.create(
                    str(self.yunet_path),
                    "",
                    (320, 320),
                    score_threshold=0.5,
                    nms_threshold=0.3,
                    top_k=5000
                )
                logger.info("YuNet Face Detector initialized.")

            # 4. Instantiate SFace Recognizer
            if self.sface_path.exists():
                self.recognizer = cv2.FaceRecognizerSF.create(
                    str(self.sface_path),
                    ""
                )
                logger.info("SFace Face Recognizer initialized.")

        except Exception as e:
            logger.warning(f"Error loading DNN face models: {e}. Falling back to Haar Cascade.")

        # Always initialize Haar Cascade as fallback
        try:
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(haar_path):
                self.haar_cascade = cv2.CascadeClassifier(haar_path)
                logger.info("Haar Cascade fallback classifier loaded.")
        except Exception as ex:
            logger.warning(f"Failed to load Haar Cascade: {ex}")

    def _load_cached_users(self):
        """Loads enrolled user embeddings from disk into memory."""
        if self.data_file.exists():
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.enrolled_users = data
                        logger.info(f"Loaded {len(self.enrolled_users)} enrolled personnel from {self.data_file}.")
                        return
            except Exception as e:
                logger.error(f"Failed to read {self.data_file}: {e}")

        # Seed sample demo personnel if no enrollments exist
        self._seed_default_personnel()

    def _seed_default_personnel(self):
        """Creates sample enrolled personnel with realistic synthetic embeddings for immediate testing."""
        np.random.seed(42)
        sample_users = [
            ("Dr. Sarah Connor", "Chief Safety Engineer"),
            ("Marcus Wright", "Authorized Facility Operator"),
            ("Elena Vance", "Senior Robotics Auditor")
        ]
        
        for name, role in sample_users:
            # Generate deterministic unit normalized 128-D vector
            vec = np.random.randn(128).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-6)
            
            # Create a simple SVG/canvas style avatar or placeholder thumbnail
            thumbnail = self._generate_avatar_thumbnail(name)

            self.enrolled_users[name] = {
                "name": name,
                "role": role,
                "embedding": vec.tolist(),
                "enrolled_at": datetime.now(timezone.utc).isoformat(),
                "thumbnail": thumbnail
            }
        
        self._save_cached_users()
        logger.info(f"Initialized {len(self.enrolled_users)} demo authorized personnel.")

    def _save_cached_users(self):
        """Serializes current enrolled personnel into JSON file."""
        try:
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(self.enrolled_users, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist enrolled faces to {self.data_file}: {e}")

    def _generate_avatar_thumbnail(self, name: str) -> str:
        """Generates a neat avatar thumbnail image in base64 format."""
        canvas = np.zeros((120, 120, 3), dtype=np.uint8)
        # Background gradient / color based on hash of name
        color_val = abs(hash(name)) % 180 + 40
        canvas[:] = (color_val // 2, 45, 100)
        
        # Draw head and shoulders silhouette
        cv2.circle(canvas, (60, 48), 24, (220, 225, 235), -1)
        cv2.ellipse(canvas, (60, 115), (42, 35), 0, 180, 360, (220, 225, 235), -1)
        
        # Initials
        initials = "".join([part[0].upper() for part in name.split()[:2] if part])
        cv2.putText(canvas, initials, (44, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 50), 2, cv2.LINE_AA)
        
        _, buf = cv2.imencode(".jpg", canvas)
        b64 = base64.b64encode(buf).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def decode_image(self, image_data: Union[bytes, str, np.ndarray]) -> Optional[np.ndarray]:
        """Decodes image from raw bytes, base64 data URI, or returns numpy array."""
        if isinstance(image_data, np.ndarray):
            return image_data

        if isinstance(image_data, str):
            # Check for data URI prefix: data:image/jpeg;base64,...
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            try:
                raw_bytes = base64.b64decode(image_data)
            except Exception as e:
                logger.error(f"Base64 decode error: {e}")
                return None
        elif isinstance(image_data, bytes):
            raw_bytes = image_data
        else:
            return None

        # Decode using OpenCV
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def detect_faces(self, image: np.ndarray, score_threshold: float = 0.5) -> List[Any]:
        """
        Detects faces in the given image.
        Returns list of face entries. If YuNet is used, each face is a 15-element array:
        [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]

        if self.detector is not None:
            try:
                self.detector.setInputSize((w, h))
                self.detector.setScoreThreshold(score_threshold)
                status, faces = self.detector.detect(image)
                if status and faces is not None and len(faces) > 0:
                    return [f for f in faces]
            except Exception as e:
                logger.debug(f"YuNet detection failed: {e}")

        # Fallback to Haar Cascade
        if self.haar_cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            boxes = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            face_list = []
            for (x, y, fw, fh) in boxes:
                # Mock 15-element array with approximate eye centers for SFace alignment
                re_x, re_y = x + int(fw * 0.3), y + int(fh * 0.35)
                le_x, le_y = x + int(fw * 0.7), y + int(fh * 0.35)
                nt_x, nt_y = x + int(fw * 0.5), y + int(fh * 0.55)
                rc_x, rc_y = x + int(fw * 0.35), y + int(fh * 0.75)
                lc_x, lc_y = x + int(fw * 0.65), y + int(fh * 0.75)
                face_list.append(np.array([
                    x, y, fw, fh,
                    re_x, re_y, le_x, le_y, nt_x, nt_y, rc_x, rc_y, lc_x, lc_y,
                    0.90
                ], dtype=np.float32))
            return face_list

        return []

    def extract_embedding(
        self,
        image: np.ndarray,
        face_info: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Extracts 128-D facial embedding using SFace or deep normalized feature extraction.
        """
        if image is None or image.size == 0:
            return None

        # 1. Primary path: SFace with landmark alignment
        if self.recognizer is not None:
            try:
                if face_info is not None and len(face_info) >= 14:
                    aligned = self.recognizer.alignCrop(image, face_info)
                else:
                    # If whole image is a cropped face, resize to standard 112x112 SFace input
                    aligned = cv2.resize(image, (112, 112))
                
                feat = self.recognizer.feature(aligned)
                # L2 normalize
                feat = feat.flatten().astype(np.float32)
                norm = np.linalg.norm(feat)
                if norm > 1e-6:
                    feat = feat / norm
                return feat
            except Exception as e:
                logger.warning(f"SFace feature extraction failed: {e}")

        # 2. Resilient fallback embedding if SFace failed
        try:
            resized = cv2.resize(image, (64, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # Calculate gradient magnitude and normalized frequency features
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag, _ = cv2.cartToPolar(gx, gy)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
            mag_hist = cv2.calcHist([mag.astype(np.uint8)], [0], None, [64], [0, 256]).flatten()
            feat = np.concatenate([hist, mag_hist]).astype(np.float32)
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                feat = feat / norm
            return feat
        except Exception as ex:
            logger.error(f"Fallback embedding generation failed: {ex}")
            return None

    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Computes cosine similarity between two unit or raw vectors.
        Returns value in [0.0, 1.0].
        """
        v1 = np.asarray(vec1, dtype=np.float32).flatten()
        v2 = np.asarray(vec2, dtype=np.float32).flatten()
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        dot = float(np.dot(v1, v2))
        similarity = dot / (norm1 * norm2)
        return max(0.0, min(1.0, float(similarity)))

    def register_face(
        self,
        name: str,
        image_data: Union[bytes, str, np.ndarray],
        role: str = "Authorized Personnel"
    ) -> Dict[str, Any]:
        """
        Admin enrollment: extracts facial embedding and saves to cache and disk.
        """
        img = self.decode_image(image_data)
        if img is None:
            raise ValueError("Failed to decode the provided face image.")

        # Detect face
        faces = self.detect_faces(img)
        face_info = faces[0] if faces else None

        # Extract 128-D embedding
        embedding = self.extract_embedding(img, face_info)
        if embedding is None:
            raise ValueError("Could not extract facial embedding from image.")

        # Create thumbnail for UI gallery
        if face_info is not None:
            x, y, w, h = [int(v) for v in face_info[:4]]
            x = max(0, x)
            y = max(0, y)
            w = min(img.shape[1] - x, max(20, w))
            h = min(img.shape[0] - y, max(20, h))
            crop = img[y : y + h, x : x + w]
            thumb_img = cv2.resize(crop, (120, 120)) if crop.size > 0 else cv2.resize(img, (120, 120))
        else:
            thumb_img = cv2.resize(img, (120, 120))

        _, buf = cv2.imencode(".jpg", thumb_img)
        thumb_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

        enrolled_record = {
            "name": name.strip(),
            "role": role.strip(),
            "embedding": embedding.tolist(),
            "enrolled_at": datetime.now(timezone.utc).isoformat(),
            "thumbnail": thumb_b64
        }

        self.enrolled_users[name.strip()] = enrolled_record
        self._save_cached_users()
        logger.info(f"Successfully registered face for '{name}'.")

        return {
            "status": "success",
            "name": name.strip(),
            "enrolled_at": enrolled_record["enrolled_at"],
            "embedding_dimension": len(enrolled_record["embedding"]),
            "message": f"Personnel '{name}' enrolled successfully."
        }

    def get_enrolled_users(self) -> List[Dict[str, Any]]:
        """Returns list of all enrolled personnel for admin gallery."""
        users_list = []
        for name, data in self.enrolled_users.items():
            users_list.append({
                "name": name,
                "role": data.get("role", "Authorized Personnel"),
                "enrolled_at": data.get("enrolled_at", ""),
                "thumbnail": data.get("thumbnail", self._generate_avatar_thumbnail(name)),
                "embedding_dimension": len(data.get("embedding", []))
            })
        return users_list

    def delete_user(self, name: str) -> bool:
        """Removes a user from memory and cache."""
        if name in self.enrolled_users:
            del self.enrolled_users[name]
            self._save_cached_users()
            logger.info(f"Removed enrolled user '{name}'.")
            return True
        return False

    def verify_access(
        self,
        image_or_crop: Union[bytes, str, np.ndarray, List[float]],
        threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Compares an incoming cropped face/frame vector against enrolled embeddings using cosine distance / similarity.
        Threshold = 0.6.
        Returns:
        {
            "access_granted": true/false,
            "matched_name": "...",
            "confidence": 0.94,
            "timestamp": "..."
        }
        """
        now_ts = datetime.now(timezone.utc).isoformat()

        # Handle direct vector input
        if isinstance(image_or_crop, (list, tuple)):
            input_vector = np.array(image_or_crop, dtype=np.float32)
            return self._match_vector(input_vector, threshold, now_ts)

        # Decode image
        img = self.decode_image(image_or_crop)
        if img is None:
            return {
                "access_granted": False,
                "matched_name": "Unknown",
                "confidence": 0.0,
                "timestamp": now_ts,
                "detail": "Invalid or unreadable image format."
            }

        # Detect faces in frame
        faces = self.detect_faces(img)
        detected_boxes = []

        if not faces:
            # Check if image itself is a cropped face
            embedding = self.extract_embedding(img, None)
            if embedding is not None:
                res = self._match_vector(embedding, threshold, now_ts)
                res["boxes"] = []
                return res
            else:
                return {
                    "access_granted": False,
                    "matched_name": "Unknown",
                    "confidence": 0.0,
                    "timestamp": now_ts,
                    "detail": "No face detected in frame.",
                    "boxes": []
                }

        best_match = {
            "access_granted": False,
            "matched_name": "Unknown",
            "confidence": 0.0,
            "timestamp": now_ts
        }

        # Evaluate each detected face in frame
        for face in faces:
            x, y, w, h = [int(v) for v in face[:4]]
            emb = self.extract_embedding(img, face)
            if emb is None:
                continue

            match_res = self._match_vector(emb, threshold, now_ts)
            box_info = {
                "box": [x, y, w, h],
                "access_granted": match_res["access_granted"],
                "matched_name": match_res["matched_name"],
                "confidence": match_res["confidence"]
            }
            detected_boxes.append(box_info)

            # Prioritize best authorized match, or highest confidence
            if match_res["confidence"] > best_match["confidence"]:
                best_match = match_res

        best_match["boxes"] = detected_boxes
        return best_match

    def _match_vector(
        self,
        query_vector: np.ndarray,
        threshold: float,
        timestamp: str
    ) -> Dict[str, Any]:
        """Compares a query embedding vector against all enrolled users."""
        if not self.enrolled_users:
            return {
                "access_granted": False,
                "matched_name": "Unknown",
                "confidence": 0.0,
                "timestamp": timestamp,
                "detail": "No enrolled personnel in database."
            }

        best_name = "Unknown"
        best_similarity = 0.0

        for name, data in self.enrolled_users.items():
            enrolled_vec = np.array(data["embedding"], dtype=np.float32)
            similarity = self.compute_cosine_similarity(query_vector, enrolled_vec)
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name

        access_granted = bool(best_similarity >= threshold)
        matched_name = best_name if access_granted else "Unknown"

        return {
            "access_granted": access_granted,
            "matched_name": matched_name,
            "confidence": round(float(best_similarity), 4),
            "timestamp": timestamp
        }


# Global singleton instance
face_service = FaceService()
