import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from collections import defaultdict
from typing import Union
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QPushButton,
                              QVBoxLayout, QWidget)
from PySide6.QtGui import QPixmap, QImage
from tqdm import tqdm


class KeyframeExtractor:
    def __init__(self, output_dir: Union[str, Path]):
        self._output_dir = output_dir
        self.saved_frame_counter = 0
        self.total_frames_all_videos = 0
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(exist_ok=True, parents=True)

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value: Union[str, Path]):
        self._output_dir = value if isinstance(value, Path) else Path(value)
        self._output_dir.mkdir(exist_ok=True, parents=True)

    def _calculate_total_frames(self, video_paths):
        """Pre-calculate total frames across all videos for proper zero-padding"""
        total = 0
        for video_path in video_paths:
            cap = cv2.VideoCapture(str(video_path))
            total += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        return total

    def _get_zero_padded_name(self, frame_idx):
        """
        Generate zero-padded filename with global continuous numbering

        Args:
            frame_idx (int): Current global frame index
        Returns:
            str: Zero-padded string with length of len(str(total_frames_all_videos)) + 5
        """
        total_digits = len(str(self.total_frames_all_videos)) + 5
        return str(frame_idx).zfill(total_digits)

    def extract_keyframes_from_images(self, image_path,
                                      mode="auto",
                                      algorithm="kmeans",
                                      n_clusters=25,
                                      interval=1):
        """
        Extract keyframes from images using specified mode with continuous numbering

        Args:
            image_path (str): Path to directory containing images
            mode (str): Extraction mode - "manual", "clustering", "all" or "uniform"
            algorithm (str): Clustering algorithm (only used in clustering mode)
            n_clusters (int): Number of clusters (only used in clustering mode)
            interval (int): Sampling interval for uniform mode
        """
        image_extensions = ('*.jpg', '*.png', '*.jpeg', '*.webp', '*.bmp')
        all_images = []
        for ext in image_extensions:
            all_images.extend(list(Path(image_path).glob(ext)))
        self.total_frames_all_videos = len(all_images)
        self.saved_frame_counter = 0  # Reset counter for new extraction
        self._extract_keyframes(all_images, mode, algorithm, n_clusters, interval)

    def extract_keyframes_from_videos(self, video_paths,
                                      mode="auto",
                                      algorithm="kmeans",
                                      n_clusters=25,
                                      interval=1):
        """
        Extract keyframes from videos using specified mode with continuous numbering

        Args:
            video_paths (list): List of video file paths
            mode (str): Extraction mode - "manual", "clustering", "all" or "uniform"
            algorithm (str): Clustering algorithm (only used in clustering mode)
            n_clusters (int): Number of clusters (only used in clustering mode)
            interval (int): Sampling interval for uniform mode
        """
        self.total_frames_all_videos = self._calculate_total_frames(video_paths)
        self.saved_frame_counter = 0  # Reset counter for new extraction
        self._extract_keyframes(video_paths, mode, algorithm, n_clusters, interval)

    def _extract_keyframes(self, input_paths, mode, algorithm, n_clusters, interval):
        if mode == "manual":
            self._manual_extraction(input_paths)
        elif mode == "auto":
            if algorithm == "kmeans":
                self._kmeans_extraction(input_paths, n_clusters, interval)
            elif algorithm == "adaptive_kmeans":
                self._adaptive_kmeans_extraction(input_paths, n_clusters, interval)
            elif algorithm == "hierarchical":
                self._hierarchical_extraction(input_paths, n_clusters, interval)
            elif algorithm == "dbscan":
                self._dbscan_extraction(input_paths, n_clusters, interval)
            elif algorithm == "uniform":
                self._uniform_extraction(input_paths, interval)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
        elif mode == "all":
            self._extract_all_frames(input_paths)
        else:
            raise ValueError(f"Invalid extraction mode: {mode}")

    def _manual_extraction(self, input_paths):
        """Manual keyframe selection with continuous numbering"""
        app = QApplication([])
        for path in input_paths:
            path = Path(path)
            if path.suffix in ('.jpg', '.png', '.jpeg', '.webp', '.bmp'):
                frames = [cv2.imread(str(path))]
            else:
                cap = cv2.VideoCapture(str(path))
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()

            for frame in frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

                # Create QT window
                window = QWidget()
                layout = QVBoxLayout()

                # Add image label
                label = QLabel()
                label.setPixmap(QPixmap.fromImage(q_img))
                layout.addWidget(label)

                # Add buttons
                btn_layout = QHBoxLayout()
                keep_btn = QPushButton("Keep Frame")
                skip_btn = QPushButton("Skip Frame")
                btn_layout.addWidget(keep_btn)
                btn_layout.addWidget(skip_btn)
                layout.addLayout(btn_layout)

                window.setLayout(layout)
                window.show()

                # Button actions
                keep = False

                def on_keep():
                    nonlocal keep
                    keep = True
                    window.close()

                def on_skip():
                    nonlocal keep
                    keep = False
                    window.close()

                keep_btn.clicked.connect(on_keep)
                skip_btn.clicked.connect(on_skip)

                app.exec_()

                if keep:
                    # Save frame with global continuous numbering
                    frame_name = f"{self._get_zero_padded_name(self.saved_frame_counter)}.jpg"
                    output_path = self.output_dir / frame_name
                    cv2.imwrite(str(output_path), frame)
                    self.saved_frame_counter += 1

    def _kmeans_extraction(self, input_paths, n_clusters=25, sample_interval=1):
        """K-Means clustering with continuous numbering"""
        for path in tqdm(input_paths, desc="K-Means extraction", unit="file"):
            path = Path(path)
            if path.suffix in ('.jpg', '.png', '.jpeg', '.webp', '.bmp'):
                frame = cv2.imread(str(path))
                resized = cv2.resize(frame, (64, 64))
                features = resized.reshape(-1).astype(np.float32)
                frames = [features]
            else:
                cap = cv2.VideoCapture(str(path))
                frames = []
                local_frame_count = 0
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for _ in tqdm(range(frame_count), desc=f"Reading {path.name}", unit="frame", leave=False):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if local_frame_count % sample_interval == 0:
                        # Resize and flatten frame as feature vector
                        resized = cv2.resize(frame, (64, 64))
                        features = resized.reshape(-1).astype(np.float32)
                        frames.append(features)

                    local_frame_count += 1

                cap.release()

            if not frames:
                continue

            # Perform K-Means clustering
            X = np.array(frames)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=1000)
            kmeans.fit(X)

            # Find closest frames to cluster centers
            closest_indices = defaultdict(list)
            for i, label in enumerate(kmeans.labels_):
                closest_indices[label].append(i)

            # Save keyframes
            saved_count = 0
            for cluster_id in range(n_clusters):
                if cluster_id in closest_indices:
                    cluster_frames = X[closest_indices[cluster_id]]
                    centroid = kmeans.cluster_centers_[cluster_id]
                    distances = np.linalg.norm(cluster_frames - centroid, axis=1)
                    min_idx = np.argmin(distances)

                    if path.suffix in ('.jpg', '.png', '.jpeg', '.webp', '.bmp'):
                        keyframe = cv2.imread(str(path))
                    else:
                        cap = cv2.VideoCapture(str(path))
                        if path.suffix not in ('.jpg', '.png', '.jpeg', '.webp', '.bmp'):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, min_idx * sample_interval)
                        ret, keyframe = cap.read()
                        cap.release()

                    if ret:
                        frame_name = f"{self._get_zero_padded_name(self.saved_frame_counter)}.jpg"
                        output_path = self.output_dir / frame_name
                        cv2.imwrite(str(output_path), keyframe)
                        self.saved_frame_counter += 1
                        saved_count += 1

            print(f"Extracted {saved_count} keyframes from {path.name}")

    def _adaptive_kmeans_extraction(self, input_paths, max_clusters=100, sample_interval=1):
        """Adaptive K-Means clustering with continuous numbering"""
        for path in tqdm(input_paths, desc="Adaptive K-Means extraction", unit="file"):
            path = Path(path)
            if path.suffix in ('.jpg', '.png', '.jpeg', '.webp', '.bmp'):
                frame = cv2.imread(str(path))
                resized = cv2.resize(frame, (64, 64))
                features = resized.reshape(-1).astype(np.float32)
                frames = [features]
            else:
                cap = cv2.VideoCapture(str(path))
                frames = []
                local_frame_count = 0
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for _ in tqdm(range(frame_count), desc=f"Reading {path.name}", unit="frame", leave=False):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if local_frame_count % sample_interval == 0:
                        # Resize and flatten frame as feature vector
                        resized = cv2.resize(frame, (64, 64))
                        features = resized.reshape(-1).astype(np.float32)
                        frames.append(features)

                    local_frame_count += 1

                cap.release()

            if not frames:
                continue

            X = np.array(frames)
            best_k = 2
            best_score = -1
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(X)
                try:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                except ValueError:
                    continue

            kmeans = KMeans(n_clusters=best_k, random_state=42)
            kmeans.fit(X)

            closest_indices = {}
            for cluster_id in range(best_k):
                cluster_frames = X[kmeans.labels_ == cluster_id]
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_frames - centroid, axis=1)
                min_idx = np.argmin(distances)
                closest_indices[cluster_id] = np.where(kmeans.labels_ == cluster_id)[0][min_idx]

            saved_count = 0
            for idx in closest_indices.values():
                if path.suffix in ('.jpg', '.png', '.jpeg', '.webp', '.bmp'):
                    keyframe = cv2.imread(str(path))
                else:
                    cap = cv2.VideoCapture(str(path))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, keyframe = cap.read()
                    cap.release()

                if ret:
                    frame_name = f"{self._get_zero_padded_name(self.saved_frame_counter)}.jpg"
                    output_path = self.output_dir / frame_name
                    cv2.imwrite(str(output_path), keyframe)
                    self.saved_frame_counter += 1
                    saved_count += 1

            print(f"Extracted {saved_count} keyframes from {path.name}")

    def _hierarchical_extraction(self, input_paths, 
                                 distance_threshold = 100000, 
                                 sample_interval=1):
        """Hierarchical clustering with continuous numbering"""
        for path in tqdm(input_paths, desc="Hierarchical extraction", unit="file"):
            path = Path(path)
            if path.suffix in ('.jpg', '.png', '.jpeg', '.webp', '.bmp'):
                frame = cv2.imread(str(path))
                resized = cv2.resize(frame, (64, 64))
                features = resized.reshape(-1).astype(np.float32)
                frames = [features]
            else:
                cap = cv2.VideoCapture(str(path))
                frames = []
                local_frame_count = 0
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for _ in tqdm(range(frame_count), desc=f"Reading {path.name}", unit="frame", leave=False):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if local_frame_count % sample_interval == 0:
                        # Resize and flatten frame as feature vector
                        resized = cv2.resize(frame, (64, 64))
                        features = resized.reshape(-1).astype(np.float32)
                        frames.append(features)

                    local_frame_count += 1

                cap.release()

            if not frames:
                continue

            X = np.array(frames)
            clustering = AgglomerativeClustering(distance_threshold=distance_threshold, 
                                                 n_clusters=None)
            labels = clustering.fit_predict(X)

            unique_labels = np.unique(labels)
            saved_count = 0
            for label in unique_labels:
                cluster_frames = X[labels == label]
                centroid = np.mean(cluster_frames, axis=0)
                distances = np.linalg.norm(cluster_frames - centroid, axis=1)
                min_idx = np.argmin(distances)
                global_idx = np.where(labels == label)[0][min_idx]

                if path.suffix in ('.jpg', '.png', '.jpeg', '.webp', '.bmp'):
                    keyframe = cv2.imread(str(path))
                else:
                    cap = cv2.VideoCapture(str(path))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, global_idx)
                    ret, keyframe = cap.read()
                    cap.release()

                if ret:
                    frame_name = f"{self._get_zero_padded_name(self.saved_frame_counter)}.jpg"
                    output_path = self.output_dir / frame_name
                    cv2.imwrite(str(output_path), keyframe)
                    self.saved_frame_counter += 1
                    saved_count += 1

            print(f"Extracted {saved_count} keyframes from {path.name}")

    def _dbscan_extraction(self, input_paths, 
                           eps=10000,
                           min_samples=5,
                           sample_interval=1):
        """DBSCAN clustering with continuous numbering"""
        for path in tqdm(input_paths, desc="DBSCAN extraction", unit="file"):
            path = Path(path)
            if path.suffix in ('.jpg', '.png', '.jpeg', '.webp', '.bmp'):
                frame = cv2.imread(str(path))
                resized = cv2.resize(frame, (64, 64))
                features = resized.reshape(-1).astype(np.float32)
                frames = [features]
            else:
                cap = cv2.VideoCapture(str(path))
                frames = []
                local_frame_count = 0
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for _ in tqdm(range(frame_count), desc=f"Reading {path.name}", unit="frame", leave=False):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if local_frame_count % sample_interval == 0:
                        # Resize and flatten frame as feature vector
                        resized = cv2.resize(frame, (64, 64))
                        features = resized.reshape(-1).astype(np.float32)
                        frames.append(features)

                    local_frame_count += 1

                cap.release()

            if not frames:
                continue

            X = np.array(frames)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)

            unique_labels = np.unique(labels)
            saved_count = 0
            for label in unique_labels:
                if label == -1:  # 噪声点
                    continue
                cluster_frames = X[labels == label]
                centroid = np.mean(cluster_frames, axis=0)
                distances = np.linalg.norm(cluster_frames - centroid, axis=1)
                min_idx = np.argmin(distances)
                global_idx = np.where(labels == label)[0][min_idx]

                if path.suffix in ('.jpg', '.png', '.jpeg', '.webp', '.bmp'):
                    keyframe = cv2.imread(str(path))
                else:
                    cap = cv2.VideoCapture(str(path))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, global_idx)
                    ret, keyframe = cap.read()
                    cap.release()

                if ret:
                    frame_name = f"{self._get_zero_padded_name(self.saved_frame_counter)}.jpg"
                    output_path = self.output_dir / frame_name
                    cv2.imwrite(str(output_path), keyframe)
                    self.saved_frame_counter += 1
                    saved_count += 1

            print(f"Extracted {saved_count} keyframes from {path.name}")

    def _extract_all_frames(self, input_paths):
        """Extract all frames with continuous numbering"""
        for path in tqdm(input_paths, desc="Extract all frames", unit="file"):
            path = Path(path)
            if path.suffix in ('.jpg', '.png', '.jpeg', '.webp', '.bmp'):
                frame = cv2.imread(str(path))
                frame_name = f"{self._get_zero_padded_name(self.saved_frame_counter)}.jpg"
                output_path = self.output_dir / frame_name
                cv2.imwrite(str(output_path), frame)
                self.saved_frame_counter += 1
            else:
                cap = cv2.VideoCapture(str(path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for _ in tqdm(range(frame_count), desc=f"Reading {path.name}", unit="frame", leave=False):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_name = f"{self._get_zero_padded_name(self.saved_frame_counter)}.jpg"
                    output_path = self.output_dir / frame_name
                    cv2.imwrite(str(output_path), frame)
                    self.saved_frame_counter += 1
                cap.release()
                print(f"Extracted {frame_count} frames from {path.name}")

    def _uniform_extraction(self, input_paths, interval):
        """Uniform sampling keyframe extraction with continuous numbering"""
        index = 0
        for path in tqdm(input_paths, desc="Uniform extraction", unit="file"):
            path = Path(path)
            if path.suffix in ('.jpg', '.png', '.jpeg', '.webp', '.bmp'):
                if index % interval == 0:
                    frame = cv2.imread(str(path))
                    frame_name = f"{self._get_zero_padded_name(self.saved_frame_counter)}.jpg"
                    output_path = self.output_dir / frame_name
                    cv2.imwrite(str(output_path), frame)
                    self.saved_frame_counter += 1
                index += 1
            else:
                cap = cv2.VideoCapture(str(path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for frame_num in tqdm(range(frame_count), desc=f"Reading {path.name}", unit="frame", leave=False):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_num % interval == 0:
                        frame_name = f"{self._get_zero_padded_name(self.saved_frame_counter)}.jpg"
                        output_path = self.output_dir / frame_name
                        cv2.imwrite(str(output_path), frame)
                        self.saved_frame_counter += 1
                cap.release()
                print(f"Extracted {frame_count // interval} keyframes from {path.name}")

    