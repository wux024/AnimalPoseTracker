import cv2
import os
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from collections import defaultdict
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QPushButton, 
                              QVBoxLayout, QWidget)
from PySide6.QtGui import QPixmap, QImage

from animalposetracker.projector import AnimalPoseTrackerProject

class KeyframeExtractor:
    def __init__(self, project):
        self.project = project
        self.output_dir = self.project.project_path / "source" / "extracted"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.global_frame_counter = 0
        self.total_frames_all_videos = 0
        
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

    def extract_keyframes(self, video_paths, mode="clustering", algorithm="kmeans", n_clusters=10):
        """
        Extract keyframes from videos using specified mode with continuous numbering
        
        Args:
            video_paths (list): List of video file paths
            mode (str): Extraction mode - "manual", "clustering", or "all"
            algorithm (str): Clustering algorithm (only used in clustering mode)
            n_clusters (int): Number of clusters (only used in clustering mode)
        """
        self.total_frames_all_videos = self._calculate_total_frames(video_paths)
        self.global_frame_counter = 0  # Reset counter for new extraction
        
        if mode == "manual":
            self._manual_extraction(video_paths)
        elif mode == "clustering":
            if algorithm == "kmeans":
                self._kmeans_extraction(video_paths, n_clusters)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
        elif mode == "all":
            self._extract_all_frames(video_paths)
        else:
            raise ValueError(f"Invalid extraction mode: {mode}")

    def _manual_extraction(self, video_paths):
        """Manual keyframe selection with continuous numbering"""
        app = QApplication([])
        
        for video_path in video_paths:
            cap = cv2.VideoCapture(str(video_path))
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Display frame in QT window
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
                    frame_name = f"{self._get_zero_padded_name(self.global_frame_counter)}.jpg"
                    output_path = self.output_dir / frame_name
                    cv2.imwrite(str(output_path), frame)
                
                self.global_frame_counter += 1
                frame_count += 1
            
            cap.release()
            print(f"Processed {frame_count} frames from {video_path.name}")

    def _kmeans_extraction(self, video_paths, n_clusters=10, sample_interval=5):
        """K-Means clustering with continuous numbering"""
        for video_path in video_paths:
            cap = cv2.VideoCapture(str(video_path))
            
            # Read and sample frames
            frames = []
            frame_indices = []  # Stores global frame indices
            local_frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if local_frame_count % sample_interval == 0:
                    # Resize and flatten frame as feature vector
                    resized = cv2.resize(frame, (64, 64))
                    features = resized.reshape(-1).astype(np.float32)
                    frames.append(features)
                    frame_indices.append(self.global_frame_counter)
                
                self.global_frame_counter += 1
                local_frame_count += 1
            
            if not frames:
                continue
            
            # Perform K-Means clustering
            X = np.array(frames)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
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
                    
                    # Get original global frame index
                    global_idx = frame_indices[closest_indices[cluster_id][min_idx]]
                    
                    # Seek to the frame in video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame_count - (self.global_frame_counter - global_idx - 1))
                    ret, keyframe = cap.read()
                    
                    if ret:
                        frame_name = f"{self._get_zero_padded_name(global_idx)}.jpg"
                        output_path = self.output_dir / frame_name
                        cv2.imwrite(str(output_path), keyframe)
                        saved_count += 1
            
            cap.release()
            print(f"Extracted {saved_count} keyframes from {video_path.name}")

    def _extract_all_frames(self, video_paths):
        """Extract all frames with continuous numbering"""
        for video_path in video_paths:
            cap = cv2.VideoCapture(str(video_path))
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame with global continuous numbering
                frame_name = f"{self._get_zero_padded_name(self.global_frame_counter)}.jpg"
                output_path = self.output_dir / frame_name
                cv2.imwrite(str(output_path), frame)
                
                self.global_frame_counter += 1
                frame_count += 1
            
            cap.release()
            print(f"Extracted {frame_count} frames from {video_path.name}")