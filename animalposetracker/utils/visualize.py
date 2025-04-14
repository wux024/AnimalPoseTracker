from pathlib import Path
from typing import List, Tuple

class Visualizer:
    def visualize(self) -> None:
        """
        Main visualization method that handles both directories and files.
        Determines whether the source is a directory or file and processes accordingly.
        """
        # Convert source path to Path object for easier manipulation
        source = Path(self.config['source'])

        # Validate path existence
        if not source.exists():
            raise FileNotFoundError(f"Source path does not exist: {source}")

        # Process based on path type
        if source.is_dir():
            self._process_directory(source)
        elif source.is_file():
            self._process_file(source)
        else:
            raise ValueError(f"Unsupported source type: {source}")

    def _process_directory(self, dir_path: Path) -> None:
        """
        Processes all valid media files in a directory.
        Separates files into images and videos for specialized processing.
        
        Args:
            dir_path: Path object pointing to the directory to process
        """
        # Supported file extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}

        # Initialize containers for found files
        images = []
        videos = []

        # Classify files in directory
        for item in dir_path.iterdir():
            if item.suffix.lower() in image_extensions:
                images.append(item)
            elif item.suffix.lower() in video_extensions:
                videos.append(item)

        # Process images if any found
        if images:
            print(f"Processing {len(images)} images...")
            for img_path in images:
                self._visualize_image(img_path)

        # Process videos if any found
        if videos:
            print(f"Processing {len(videos)} videos...")
            for vid_path in videos:
                self._visualize_video(vid_path)

    def _process_file(self, file_path: Path) -> None:
        """
        Processes a single file based on its extension.
        
        Args:
            file_path: Path object pointing to the file to process
        """
        # Supported file extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}

        # Route to appropriate processor
        if file_path.suffix.lower() in image_extensions:
            self._visualize_image(file_path)
        elif file_path.suffix.lower() in video_extensions:
            self._visualize_video(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _visualize_image(self, image_path: Path) -> None:
        """
        Core image visualization logic.
        
        Args:
            image_path: Path object pointing to the image to visualize
        """
        print(f"Visualizing image: {image_path}")
        # Implementation example:
        # 1. Load image using OpenCV/PIL
        # 2. Apply visualization algorithms
        # 3. Save/display results

    def _visualize_video(self, video_path: Path) -> None:
        """
        Core video visualization logic.
        
        Args:
            video_path: Path object pointing to the video to visualize
        """
        print(f"Visualizing video: {video_path}")
        # Implementation example:
        # 1. Open video stream using OpenCV
        # 2. Process frame-by-frame
        # 3. Generate output visualization