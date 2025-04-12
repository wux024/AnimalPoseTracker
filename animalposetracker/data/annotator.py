from typing import List, Tuple

class AnimalPoseFrameExtractor:
    def __init__(self,
                 source_dir: str = 'sources',
                 output_dir: str = 'temp',
                 extract_method: str = 'auto',
                 extract_algorithm: str = 'kmeas',
                 cluster_step: int = 10,
                 *args, **kwargs):
        pass
class AnimalPoseAnnotator:
    def __init__(self,
                 source_dir: str = 'sources',
                 output_dir: str = 'datasets',
                 keypoints: int = 17,
                 keypoint_names: List[str] = [],
                 classes: int = 1,
                 class_names: List[str] = [],
                 visible: bool = True,
                 YOLO: bool = False,
                 COCO: bool = True,
                 train_val_test_split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 *args, **kwargs
                 ):
        pass
