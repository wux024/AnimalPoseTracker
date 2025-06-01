from pathlib import Path

FILE_DIR = Path(__file__).resolve()
CFG_DIR = FILE_DIR.parent
DATA_DIR = CFG_DIR / 'datasets'
MODEL_DIR = CFG_DIR /'models'
DEFAULT_CFG_PATH = CFG_DIR / 'default.yaml'

MODEL_YAML_PATHS = {
    'AnimalRTPose': MODEL_DIR / 'animalrtpose.yaml',
    'AnimalRTPose-P6': MODEL_DIR / 'animalrtpose-p6.yaml',
    #'AnimalViTPose': MODEL_DIR / 'animalvitpose.yaml',
    'SPIPose': MODEL_DIR /'spipose.yaml',
    'YOLOv12-Pose': MODEL_DIR / 'yolov12-pose.yaml',
    'YOLO11-Pose': MODEL_DIR / 'yolo11-pose.yaml',
    'YOLOv8-Pose': MODEL_DIR / 'yolov8-pose.yaml',
    'YOLOv8-Pose-P6': MODEL_DIR / 'yolov8-pose-p6.yaml',
}

DATA_YAML_PATHS = {
    'AcinoSet': DATA_DIR / 'acinoset.yaml',
    'AnimalKingdom': DATA_DIR / 'animalkingdom.yaml',
    'AnimalPose': DATA_DIR / 'animalpose.yaml',
    'AnimalPoseFly': DATA_DIR / 'animalposefly.yaml',
    'AnimalPoseMouse': DATA_DIR / 'animalposemouse.yaml',
    'AP10K': DATA_DIR / 'ap10k.yaml',
    'APT36K': DATA_DIR / 'apt36k.yaml',
    'APTv2': DATA_DIR / 'aptv2.yaml',
    'ATRW': DATA_DIR / 'atrw.yaml',
    'AWA': DATA_DIR / 'awa.yaml',
    'COCO-Pose': DATA_DIR / 'coco-pose.yaml',
    'Fish': DATA_DIR / 'fish.yaml',
    'Fly': DATA_DIR / 'fly.yaml',
    'Horse10': DATA_DIR / 'horse10.yaml',
    'locust': DATA_DIR / 'locust.yaml',
    'LoTE-Web': DATA_DIR / 'lote_web.yaml',
    'LOTE-Wild': DATA_DIR / 'lote_wild.yaml',
    'LSP': DATA_DIR / 'lsp.yaml',
    'Macaque': DATA_DIR /'macaque.yaml',
    'Marmoset': DATA_DIR /'marmoset.yaml',
    'Tri-Mouse': DATA_DIR /'trimouse.yaml',
    'OpenMonkeyChallenge': DATA_DIR / 'openmonkeychallenge.yaml',
    'Pups': DATA_DIR / 'pups.yaml',
    'Quadruped-80K': DATA_DIR / 'quadruped-80k.yaml',
    'StanfordExtra': DATA_DIR /'stanfordextra.yaml',
    'TopViewMouse': DATA_DIR / 'topviewmouse.yaml',
    'Zebra': DATA_DIR / 'zebra.yaml',
}

WEIGHT_URLS = {
    'AnimalRTPose':
    {
        'N': 'https://docs.google.com/uc?export=download&id=1wBrx_Aijh543GEvCF2_p0utmQ0CuWNmx',
        'S': 'https://docs.google.com/uc?export=download&id=1P2NP5IZT3rMHMAPAekhkg3hCYOpD5uKb',
        'M': 'https://docs.google.com/uc?export=download&id=1eoo3MW4teovN6iJ4lQgMMS1cpZkXac5f',
        'L': 'https://docs.google.com/uc?export=download&id=1H8928gh3tkjG06f5qJ7J9Pk_1QbyDtCz',
        'X': 'https://docs.google.com/uc?export=download&id=1Yb4oV3iXicHmEGSlOYL4WqLq1c2_DQOJ',
    },
    'AnimalRTPose-P6':
    {
        'N': None,
        'S': None,
        'M': None,
        'L': None,
        'X': None,
    },
    'YOLOv8-Pose':
    {
        'N': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt',
        'S': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-pose.pt',
        'M': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-pose.pt',
        'L': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-pose.pt',
        'X': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-pose.pt',
    },
    'YOLOv8-Pose-P6':
    {
        'N': None,
        'S': None,
        'M': None,
        'L': None,
        'X': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-pose-p6.pt',
    },
    'YOLO11-Pose':
    {
        'N': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt',
        'S': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt',
        'M': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt',
        'L': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt',
        'X': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt',
    },
    'YOLOv12-Pose':
    {
        'N': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt',
        'S': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s.pt',
        'M': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov12m.pt',
        'L': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov12l.pt',
        'X': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov12x.pt',
    }
}