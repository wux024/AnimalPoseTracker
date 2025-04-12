from pathlib import Path

FILE_DIR = Path(__file__).resolve()
CFG_DIR = FILE_DIR.parent
DATA_DIR = CFG_DIR / 'datasets'
MODEL_DIR = CFG_DIR /'models'
DEFAULT_CFG_PATH = CFG_DIR / 'default.yaml'

MODEL_YAML_PATHS = {
    'AnimalRTPose': MODEL_DIR / 'animalrtpose.yaml',
    'AnimalRTPose-P6': MODEL_DIR / 'animalrtpose-p6.yaml',
    'AnimalViTPose': MODEL_DIR / 'animalvitpose.yaml',
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