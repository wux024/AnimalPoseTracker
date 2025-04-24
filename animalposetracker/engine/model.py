import subprocess
class ModelEngine:
    def __init__(self, config: str):
        self.config = config

    def train(self):
        cmd = [
            "yolo",
            "pose",
            "train",
            f"cfg={self.config}"
        ]
        subprocess.run(cmd)
            
    
    def val(self):
        cmd = [
            "yolo",
            "pose",
            "val",
            f"cfg={self.config}"
        ]
        subprocess.run(cmd)

    def evaluate(self):
        cmd = [
            "yolo",
            "pose",
            "val",
            f"cfg={self.config}"
        ]
        subprocess.run(cmd)

    def predict(self):  
        cmd = [
            "yolo",
            "pose",
            "predict",
            f"cfg={self.config}"
        ]
        subprocess.run(cmd)

    def export(self):
        cmd = [
            "yolo",
            "pose",
            "export",
            f"cfg={self.config}"
        ]
        subprocess.run(cmd)

