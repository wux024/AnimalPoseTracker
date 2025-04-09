import os
import json





class AnimalPoseTrackerProject:
    def __init__(self, project_path: str, 
                 project_name: str, 
                 worker: str, 
                 model_type: str, 
                 model_scale: float, 
                 keypoints: int, 
                 classes: int, 
                 keypoints_name: list, 
                 classes_name: list, 
                 date: str, 
                 sources: list):
        
        self.project_path = project_path
        self.project_name = project_name
        self.worker = worker
        self.model_type = model_type
        self.model_scale = model_scale
        self.keypoints = keypoints
        self.classes = classes
        self.keypoints_name = keypoints_name
        self.classes_name = classes_name
        self.date = date
        self.sources = sources   
    
    def print_project_info(self):
        print("Project Path: ", self.project_path)
        print("Project Name: ", self.project_name)
        print("Worker: ", self.worker)
        print("Model Type: ", self.model_type)
        print("Model Scale: ", self.model_scale)
        print("Keypoints number: ", self.keypoints)
        print("Classes number: ", self.classes)
        print("Keypoints Name: ", self.keypoints_name)
        print("Classes Name: ", self.classes_name)
        print("Date: ", self.date)
        print("Sources: ", self.sources)
    
    def create_new_project(self):
        self.create_project_dirs()
        self.create_project_config()
        self.create_dataset_config()
        self.create_model_config()

    def create_public_dataset_project(self):
        pass

    def add_source_to_project(self, source_path: str):  
        pass

    def load_project(self):
        pass

    def save_project(self):         
        pass


    def create_project_dirs(self):
        dirs_to_create = [
            self.project_path,
            os.path.join(self.project_path, "configs"),
            os.path.join(self.project_path, "datasets"),
            os.path.join(self.project_path, "datasets", "images", "train"),
            os.path.join(self.project_path, "datasets", "images", "val"),
            os.path.join(self.project_path, "datasets", "images", "test"),
            os.path.join(self.project_path, "datasets", "labels", "train"),
            os.path.join(self.project_path, "datasets", "labels", "val"),
            os.path.join(self.project_path, "datasets", "labels", "test"),
            os.path.join(self.project_path, "datasets", "annotations", "train"),
            os.path.join(self.project_path, "datasets", "annotations", "val"),
            os.path.join(self.project_path, "datasets", "annotations", "test"),
            os.path.join(self.project_path, "pretrained"),
            os.path.join(self.project_path, "runs"),
            os.path.join(self.project_path, "sources"),
        ]

        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)

    def create_project_config(self):
        project_config = {
            "project_name": self.project_name,
            "worker": self.worker,
            "model_type": self.model_type,
            "model_scale": self.model_scale,
            "keypoints": self.keypoints,
            "classes": self.classes,
            "keypoints_name": self.keypoints_name,
            "classes_name": self.classes_name,
            "date": self.date,
            "sources": self.sources,
        }

        with open(os.path.join(self.project_path, "configs", "config.json"), "w") as f:
            json.dump(project_config, f, indent=4)

    def create_dataset_config(self):
        pass

    def create_model_config(self):
        pass

    def update_project_config(self):
        pass

    def update_dataset_config(self):
        pass

    def update_model_config(self):
        pass



