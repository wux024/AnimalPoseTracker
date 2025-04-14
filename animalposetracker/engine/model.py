from ultralytics import YOLO

class ModelEngine:
    def __init__(self, config = None):
        self.config = config
        self.model = None

    def model_init(self, config):
        self.config = config
        self.model = YOLO(model=self.config['model'])

    def train(self):
        self.config['name'] = 'train'
        self.model.train(**self.config)
    
    def val(self):
        self.config['name'] = 'val'
        self.model.val(**self.config)

    def evaluate(self):
        self.config['name'] = 'evaluate'
        self.model.val(**self.config)

    def predict(self):  
        self.config['name'] = 'predict'
        self.model.predict(**self.config)

    def export(self):
        self.config['name'] = 'export'
        self.model.export(**self.config)

