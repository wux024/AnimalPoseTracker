from .base import ModelEngine
class InferenceEngine(ModelEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)