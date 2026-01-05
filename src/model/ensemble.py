
class Ensemble:

    def __init__(self, models : dict):
        self.models = models
        self.total_weights = 0 
        for name, model in models.items():
            pipe = self._load(path=model)
            weight = self._get_recall(pipe)
            self.models[name] = (pipe, weight)
            self.total_weights += weight
        self.thresholds = self.total_weights / 2

    def _get_recall(self,model):
        pass

    def _load(self,path:str):
        pass

    def classify(self,raw):
        pass