from model import BaseModel
import numpy as np

class TestBasicModel(BaseModel):
    def predict_proba(self, data):
        return np.array([[0.5] * 2] * len(data))
