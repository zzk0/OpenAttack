import numpy as np
from .base import AttackGoal

class ClassifierGoal(AttackGoal):
    def __init__(self, target, targeted):
        self.target = target
        self.targeted = targeted
    
    @property
    def is_targeted(self):
        return self.targeted

    def check(self, adversarial_sample, prediction):
        if not isinstance(prediction, np.ndarray):
            prediction = prediction.float().cpu().numpy()
        if self.targeted:
            return np.all(np.isclose(prediction, self.target))
        else:
            return not np.all(np.isclose(prediction, self.target))
