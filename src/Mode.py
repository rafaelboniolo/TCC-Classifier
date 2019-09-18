from enum import Enum
from Classifier.src.ConfigDataset import ConfigDataset

class Mode(Enum):
    TRAIN = ConfigDataset().getTrainPath()
    TEST = ConfigDataset().getTestPath()
    VAL = ConfigDataset().getValPath()