from enum import Enum
from configdataset import ConfigDataset

class Mode(Enum):
    TRAIN = ConfigDataset().getTrainPath()
    TEST = ConfigDataset().getTestPath()
    VAL = ConfigDataset().getValPath()