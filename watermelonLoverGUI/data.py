from enum import Enum
# Datatype to hold data returned from ML model.

class data():
    def __init__(self):
        self.sweetness = -1
        self.quality = -1
        self.valid = False

    def recordData(self, sweetness, quality):
        self.sweetness = sweetness
        self.quality = quality
        self.valid = True

    def clearData(self):
        self.sweetness = -1
        self.quality = -1
        self.valid = False

