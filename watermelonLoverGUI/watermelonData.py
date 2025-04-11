from os.path import isfile, join
from copy import deepcopy

# Datatype to hold data returned from ML model.

class watermelonData():
    def __init__(self):
        #Inputs
        self._wav_filepath = ""
        self._weight = 0.000
        
        #Outputs
        self.sweetness = -1
        self.quality = -1
        
        self.data_valid = False
        self.results_valid = False
    
    @property
    def weightData(self) -> float:
        return self._weight
        
    @weightData.setter
    def weightData(self, weight: float):
        self._weight = weight
        self.data_valid = (weight > 0.0 and isfile(self.wav_filepath))
    
    @property
    def audioData(self) -> str:
        return self._wav_filepath
    
    @audioData.setter
    def audioData(self, wav_filepath: str):
        """Updates referenced wav filepath and data validity status."""
        self._wav_filepath = wav_filepath
        self.data_valid = (weight > 0.0 and isfile(self.wav_filepath))
    
    def recordResults(self, sweetness, quality, weight):
        if (not self.data_valid):
            self.results_valid = False
        self.sweetness = sweetness
        self.quality = quality
        self.weight = weight
        self.results_valid = True

    def clearData(self):
        self.session_id = ""
        self.session_dir = ""
        self.wav_filename = ""
        self.sweetness = -1
        self.quality = -1
        self.weight = 0.000
        self.sweetness = -1
        self.quality = -1
        self.weight = -1
        self.data_valid = False
        self.results_valid = False

