from .watermelonData import WatermelonData
from .DataCollector import DataCollector
from datetime import datetime

def make_timestamp():
    """Make a timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
