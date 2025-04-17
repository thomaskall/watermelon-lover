from .data_collection import *
from datetime import datetime

def make_timestamp():
    """Make a timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
