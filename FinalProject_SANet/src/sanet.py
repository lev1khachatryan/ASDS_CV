import numpy as np
import tensorflow as tf
from utils import *


class SANet:
    '''
    Style-Attentional Network learns the mapping 
    between the content features and the style features 
    by slightly modifying the self-attention mechanism
    '''

    def __init__(self, in_dim):
        