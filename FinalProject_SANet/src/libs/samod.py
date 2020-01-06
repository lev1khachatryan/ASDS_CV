'''
Style-Attentional Module combines self style-attentional 
networks to construct whole module
'''

import numpy as np
import tensorflow as tf
from sanet import SANet
# import tensorflow.contrib as tf_contrib
from functions import *


class SAMod:
    '''
    Style-Attentional Module combines self style-attentional networks
    to construct whole module
    '''
    def __init__(self, num_filter):
        self.num_filter = num_filter
        self.SANet1 = SANet(num_filter)
        self.SANet2 = SANet(num_filter)

    def map(self, content_4_1, content_5_1, style_4_1, style_5_1):
        Fcsc_4 = self.SANet1.map(content_4_1, style_4_1)
        
        Fcsc_5 = self.SANet2.map(content_5_1, style_5_1)
        Fcsc_5_up = upsample(Fcsc_5)
        
        Fcsc_4_plus_5 = Fcsc_4 + Fcsc_5_up
        Fcsc_m = conv(Fcsc_4_plus_5, self.num_filter, kernel=3, stride=1, pad_type='reflect')
        
        return Fcsc_m