'''
Style Transfer Network - Main network, which combines all rest
'''

import tensorflow as tf

# from utils import *
from functions import *

from encoder import Encoder
from decoder import Decoder
from samod import SAMod

class STNet:

    def __init__(self, encoder_weights_path):
        self.encoder = Encoder(encoder_weights_path)
        self.decoder = Decoder()
        self.SAModule = SAMod(512)

    def transform(self, content, style):
        # switch RGB to BGR
        # content = tf.reverse(content, axis=[-1])
        # style   = tf.reverse(style,   axis=[-1])

        # preprocess image
        # content = self.encoder.preprocess(content)
        # style   = self.encoder.preprocess(style)

        # encode image
        enc_c_layers = self.encoder.encode(content)
        enc_s_layers = self.encoder.encode(style)

        self.encoded_content_layers = enc_c_layers
        self.encoded_style_layers   = enc_s_layers

        Fcsc_m = self.SAModule.map(enc_c_layers['relu4_1'], enc_c_layers['relu5_1'], enc_s_layers['relu4_1'], enc_s_layers['relu5_1'])
        self.Fcsc_m = Fcsc_m

        # decode target features back to image (generate image)
        Ics = self.decoder.decode(Fcsc_m)

        # deprocess image
        # Ics = self.encoder.deprocess(Ics)

        # switch BGR back to RGB
        # Ics = tf.reverse(Ics, axis=[-1])

        # clip to 0..255
        # Ics = tf.clip_by_value(Ics, 0.0, 255.0)

        return Ics