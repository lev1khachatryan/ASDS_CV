'''
Style Transfer Network - Main network, which combines all rest
'''

import tensorflow as tf

from utils import *
from functions import *

from encoder import Encoder
from decoder import Decoder
from SANet.samod import SAMod

class STNet:

    def __init__(self, encoder_weights_path):
        self.encoder = Encoder(encoder_weights_path)
        self.decoder = Decoder()
        self.SAModule = SAMod(512)

    def transform(self, content, style):
        # switch RGB to BGR
        content = tf.reverse(content, axis=[-1])
        style   = tf.reverse(style,   axis=[-1])

        # preprocess image
        content = self.encoder.preprocess(content)
        style   = self.encoder.preprocess(style)

        # encode image
        enc_c_layers = self.encoder.encode(content)
        enc_s_layers = self.encoder.encode(style)

        self.encoded_content_layers = enc_c_layers
        self.encoded_style_layers   = enc_s_layers

        target_features = self.SAModule.map(enc_c_layers['relu4_1'], enc_c_layers['relu5_1'], enc_s_layers['relu4_1'], enc_s_layers['relu5_1'])
        self.target_features = target_features

        # decode target features back to image
        generated_img = self.decoder.decode(target_features)

        # deprocess image
        generated_img = self.encoder.deprocess(generated_img)

        # switch BGR back to RGB
        generated_img = tf.reverse(generated_img, axis=[-1])

        # clip to 0..255
        generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)

        return generated_img