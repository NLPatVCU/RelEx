import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.engine import Layer

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=False
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable, name="{}_module".format(self.name))
        self.trainable_weights += tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)
    
    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),as_dict=True,signature='default',)['default']
        result=tf.expand_dims(result,axis=0)
        print(result.shape)
        return result
    
 #   def compute_mask(self, inputs, mask=None):
  #      return K.not_equal(inputs, '--PAD--')
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],100, self.dimensions)
