# CNN models

This provides instructions on the CNN models explained before. We used 3 Convolutional Neural Network (CNN) models here. Models differ based on the representation of the data fed into the models.
1. Segment CNN
2. Sentence CNN
  a. Single label Sentence CNN
  b. Multi label Sentence CNN
  
Following parameters and flags can be set / tuned in all 3 models. Default values are mentioned.
### Flags:
- embedding - whether to use the pre-trained word embedding weights (default-False)
- Cross_validation - perform cross validation (default-False)

### Parameters: 
- epochs - number of times the data is fed into the model (default-20)
- batch_size - size of a batch (default-512)
- filters - number of output filters (default-32)
- filter_conv - number of convolution filters (default-1)
- filter_maxPool - number of pooling filters (default-5)
- activation - (default-'relu')
- output_activation - activation function of the output layer (default-'sigmoid')
- drop_out- regularized technique to reduce overfitting (default-0.5)
- loss -(default-'categorical_crossentropy')
- optimizer -(default-'rmsprop')
- metrics - List of metrics to be evaluated by the model during training and testing (default-'accuracy')
  
## Segment CNN
The extracted segments and labels are fed into separate convolutional units and finally concatenated before the fixed length vector is fed to the dense layer that performs the classification.  Source code and related comments can be found [here](https://github.com/SamMahen/RelEx/blob/master/relex/RelEx_NN/cnn/segment_cnn.py).

Following snippet shows how each CNN unit is defined and concatenated at the end. 
```python
def define_model(self):
  input_shape = Input(shape=(self.data_model.maxlen,))
  embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                        weights=[self.embedding.embedding_matrix], trainable=False)(input_shape)
  conv = Conv1D(filters=self.filters, kernel_size=self.filter_conv, activation=self.activation)(embedding)
  drop = Dropout(self.drop_out)(conv)
  pool = MaxPooling1D(pool_size=self.filter_maxPool)(drop)
  flat = Flatten()(pool)

  return flat, input_shape

def build_segment_cnn(self, no_classes):
  flat1, input_shape1 = self.model_without_Label()
  flat2, input_shape2 = self.model_without_Label()
  flat3, input_shape3 = self.model_without_Label()
  flat4, input_shape4 = self.model_without_Label()
  flat5, input_shape5 = self.model_without_Label()

  # merge
  merged = concatenate([flat1, flat2, flat3, flat4, flat5])
  # interpretation
  dense1 = Dense(18, activation=self.activation)(merged)
  outputs = Dense(no_classes, activation=self.output_activation)(dense1)

  model = Model(inputs=[input_shape1, input_shape2, input_shape3, input_shape4, input_shape5], outputs=outputs)
  model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
  return model
```
## Single label Sentence CNN
Sentences and labels are fed into one convolutional unit. Each sentence has one label. If a sentence had more than one label, it is repreated. Source code and related comments can be found [here](https://github.com/SamMahen/RelEx/blob/master/relex/RelEx_NN/cnn/sentence_cnn.py).

Following snippet shows how the CNN model is defined for single label sentence CNN.
```python
#Define the model with different parameters and layers when running for multi label sentence CNN

input_shape = Input(shape=(self.data_model.maxlen,))
embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim)(input_shape)

if self.embedding:
    embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                          weights=[self.embedding.embedding_matrix], trainable=False)(input_shape)
conv1 = Conv1D(filters=self.filters, kernel_size=self.filter_conv, activation=self.activation)(embedding)
pool1 = MaxPooling1D(pool_size=self.filter_maxPool)(conv1)

conv2 = Conv1D(filters=self.filters, kernel_size=self.filter_conv, activation=self.activation)(pool1)
drop = Dropout(self.drop_out)(conv2)

flat = Flatten()(drop)
dense1 = Dense(self.filters, activation=self.activation)(flat)
outputs = Dense(no_classes, activation=self.output_activation)(dense1)

model = Model(inputs=input_shape, outputs=outputs)
model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
```

## Multi label Sentence CNN
Sentences and labels are fed into one convolutional unit but a sentence can have more than one label. Source code and related comments can be found [here](https://github.com/SamMahen/RelEx/blob/master/relex/RelEx_NN/cnn/sentence_cnn.py). To activate the multi label sentence CNN model, multi label flag should be enabled [here](https://github.com/SamMahen/RelEx/blob/master/relex/RelEx_NN/model/model.py).

Multi label model differs slightly from the single label in structure and the parameters as shown below.
```python
#Define the model with different parameters and layers when running for multi label sentence CNN
model = Sequential()
model.add(Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                    weights=[self.embedding.embedding_matrix], input_length=self.data_model.maxlen))
model.add(Dropout(self.drop_out))
model.add(Conv1D(self.filters, self.filter_conv, padding='valid', activation=self.activation, strides=1))
model.add(GlobalMaxPool1D())
model.add(Dense(len(self.data_model.encoder.classes_)))
model.add(Activation(self.output_activation))

model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
```
Following parameters should be tuned as follows for multi label senetnce CNN.
### Parameters: 
- filters - number of output filters (default-300)
- filter_conv - number of convolution filters (default-3)
- drop_out- regularized technique to reduce overfitting (default-0.1)
- loss -(default-'binary_crossentropy')
- optimizer -(default-'adam')
- metrics - List of metrics to be evaluated by the model during training and testing (default-'categorical_accuracy')
  
