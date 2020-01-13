# Neural network models

We used 2 simple Neural Network models here. Models differ based on the shape of the tensor that is fed into the models.
1. Data padded to the same length, and turned into an integer tensor of shape (samples, word_indices).
2. Data one-hot-encoded and turned into vectors of 0s and 1s.

Following parameters and flags can be set / tuned. Default values are mentioned.
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

Note that, source code and related comments can be found [here](https://github.com/SamMahen/RelEx/blob/master/relex/RelEx_NN/nn/simple_NN.py).

