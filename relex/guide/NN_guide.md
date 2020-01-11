
Simple Neural Network is used here. There are 2 models based on the shape of the tensor of the converted data. 
Data(lists) should be padded to the same length, and turned into an integer tensor of shape (samples, word_indices).
Data lists can be one-hot-encoded and turned into vectors of 0s and 1s.
Following parameters and flags can be set / tuned. Default values are mentioned.
Flags:
embedding -> False (default) - trained on the dataset; True - use pre-trained word embeddings
Cross_validation -> False (default)

Parameters:
epochs=20, batch_size=512, filters=32, filter_conv=1, filter_maxPool=5, activation='relu', output_activation='sigmoid',drop_out=0.5, loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy']
