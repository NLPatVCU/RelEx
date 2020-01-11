Words are turned into word vectors here. There are 2 options to apply word embeddings.
Train on the dataset itself
Use the dataset itself to learn the word embeddings. 
Using pre-trained word embeddings
When we don’t have enough data available to learn truly powerful features, we can load embedding vectors from a pre-computed embedding space.
External embeddings are read in and an embedding matrix is built which is loaded into an Embedding layer. It must be a matrix of shape (max_words, embedding_dim).
