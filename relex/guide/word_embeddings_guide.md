# Word embeddings

The word embeddings map a set of words or phrases in a vocabulary to real-valued vectors. You can find source code and related comments [here](https://github.com/SamMahen/RelEx/blob/master/relex/RelEx_NN/embeddings/embeddings.py). We use the dataset itself to learn the word embeddings or instead of learning word embeddings jointly with the problem we want to solve, we can load embedding vectors from a pre-computed embedding space. When we don’t have enough data available to learn truly powerful features, we can load embedding vectors from a pre-computed embedding space.

Both ways are handled in our package. For example you can see both ways are handled with a flag [here](https://github.com/SamMahen/RelEx/blob/master/relex/RelEx_NN/cnn/sentence_cnn.py).

```python
 embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim)(input_shape)

  if self.embedding:
      embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim, weights=[self.embedding.embedding_matrix], trainable=False)(input_shape)
```
External embeddings are read in and an embedding matrix is built which is loaded into an Embedding layer. It must be a matrix of the shape (max_words, embedding_dim).
How pre-trained word embeddings are read in and loaded into an embedding matrix is shown [here](https://github.com/SamMahen/RelEx/blob/master/relex/RelEx_NN/embeddings/embeddings.py).
