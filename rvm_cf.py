import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

from google.cloud import storage

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x
  


class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states

def cors_enabled_function(request):

    request_json = request.get_json(silent=True)
    request_args = request.args

    # For more information about CORS and CORS preflight requests, see:
    # https://developer.mozilla.org/en-US/docs/Glossary/Preflight_request

    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return ('', 204, headers)

    else:

        if request_json and 'word' in request_json:
            word = request_json['word']
            length = request_json['length']
        elif request_args and 'word' in request_args:
            word = request_args('word')
            length = request_args('length')




        returnString = ""

        client = storage.Client()
        
        bucket = client.get_bucket('one_step')
        blob = bucket.blob('mw.h5')
        blob.download_to_filename('/tmp/mw.h5')
        
        model = MyModel(
        # Be sure the vocabulary size matches the `StringLookup` layers.
        vocab_size=57,
        embedding_dim=256,
        rnn_units=1024)
        
        
        
        
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam', loss=loss)
        
        
        vocab = vocab = ['\n', '\r', ' ', '$', '%', '&', "'", '*', '+', ',', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ';', '<', '=', '>', '@', 'A', 'E', 'M', 'N', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '×']

        ids_from_chars = preprocessing.StringLookup(
            vocabulary=list(vocab), mask_token=None)
        chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
        
        one_step_reloaded = OneStep(model, chars_from_ids, ids_from_chars)
        one_step_reloaded.compile(optimizer='adam', loss=loss)
        one_step_reloaded.generate_one_step(['a'], states=None)
        one_step_reloaded.built = True
        one_step_reloaded.load_weights('/tmp/mw.h5')

        states = None
        next_char = tf.constant([word])
        result = [next_char]

        for n in range(int(length)):
          next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
          result.append(next_char)

        returnString = tf.strings.join(result)[0].numpy().decode("utf-8")

        # Set CORS headers for the main request
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
            
        
        return (returnString, 200, headers)

