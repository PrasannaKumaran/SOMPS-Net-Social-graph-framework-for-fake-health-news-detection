import random
import numpy as np
import pickle as pkl
from numpy import load
import tensorflow as tf
import keras.backend as K
from keras.engine.topology import Layer
from keras.preprocessing.text import Tokenizer
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import MultiHeadAttention
from keras import activations, initializers, constraints, regularizers

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

class MultiGraphCNN(Layer):

    def __init__(self,
                 output_dim,
                 num_filters,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MultiGraphCNN, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.num_filters = num_filters
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_initializer.__name__ = kernel_initializer
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):

        if self.num_filters != int(input_shape[1][-2]/input_shape[1][-1]):
            raise ValueError('num_filters does not match with graph_conv_filters dimensions.')

        self.input_dim = input_shape[0][-1]
        kernel_shape = (self.num_filters * self.input_dim, self.output_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):

        output = graph_conv_op(inputs[0], self.num_filters, inputs[1], self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][1], self.output_dim)
        return output_shape

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'num_filters': self.num_filters,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)  
        }
        base_config = super(MultiGraphCNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
def normalize_adj_numpy(adj, symmetric=True):
    if symmetric:
        d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d)
    else:
        d = np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj)
    return a_norm

def preprocess_adj_tensor(adj_tensor, symmetric=True):
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)        
        adj_out_tensor.append(adj)
    adj_out_tensor = np.array(adj_out_tensor)
    return adj_out_tensor

def graph_conv_op(x, num_filters, graph_conv_filters, kernel):

    if len(x.get_shape()) == 2:
        conv_op = K.dot(graph_conv_filters, x)
        conv_op = tf.split(conv_op, num_filters, axis=0)
        conv_op = K.concatenate(conv_op, axis=1)
    elif len(x.get_shape()) == 3:
        conv_op = K.batch_dot(graph_conv_filters, x)
        conv_op = tf.split(conv_op, num_filters, axis=1)
        conv_op = K.concatenate(conv_op, axis=2)
    else:
        raise ValueError('x must be either 2 or 3 dimension tensor'
                         'Got input shape: ' + str(x.get_shape()))

    conv_out = K.dot(conv_op, kernel)
    return conv_out

HP_word_embeddings_path = hp.HParam(
    'word_embeddings',
    hp.Discrete([
        './embeds/glove.6B.100d.txt',
        './embeds/glove.twitter.6B.100d.txt'
        ])
    )

HP_test_size_main = hp.HParam(
    'testSize_main',
    hp.Discrete([0.2, 0.25, 0.3])
    )

HP_test_size_validation = hp.HParam(
    'validation_split',
    hp.Discrete([0.2, 0.4, 0.6, 0.8])
    )

HP_num_heads = hp.HParam(
    'num_heads',
    hp.Discrete([4, 8, 12, 16])
    )

HP_key_dimension = hp.HParam(
    'key_dimension',
    hp.Discrete([4, 8, 12, 16])
    )

HP_value_dimension = hp.HParam(
    'value_dimension',
    hp.Discrete([4, 8, 12, 16])
    )

HP_OPTIMIZER = hp.HParam(
    'optimizer',
    hp.Discrete(['adam', 'sgd']))
    
METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('final_logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[
      HP_word_embeddings_path,
      HP_test_size_main,
      HP_test_size_validation,
      HP_num_heads,
      HP_key_dimension,
      HP_value_dimension,
      HP_OPTIMIZER
      ],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='accuracy')],
  )

y = load('hs_news_labels.npy')
graph_tweet_data = load('./tweet_data/graph_tweet_data.npy')
graph_retweet_data = load('./retweet_data/graph_retweet_data.npy')

f = open("./tweet_data/tweet_texts_required.pkl", "rb")
tweet_texts = pkl.load(f)

num_filters = 1
source_retweet_length=16
number_of_feature=14
retweet_user_size = 16
tweet_user_size = 32

def train_test_model(hparams):
    graph_conv_filters_tweet = load('./tweet_data/graph_tweet_data_network_similarity.npy')
    graph_conv_filters_retweet = load('./retweet_data/graph_retweet_data_network_similarity.npy')

    tok = Tokenizer()
    tok.fit_on_texts(tweet_texts)
    vocab_size = len(tok.word_index) + 1
    encoded_docs = tok.texts_to_sequences(tweet_texts)

    max_length = 16
    tweet_padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    
    embeddings_index = dict()
    f = open(hparams[HP_word_embeddings_path])
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in tok.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    r_state = random.randint(0,84894)
    
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(graph_tweet_data, y, test_size=hparams[HP_test_size_main], random_state=r_state, stratify=y)
    X_train_rt, X_test_rt, y_train_t, y_test_t = train_test_split(graph_retweet_data, y, test_size=hparams[HP_test_size_main], random_state=r_state, stratify=y)
    WX_train_rt, WX_test_rt, y_train_t, y_test_t = train_test_split(tweet_padded_docs, y, test_size=hparams[HP_test_size_main], random_state=r_state, stratify=y)

    MX_train_t, MX_test_t, y_train_t, y_test_t = train_test_split(graph_conv_filters_tweet, y, test_size=hparams[HP_test_size_main], random_state=r_state, stratify=y)
    MX_train_rt, MX_test_rt, y_train_rt, y_test_tt = train_test_split(graph_conv_filters_retweet, y, test_size=hparams[HP_test_size_main], random_state=r_state, stratify=y)


    winput = tf.keras.layers.Input(shape=(source_retweet_length,))
    wembed = tf.keras.layers.Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=16, trainable=False)(winput)
    wembed = tf.keras.layers.GRU(100, return_sequences=True)(wembed)

    rmain_input_tweet = tf.keras.layers.Input(shape=(tweet_user_size, number_of_feature))
    rmain_input_retweet = tf.keras.layers.Input(shape=(retweet_user_size, number_of_feature))

    graph_conv_filters_input_tweet = tf.keras.layers.Input(shape=(tweet_user_size, tweet_user_size))
    gmain_input_tweet = MultiGraphCNN(hparams[HP_key_dimension], num_filters)([rmain_input_tweet, graph_conv_filters_input_tweet])

    graph_conv_filters_input_retweet = tf.keras.layers.Input(shape=(retweet_user_size, retweet_user_size))
    gmain_input_retweet = MultiGraphCNN(hparams[HP_key_dimension], num_filters)([rmain_input_retweet, graph_conv_filters_input_retweet])
    
    layer_tweet = MultiHeadAttention(
        num_heads=hparams[HP_num_heads],
        key_dim=hparams[HP_key_dimension],
        value_dim=hparams[HP_value_dimension],
        dropout=0.5,
        attention_axes=(1,2),
        kernel_initializer=tf.keras.initializers.HeUniform(),
        bias_initializer=tf.keras.initializers.HeUniform(),
    )
    output_tensor_tweet, weights = layer_tweet(
        wembed,
        gmain_input_tweet,
        return_attention_scores=True
    )

    layer_retweet = MultiHeadAttention(
        num_heads=hparams[HP_num_heads],
        key_dim=hparams[HP_key_dimension],
        value_dim=hparams[HP_value_dimension],
        dropout=0.5,
        attention_axes=(1,2),
        kernel_initializer=tf.keras.initializers.HeUniform(),
        bias_initializer=tf.keras.initializers.HeUniform(),
        )
    output_tensor_retweet, weights = layer_retweet(
        wembed,
        gmain_input_retweet,
        return_attention_scores=True
    )

    output_tensor = tf.keras.layers.Concatenate()([
        output_tensor_tweet,
        output_tensor_retweet]
    )
    x = tf.keras.layers.Dense(64, activation="relu")(output_tensor)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(32,activation="relu")(x)
    prediction = tf.keras.layers.Dense(1,activation="sigmoid")(x)
    model = tf.keras.Model([
    winput,
    rmain_input_tweet,
    rmain_input_retweet,
    graph_conv_filters_input_tweet,
    graph_conv_filters_input_retweet], prediction)

    Adam= tf.keras.optimizers.Adam(lr=0.001)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5, verbose=2)
    model.compile(optimizer=hparams[HP_OPTIMIZER] ,loss="binary_crossentropy", metrics=['accuracy'])

    history=model.fit([
    np.array(WX_train_rt),
    np.array(X_train_t),
    np.array(X_train_rt),
    np.array(MX_train_t),
    np.array(MX_train_rt)
    ]
    ,np.array(y_train_t),
    epochs=13,
    validation_split=hparams[HP_test_size_validation],
    callbacks=[early_stopping]
    )
    
    _, accuracy = model.evaluate([
    np.array(WX_test_rt),
    np.array(X_test_t),
    np.array(X_test_rt),
    np.array(MX_test_t),
    np.array(MX_test_rt)],
    np.array(y_test_t), verbose=1)
    
    return accuracy

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0

for we in HP_word_embeddings_path.domain.values:
  for tm in HP_test_size_main.domain.values:
    for tsec in HP_test_size_validation.domain.values:
      for nh in HP_num_heads.domain.values:
        for kd in HP_key_dimension.domain.values:
          for vd in HP_value_dimension.domain.values:
            for opt in HP_OPTIMIZER.domain.values:
              hparams = {
                HP_word_embeddings_path: we,
                HP_test_size_main: tm,
                HP_test_size_validation: tsec,
                HP_num_heads: nh,
                HP_key_dimension: kd,
                HP_value_dimension: vd,
                HP_OPTIMIZER: opt,
                }
              run_name = "run-%d" % session_num
              print('--- Starting trial: %s' % run_name)
              print({h.name: hparams[h] for h in hparams})
              run('final_logs/hparam_tuning/' + run_name, hparams)
              session_num += 1