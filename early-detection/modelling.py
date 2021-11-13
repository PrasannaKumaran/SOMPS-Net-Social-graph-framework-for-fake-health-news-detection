import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from numpy import load
import tensorflow as tf
import keras.backend as K
from keras.layers import Flatten
from keras.engine.topology import Layer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import MultiHeadAttention
from keras import activations, initializers, constraints, regularizers
tqdm.pandas()

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def f1_score(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
	
    
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return f1

class MultiGraphCNN(Layer):
    def __init__(self,
                 output_dim,
                 num_filters,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform',
                 kernel_regularizer='l1',
                 bias_regularizer='l1',
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

def Model(paths):

    required_articles = pkl.load(open(f'{paths["reqd_articles_path"]}', 'rb'))
    publisher = pd.read_csv(f'{paths["publisher_path"]}', index_col=0)
    publisher = publisher[publisher['news_id'].isin(required_articles)]
    PX = publisher.drop(['news_id', 'target'], axis = 1)
    
    graph_conv_filters_tweet = load(f'{paths["graph_conv_tweet_path"]}')
    graph_conv_filters_retweet = load(f'{paths["graph_conv_retweet_path"]}')

    y = load(f'{paths["labels_path"]}')

    graph_tweet_data = load(f'{paths["graph_tweet_data_path"]}')
    graph_retweet_data = load(f'{paths["graph_retweet_data_path"]}')
                                    
    tweets_text = []
    for news in required_articles:
        article = load(f'/hdd/fakenews/Prasanna/FakeHealth /data/HS_tweet_texts_embedded/{news}.npy')
        tweets_text.append(article)
        
    retweets_text = []
    for news in required_articles:
        article = load(f'/hdd/fakenews/Prasanna/FakeHealth /data/HS_retweet_texts_embedded/{news}.npy')
        retweets_text.append(article)
                                    
    num_filters = 1
    source_retweet_length = np.array(retweets_text).shape[1]
    source_tweet_length = np.array(tweets_text).shape[1]
    number_of_feature_tweets = graph_tweet_data.shape[2]
    number_of_feature_retweets = graph_retweet_data.shape[2]
    retweet_user_size = graph_retweet_data.shape[1]
    tweet_user_size = graph_tweet_data.shape[1]
    GCN_output_dim = 16
    r_state = 64784
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(graph_tweet_data, y, test_size=0.15, random_state=r_state, stratify=y)
    X_train_rt, X_test_rt, y_train_t, y_test_t = train_test_split(graph_retweet_data, y, test_size=0.15, random_state=r_state, stratify=y)

    WX_train_t, WX_test_t, y_train_t, y_test_t = train_test_split(tweets_text, y, test_size=0.15, random_state=r_state, stratify=y)
    WX_train_rt, WX_test_rt, y_train_t, y_test_t = train_test_split(retweets_text, y, test_size=0.15, random_state=r_state, stratify=y)

    PX_train, PX_test, y_train_t, y_test_t = train_test_split(PX, y, test_size=0.15, random_state=r_state, stratify=y)

    MX_train_t, MX_test_t, y_train_t, y_test_t = train_test_split(graph_conv_filters_tweet, y, test_size=0.15, random_state=r_state, stratify=y)
    MX_train_rt, MX_test_rt, y_train_rt, y_test_t = train_test_split(graph_conv_filters_retweet, y, test_size=0.15, random_state=r_state, stratify=y)

    winput_tweet = tf.keras.layers.Input(shape=(source_tweet_length,100))
    wembed_tweet = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True))(winput_tweet)
    
    winput_retweet = tf.keras.layers.Input(shape=(source_retweet_length,100))
    wembed_retweet = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True))(winput_retweet)
                                    
    data_input = tf.keras.layers.Input(shape=(710,))
    p = tf.keras.layers.Dense(64, activation="relu")(data_input)
    p = tf.keras.layers.Flatten()(p)
                                    
    rmain_input_tweet = tf.keras.layers.Input(shape=(tweet_user_size, number_of_feature_tweets))
    rmain_input_retweet = tf.keras.layers.Input(shape=(retweet_user_size, number_of_feature_retweets))
                                    
    graph_conv_filters_input_tweet = tf.keras.layers.Input(shape=(tweet_user_size, tweet_user_size))
    gmain_input_tweet = MultiGraphCNN(GCN_output_dim, num_filters)([rmain_input_tweet, graph_conv_filters_input_tweet])
    gmain_input_tweet = MultiGraphCNN(GCN_output_dim, num_filters)([gmain_input_tweet, graph_conv_filters_input_tweet])
    gmain_input_tweet = MultiGraphCNN(GCN_output_dim, num_filters)([gmain_input_tweet, graph_conv_filters_input_tweet])


    graph_conv_filters_input_retweet = tf.keras.layers.Input(shape=(retweet_user_size, retweet_user_size))
    gmain_input_retweet = MultiGraphCNN(GCN_output_dim, num_filters)([rmain_input_retweet, graph_conv_filters_input_retweet])
    gmain_input_retweet = MultiGraphCNN(GCN_output_dim, num_filters)([gmain_input_retweet, graph_conv_filters_input_retweet])
    gmain_input_retweet = MultiGraphCNN(GCN_output_dim, num_filters)([gmain_input_retweet, graph_conv_filters_input_retweet])
                                    
    layer_tweet = MultiHeadAttention(num_heads=16, key_dim=4, value_dim=12, dropout=0.5, attention_axes=(1,2), kernel_initializer='glorot_uniform',
                 bias_initializer=tf.keras.initializers.HeUniform(),
                 )
    output_tensor_tweet, weights = layer_tweet(wembed_tweet, gmain_input_tweet, 
                                   return_attention_scores=True)

    layer_retweet = MultiHeadAttention(num_heads=16, key_dim=4, value_dim=12, dropout=0.5, attention_axes=(1,2), kernel_initializer='glorot_uniform',
                     bias_initializer=tf.keras.initializers.HeUniform(),
                     )
    output_tensor_retweet, weights = layer_retweet(wembed_retweet, gmain_input_retweet, 
                               return_attention_scores=True)
    
    output_tensor = tf.keras.layers.Concatenate()([output_tensor_tweet, output_tensor_retweet])
    output_tensor = Flatten()(output_tensor)
    output_tensor = tf.keras.layers.Concatenate()([output_tensor, p])

    x = tf.keras.layers.Dense(1024, activation="tanh")(output_tensor)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation="tanh")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="tanh")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128,activation="tanh")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64,activation="tanh")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(32,activation="tanh")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(10,activation="tanh")(x)

    prediction = tf.keras.layers.Dense(1,activation="sigmoid")(x)
    model = tf.keras.Model([
        winput_tweet,
        winput_retweet,
        data_input,
        rmain_input_tweet,
        rmain_input_retweet,
        graph_conv_filters_input_tweet,
        graph_conv_filters_input_retweet,
    ], prediction)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5, verbose=2)
    model.compile(optimizer='sgd' ,loss="binary_crossentropy", metrics=['accuracy', f1_score])

    history=model.fit([
    np.array(WX_train_t),
    np.array(WX_train_rt),
    np.array(PX_train),
    np.array(X_train_t),
    np.array(X_train_rt),
    np.array(MX_train_t),
    np.array(MX_train_rt),
    ]
    ,np.array(y_train_t),epochs=10, validation_split=0.2, callbacks=[early_stopping])
   
    scores=model.evaluate([
    np.array(WX_test_t),
    np.array(WX_test_rt),
    np.array(PX_test),
    np.array(X_test_t),
    np.array(X_test_rt),
    np.array(MX_test_t),
    np.array(MX_test_rt)],
    np.array(y_test_t), verbose=1)
                                    
    return scores

model_scores = []
for i in tqdm([4, 8, 12, 16, 20, 24]):
    current_dir = f'./data/{i}_Hours'
    required_articles = f'./data/{i}h_required_news_articles.pkl'
    publisher_path = '../data/publisher_news.csv'
    graph_conv_tweet_path = f'./data/{i}_hours/graph_tweet_data_network_similarity.npy'
    graph_conv_retweet_path = f'./data/{i}_hours/graph_retweet_data_network_similarity.npy'
    graph_tweet_data_path = f'./data/{i}_hours/graph_tweet_data_network_similarity.npy'
    graph_retweet_data_path = f'./data/{i}_hours/graph_retweet_data_network_similarity.npy'
    labels_path = f'./data/HealthStory_{i}_hours/labels.npy'
    
    paths = {
        'reqd_articles_path':required_articles,
        'publisher_path':publisher_path,
        'graph_conv_tweet_path':graph_conv_tweet_path,
        'graph_conv_retweet_path':graph_conv_retweet_path,
        'graph_tweet_data_path':graph_tweet_data_path,
        'graph_retweet_data_path':graph_retweet_data_path,
        'labels_path':labels_path
    }
    scores = Model(paths)
    model_scores.append(
        {
            "Hours" : i,
            "Accuracy": scores[1],
            "F1": scores[2]
        })

results = pd.DataFrame(model_scores)
results.to_csv('early_detection_results.csv')