import sys
import traceback

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

app = Flask(__name__, static_url_path='')

# input path
path_data = 'data'
path_model = 'model'

# feature definitions
feature_names = ['v1','v2','v3','v7','v8','v9','v10','v11','v12','v15']
label_name = ['classLabel']

feature_columns = [
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key="v1", 
            vocabulary_list=["a", "b"])
    ),
    tf.feature_column.numeric_column('v2'),
    tf.feature_column.numeric_column('v3'),
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key="v7", 
            vocabulary_list=["v", "h", "bb", "ff", "z", "j", "n", "dd", "o"])
    ),
    tf.feature_column.numeric_column('v8'),
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key="v9", 
            vocabulary_list=["f", "t"])
    ),
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key="v10", 
            vocabulary_list=["f", "t"])
    ),
    tf.feature_column.indicator_column(
        tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column('v11'),
            list(np.linspace(1.,20.,20)))
    ),
    tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key="v12", 
            vocabulary_list=["f", "t"])
    ),
    tf.feature_column.indicator_column(
        tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column('v15'),
            list(np.linspace(500.,5000.,10)))
    ),
]

# model instantiation, it will restore a previous version of any saved model
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[512, 256, 128],
#     dropout=0.5,
#     optimizer=tf.train.ProximalAdagradOptimizer(
#       learning_rate=0.1,
#       l1_regularization_strength=0.001
#     ),
    label_vocabulary=["yes.", "no."],
    model_dir=path_model)

# Online training is done with a batch of 1 sent via json at the train endpoint
def input_fn_train(input):
    def split_features_label(dinput):
        label = dinput['classLabel']
        del dinput['classLabel']
        features = dinput
        return features, label

    dataset = tf.data.Dataset.from_tensor_slices(dict(input))
    dataset = dataset.map(split_features_label)
    dataset = dataset.batch(1)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    
    return batch_features, batch_labels

# Training endpoint, we will return the step value
@app.route('/train', methods=['POST'])
def train():
    try:
        input = pd.DataFrame([request.json])
        input = input.reindex(columns=feature_names+label_name, fill_value=0)

        classifier.train(input_fn=lambda: input_fn_train(input))

        step = classifier.get_variable_value('global_step')

        return jsonify({'training_step': int(step)})

    except Exception as e:

        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

# Similarly, we have our prediction
def input_fn_predict(input):
    dataset = tf.data.Dataset.from_tensor_slices(dict(input))
    dataset = dataset.batch(1)

    iterator = dataset.make_one_shot_iterator()
    batch_features = iterator.get_next()
    
    return batch_features

# and associated endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input = pd.DataFrame([request.json])
        input = input.reindex(columns=feature_names, fill_value=0)

        predictions = list(classifier.predict(input_fn=lambda: input_fn_predict(input)))
        result = predictions[0]['classes'][0].decode("utf-8")

        return jsonify({'classLabel': result})

    except Exception as e:

        return jsonify({'error': str(e), 'trace': traceback.format_exc()})
