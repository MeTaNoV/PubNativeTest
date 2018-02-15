# PubNative ML Test

## Introduction

To solve the test proposed, I will use:
- python 3.6
- virtualenv
- Jupyter notebook
- numpy / pandas / matplotlib / seaborn
- TensorFlow in its last version (1.5)
- Google Cloud Dataprep
- flask

## Setup

Clone the following repository: https://github.com/MeTaNoV/PubNativeTest.git

```
cd /some/path/PubNativeTest
``` 

Setup the environment by using virtualenv:

```
virtualenv env
```

Then activate the environment:

```
source env/bin/activate
```

And install the required package:

```
pip install jupyter numpy pandas matplotlib seaborn tensorflow flask
```

## Approach

### Main Test

The first task that we will perform is data analysis and cleaning that will be performed with Google Cloud Dataprep. Dataprep is a powerful tool that will enable us to visualize and perform data transformation with the same interface. Moreover, the data transformation could be reused across dataset easily and are performed on Google Cloud Dataflow (Apache Beam). From this data analysis, we will be able to derive a first set of features to be used in our Machine Learning model.

The second task will consist in different experiment that we will run in a Jupyter notebook. To run those experiments, we will use Tensoflow, and especially the following API:
- The Dataset API (tf.data.*) enable us to read our data and prepare them to be fed inside the model. It is also possible to shuffle, batch, etc... in this phase. This will be implemented inside our `input_fn` that basically will return a batch of features associated with its batch of labels
- The Feature Column API (tf.feature_column.*) enable us to perform our feature engineering. Indeed, we will be able to choose between numercial, categorical or bucketized value for our different feature columns to be used in our model.
- The Estimator API (tf.estimator.*) enables us to avoid writing our model from scratch, and instead use some canned estimators to be able to perform our classification. We will first try a `LinearClassifier` then probably a `DNNClassifier` if the first was not performing well and eventually a `DNNLinearCombinedClassifier` to try to improve our predictions. It is worth to note that each of those estimators could be configured, for example using a specific Optimizer. They provide us with the `train`/`evaluate`/`predict` methods which will be used to perform our training, evaluation and prediction. Lastly, we are able to export our model after being trained for serving purpose.

### Bonus and Mega Bonus Point:

Tensorflow provides the ability with its Estimator to save automatically a model every time a training occurs.
Therefore we could use this feature to create a flask server that will provide one end point for the training and another for the prediction.
This solution is simple and fast to set up but will not scale since it lives on one server only.
To make this solution scale, we would need to decouple the training from the prediction and use some intermediate storage bucket for the model. This could also enable A/B testing when releasing a new version of the model.
Other solutions exist, namely, GCP provides a way to train and serve a Tensorflow model in a no-ops manner thanks to its ML Engine service. 

## Data Analysis and Cleaning

As mentioned before, I used Cloud Dataprep (on GCP) to perform a quick analysis of the data and create a flow to clean the data accordingly.

What we are looking for in a first step, is:
- to look for missing or NA values, 
- to see if traing and validation have some similar distribution
- to see if the data needs to be transformed
- and therefore derive the type of feature column that we will be using

Let's have a closer look at the different columns now:

- v1: 
  - 2 categories `a` and `b` and 39 NA
  - 39 NA could be filtered out
  - training and validation dataset distributions seem similar
  - good categorical feature candidate

- v2: 
  - numerical values betwwen roughly 10 and 80
  - 39 NA could be filtered out
  - training and validation dataset distributions seem similar
  - replace `,` with `.`
  - could be standardized with mean=32.88 and stddev=12.49
  - good numerical feature candidate

- v3:
  - numerical values betwwen roughly 0 and 0.003
  - no NA
  - training and validation dataset distributions seem similar
  - replace `,` with `.`
  - could be standardized with mean=0.0005951 and stddev=0.0005422
  - good numerical feature candidate

- v4: 
  - 3 categories `u`, `y` and `l`
  - 64 NA => NA could be filtered out
  - training and validation dataset distributions seem similar
  - good categorical feature candidate

- v5:
  - same as v4 with `u=g`, `y=p` and `l=gg`
  - we will drop this column

- v6: 
  - 14 categories `c`, `q`, `W`, `cc`, `x`, `aa`, `i`, `m`, `k`, `e`, `ff`, `d`, `j`, `r`
  - 66 NA => NA could be filtered out
  - training and validation share the same category but distribution is a bit different (we will see that this is plausible since the labels don't share the same distribution either)
  - good categorical feature candidate

- v7: 
  - 9 categories `v`, `h`, `bb`, `ff`, `z`, `j`, `n`, `dd`, `o`
  - 66 NA => NA could be filtered out
  - training and validation share the same categories and distribution
  - good categorical feature candidate

- v8: 
  - numerical values between roughly 0 and 30
  - no NA
  - training and validation dataset distributions seem similar
  - replace `,` with `.`
  - (we could have filtered values > 20, we'll see if necessary later)
  - could be standardized with mean=3.49 and stddev=4.38
  - good numerical feature candidate

- v9: 
  - 2 categories `f` and `t`
  - no NA
  - training and validation share the same category but distribution is a bit different (we will see that this is plausible since the labels don't share the same distribution either)
  - good categorical feature candidate
  
- v10: 
  - 2 categories `f` and `t`
  - no NA
  - training and validation dataset distributions seem similar
  - good categorical feature candidate

- v11: 
  - numerical value
  - no NA
  - considering the given distribution, we might want to bucketize this column in 20+1 different buckets for ranges of length 1 between [0,20[ and the remaining
  - good bucketized feature candidate

- v12:
  - 2 categories `f` and `t`
  - no NA
  - training and validation dataset distributions seem similar
  - good categorical feature candidate
  
- v13: 
  - 3 categories `q`, `s` and `p`
  - given the distribution, we will ignore this feature in a first step

- v14:
  - numerical
  - 25% are 0, we could create 10+1 buckets for ranges of length 50 between [0,500[ and the remaining
  - good bucketized feature candidate

- v15:
  - numerical
  - 40% are 0, we could create 10+1 buckets for ranges of length 500 between [0,5000[ and the remaining
  - good bucketized feature candidate

- v17: 
  - multiple of v14 by a factor of 10000
  - we will drop this column

- v18: 
  - Most of the values are NA
  - we will drop this column

- v19: 
  - is the same as our classLabel in the training dataset and totally different on the validation dataset
  - we will drop this column

- classLabel: 
  - our training dataset is unbalanced wrt our class label, but the test dataset is balanced so we can use the accuracy to assess our predictions. If it was not the case, we would use precision and recall in addition to the accuracy.
  - we have to be careful and be sure that our training does not overfit the `yes` label. If it is the case, a good idea would be to merge both datasets and recreate some new training and validation dataset out of it.

After this cleaning phase, we reduced our dataset from:
- 3700 rows to 3522 rows for the training dataset
- 200 rows to 191 rows for the validation dataset.

It  seems pretty reasonable.

The cleaned data that we will use in the experiment are located in the `data` folder with the following name:
- `training_cleaned.csv`
- `validation_cleaned.csv`

## Notebook Kernel

We will now perform our experiments with Jupyter notebook. Simply execute the followiing command:

```
jupyter notebook
```

We will perform first a deeper exploration of the data thanks to the `Deeper_Data_Exploration.ipynb` notebook.
This task gave us more intuition on the effect of each features on the `classLabel` to predict.
Namely, we can clearly see that `v9` has the biggest correlation with it. And looking closer at this relation, if we decide to use only `v9` as the prediction, we would obtain 92% accuracy on the training set and 86% on the training dataset which give us a good baseline to achieve in terms of efficiency.
I also discovered that `v10` and `v11` are closely related since `v10 = (v11 != 0)`, but we will keep both since `v11` contains a more detailed information that might be useful.
Lastly, from the correlation heatmap, I decided to remove some features, namely `v4`, `v6` and `v14`.

Then open the notebook named `PN_experiment.ipynb` and follow the comments/instruction from there.

## Serving the model

Now that we finalized our model, we need to be able to serve it. We will use a simple flask application to perform this task.

The code for the server application can be found in `ml_server.py`.

It basically creates a server with two endpoints. I slightly adapted the code from the notebook to take as input some JSON data received via the endpoints, and feed it to the model for training or prediction.

To launch the server, simply execute the following:

```bash
./run_server.sh
```

The server could be quickly tested by running the prediction or training scripts in another console:

```bash
./run_prediction.sh
```

and 

```bash
./run_training.sh
```

Note that the data used as input are already pre-processed data. It would require a bit more work to integrate the transformation pipeline on the server.
