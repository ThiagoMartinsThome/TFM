from google.cloud import bigquery
# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import TimeDistributed, RepeatVector
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization
from numpy import mean
from numpy import absolute
from numpy import loadtxt
from sklearn.model_selection import cross_val_score,  cross_validate
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_union
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.pipeline import FeatureUnion
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, Normalizer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras import regularizers
import joblib
# %matplotlib inline
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
# example of power transform input and output variables for regression.

df = pd.read_csv('data/data_20day_chg_target.csv', parse_dates=['date'])

# load data
data = df.copy()
data.drop('long_name', axis=1, inplace=True)
data.dropna(inplace=True, axis=0)
data.sort_values(by=['ticker', 'date'], ascending=True, inplace=True)
data.set_index('date', inplace=True)
#data.head()

print('Null values: ', data.isnull().sum().sum())

# Define features and target
X, y = data.drop('short_result', axis=1), data['short_result']
train = data[data['year'] <= 2010]
val = data[(data['year'] > 2010) & (data['year'] <= 2015)]
test = data[data['year'] > 2015]

X_train, y_train = train.drop('short_result', axis=1), train['short_result']
X_test, y_test = test.drop('short_result', axis=1), test['short_result']
X_val, y_val = val.drop('short_result', axis=1), val['short_result']


#Pipeline
#Column transformation
# determine categorical and numerical features
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

# Column preparation
transformer = [('onehot', OneHotEncoder(sparse=False), categorical_cols),
               ('num', StandardScaler(), numerical_cols),]
col_transform = ColumnTransformer(transformers=transformer)

pipeline_x = Pipeline(steps=[('prep', col_transform),
                             ('pca', PCA(n_components=34, random_state=1)),
                             #('pca', PCA(.98, random_state=1))
                             #('scaler2', StandardScaler()),
                             ])

pipeline_x.fit(X)

joblib.dump(pipeline_x, 'model/pipeline_x.pkl', compress=1)

with open('model/pipeline_x.pkl',  'rb') as f:
    pipeline_x_loaded = joblib.load(f)

X_train_scaled, y_train_scaled = pipeline_x_loaded.transform(X_train), y_train.values.reshape((-1, 1))
X_val_scaled, y_val_scaled = pipeline_x_loaded.transform(X_val), y_val.values.reshape((-1, 1))

#data_scaled = pd.DataFrame(X_train_scaled, columns=range(0,X_train_scaled.shape[1]), index=range(0,X_train_scaled.shape[0]))

X_train_scaled.astype('float32')
y_train_scaled.astype('float32')

X_val_scaled.astype('float32')
y_val_scaled.astype('float32')

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# choose a number of time steps
n_steps = 20
# Concat features and target
train_dataset = np.hstack((X_train_scaled, y_train_scaled))
val_dataset = np.hstack((X_val_scaled, y_val_scaled))
# convert into input/output
X_train_split, y_train_split = split_sequences(train_dataset, n_steps)
y_train_split = np.reshape(y_train_split, (-1, 1))

X_val_split, y_val_split = split_sequences(val_dataset, n_steps)
y_val_split = np.reshape(y_val_split, (-1, 1))
# the dataset knows the number of features, e.g. 2
n_features = X_train_split.shape[2]


model = Sequential()
model.add(Bidirectional(LSTM(50,
                             activation='relu',
                             kernel_regularizer='l2',
                             dropout=0.1,
                             recurrent_dropout=0.1,
                             go_backwards=True,
                             return_sequences=True),
                        input_shape=(n_steps, n_features)))
model.add(LSTM(units=25,
               dropout=0.2,
               go_backwards=True,
               return_sequences=True))
model.add(LSTM(units=25,
               dropout=0.2,
               go_backwards=True,
               return_sequences=True))
model.add(LSTM(units=25,
               dropout=0.2,
               go_backwards=True,
               return_sequences=True))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

"""# Simple model
model = Sequential()
model.add(Bidirectional(LSTM(150, activation='relu', return_sequences=True), input_shape=(n_steps, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=30))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])"""

#Model path
checkpoint_path = 'model/Blstm_model.h5'

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              save_weights_only=False,
                              verbose=1, save_best_only=True,
                              monitor='loss')
# Create a callback that prevents overfitting
early_stopping = EarlyStopping(monitor='loss', mode='auto', verbose=2, patience=10)

# Create a call back to improve the learning of the model
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=2, min_lr=0.00001, mode='min')

# fit model

history = model.fit(X_train_split, y_train_split, epochs=20, batch_size=1024,
                    callbacks=[cp_callback, early_stopping, reduce_lr],
                    validation_data=(X_val_split, y_val_split), shuffle=True)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

X_test_scaled = pipeline_x.transform(X_test)
y_test_scaled = y_test.values.reshape((-1, 1))

X_test_scaled.astype('float32')
y_test_scaled.astype('float32')

# Concat features and target
test_dataset = np.hstack((X_test_scaled, y_test_scaled))
# convert into input/output
X_test_split, y_test_split = split_sequences(test_dataset, n_steps)
y_test_split = np.reshape(y_test_split, (-1, 1))
# the dataset knows the number of features, e.g. 2
n_features = X_test_split.shape[2]

# Load model
model = load_model('model/Blstm_model.h5')
yhat = model.predict(X_test_split)

evaluation = model.evaluate(X_test_split, y_test_split)
print('Model evaluation: ', evaluation)

df_predicted = pd.DataFrame()
df_predicted['short_result'] = y_test.iloc[n_steps -1:]
df_predicted['predictions'] = yhat
print(df_predicted.head(20))

plt.plot(df_predicted.iloc[-1000:])
plt.show()

print(df_predicted.describe().T)


# Split into sequences
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# choose a number of time steps
n_steps_split = 30 #60
# Concat features and target
train_dataset = np.hstack((X_train_scaled, y_train_scaled))
val_dataset = np.hstack((X_val_scaled, y_val_scaled))
# convert into input/output
X_train_split, y_train_split = split_sequences(train_dataset, n_steps)
y_train_split = np.reshape(y_train_split, (-1, 1))
X_val_split, y_val_split = split_sequences(val_dataset, n_steps)
y_val_split = np.reshape(y_val_split, (-1, 1))

# the dataset knows the number of features, e.g. 2
n_features = X_train_split.shape[2]
n_seq = 5
n_steps = 6
#We can define the ConvLSTM as a single layer in terms of the number of filters
#and a two-dimensional kernel size in terms of (rows, columns).
#As we are working with a one-dimensional series, the number of rows is always
#fixed to 1 in the kernel.
rows = 1

#X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_split, y_train_split, test_size=0.3, shuffle=False)

X_train_split = X_train_split.reshape((X_train_split.shape[0], n_seq, rows, n_steps, n_features))
X_val_split = X_val_split.reshape((X_val_split.shape[0], n_seq, rows, n_steps, n_features))
print(X_train_split.shape)

# define model
"""model = Sequential()
model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), activation='relu', input_shape=(n_seq, rows, n_steps, n_features),
                     padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                     padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                     padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(1))"""

"""model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_seq, rows, n_steps, n_features)))
model.add(Flatten())
model.add(RepeatVector(1))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(20, activation='relu')))
model.add(TimeDistributed(Dense(1)))"""

model = Sequential()
model.add(ConvLSTM2D(filters=90, kernel_size=(5, 6), 
                     input_shape=(n_seq, rows, n_steps, n_features),
                     padding='same', 
                     return_sequences=True))
model.add(BatchNormalization())
model.add(Flatten())
model.add(RepeatVector(1))
model.add(Bidirectional(LSTM(50, return_sequences=True)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(TimeDistributed(Dense(1)))
#model.add(Dense(units=1))
model.add(Activation('relu'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# print(model.summary())

# Model path
checkpoint_path = 'ConvLSTM2D_model.h5'

## Callbacks:
# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              save_weights_only=False,
                              verbose=1, save_best_only=True,
                              monitor='loss')
# Create a callback that prevents the overfitting
early_stopping = EarlyStopping(monitor='loss', mode='auto', verbose=2,
                               patience=10)
# Create a callback to increase the learning rate
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=1, min_lr=1e-5, mode='min')
# model.summary()

# fit model

history = model.fit(X_train_split,y_train_split,epochs=200,batch_size=1024,
                    callbacks=[cp_callback, early_stopping, reduce_lr],
                    validation_data=(X_val_split, y_val_split), shuffle=False)

# Concat features and target
test_dataset = np.hstack((X_test_scaled, y_test_scaled))
# convert into input/output
X_test_split, y_test_split = split_sequences(test_dataset, n_steps_split)
y_test_split = np.reshape(y_test_split, (-1, 1))
# the dataset knows the number of features, e.g. 2
n_features = X_test_split.shape[2]

X_test_split = X_test_split.reshape((X_test_split.shape[0], n_seq, rows, n_steps, n_features))