"""
pandas
numpy
tensorflow
google-cloud-bigquery
google-cloud-bigquery-storage
google-cloud-storage
joblib
pyarrow
fastparquet
sklearn
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage
import pyarrow
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from sklearn.pipeline import Pipeline
from sklearn.compose import  ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA



def get_prediction(event, context):
     # BQ credentials
     client = bigquery.Client(project='tradebot-tfm')
     # Load the historical stock data from BQ
     sql_hist = """
          SELECT *
          FROM `tradebot-tfm.tradingbot_query.dataset_hist`
          WHERE
          date >= DATE_SUB(CURRENT_DATE(), INTERVAL 40 DAY)
          ORDER BY
          ticker,
          date
          """
     df = client.query(sql_hist).to_dataframe()
     print('ok1')
     data = df.copy()
     # Convert the date column to datetime
     data['date'] = pd.to_datetime(data['date'])
     data.drop( 'long_name', axis=1, inplace=True)
     data.dropna(inplace=True, axis=0)
     data.set_index('date', inplace=True)
     print('ok2')
     # Load model
     # Get cloud storage credentials
     storage_client = storage.Client(project='tradebot-tfm')
     bucket = storage_client.get_bucket('tradebot_bucket')
     blob = bucket.blob('Blstm_model_9.h5')
     blob.download_to_filename('/tmp/Blstm_model_9.h5')


     blob = bucket.blob('pipeline_x.pkl')
     blob.download_to_filename('/tmp/pipeline_x.pkl')
     print('ok3')
     with open('/tmp/pipeline_x.pkl',  'rb') as pipeline:
          pipeline_x_loaded = joblib.load(pipeline)
     print('ok4')
     scaled_data = pipeline_x_loaded.fit_transform(data)
     # split a multivariate sequence into samples
     def split_sequences(sequences, n_steps):
          X = list()
          for i in range(len(sequences)):
               # find the end of this pattern
               end_ix = i + n_steps
               # check if we are beyond the dataset
               if end_ix > len(sequences):
                    break
               # gather input and output parts of the pattern
               seq_x = sequences[i:end_ix]
               X.append(seq_x)
          return np.array(X)

     # choose a number of time steps
     n_steps = 20
     # convert into input/output
     X_split= split_sequences(scaled_data, n_steps)
     # the dataset knows the number of features, e.g. 2
     n_features = X_split.shape[2]
     X_split.astype('float32')
     print('ok5')
     model = load_model('/tmp/Blstm_model_9.h5')
     yhat = model.predict(X_split)
     print('ok6')
     df_predicted = df.copy()
     df_predicted = df_predicted.iloc[n_steps -1:]
     df_predicted['predictions'] = yhat

     # Get the latest date for the data we have
     current_data_date = df_predicted['date'].max()

     # Build dataframe t predict
     df_predicted = df_predicted[df_predicted['date'] == current_data_date]
     df_predicted = df_predicted[['date','ticker', 'long_name', 'sector', 'industry', 'close', 'predictions']]

     df_predicted.reset_index(inplace=True, drop=True)
     print('ok7')

     def strategy(value):
          if value < -2.0:
               return "Strong Sell"
          elif (value >= -2.0) & (value < -0.0):
               return "Sell"
          elif (value >= -0.0) & (value <= 5.0):
               return "Neutral"
          elif (value > 5.0) & (value <= 9.0):
               return "Buy"
          elif value > 9.0:
               return "Strong Buy"

     df_predicted.loc[:, 'recommendations'] = df_predicted['predictions'].apply(strategy)

     # Add to bigquery
     client = bigquery.Client(project='tradebot-tfm')

     dataset_id = 'tradingbot_query'
     table_id = 'predictions'

     dataset_ref = client.dataset(dataset_id)
     table_ref = dataset_ref.table(table_id)

     job_config = bigquery.LoadJobConfig()
     job_config.source_format = 'CSV'
     job_config.autodetect = True
     job_config.ignore_unknown_values = True
     job = client.load_table_from_dataframe(
          df_predicted,
          table_ref,
          location='US', 
          job_config=job_config
          )
     job.result()
     return 'Success'
