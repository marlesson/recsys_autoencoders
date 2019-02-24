import click
import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
from util import *
from model.CDAEModel import *
from model.AutoEncModel import *
from model.AutoEncContentModel import *
from evaluation.model_evaluator import *

# main
# ----------------------------------------------
@click.command(help="Recommender Matrix Fatorization Model")
@click.option("--name", type=click.Choice(['auto_enc', 'cdae', 'auto_enc_content']))
@click.option("--model_path", type=click.STRING)
@click.option("--user_id", type=click.INT, default=1)
@click.option("--topn", type=click.INT, default=10)
@click.option("--view", type=click.INT, default=0)
@click.option("--output", type=click.STRING, default='./data/predict.csv')
def run(name, model_path, user_id, topn, view, output):
  
  # Load Dataset
  articles_df, _n, interactions_hist, _n2, _n3 = load_dataset()


  #Creating a sparse pivot table with users in rows and items in columns
  users_items_matrix_df = interactions_hist.pivot(index   = 'user_id', 
                                                  columns = 'content_id', 
                                                  values  = 'view').fillna(0)


  # Data
  users_items_matrix    = users_items_matrix_df.values
  users_ids             = list(users_items_matrix_df.index)

  if name == 'cdae':
    model = CDAEModel()
  elif name == 'auto_enc':
    model = AutoEncModel()
  elif name == 'auto_enc_content':
    model = AutoEncContentModel()

  # Input - Prepare input layer
  X, y   = model.data_preparation(interactions_hist, users_items_matrix_df)

  # Keras Model
  model  = mlflow.keras.load_model(model_path+name)

  # Predict
  if view == 0: # New Predic
    pred_score  = model.predict(X) * (X[0] == view)
  else: # User Interactive Hist
    pred_score  = users_items_matrix


  # converting the reconstructed matrix back to a Pandas dataframe
  cf_preds_df = pd.DataFrame(pred_score, 
                              columns = users_items_matrix_df.columns, 
                              index=users_ids).transpose()
  

  # Evaluation Model
  cf_recommender_model = CFRecommender(cf_preds_df, articles_df)

  # Recommender
  rec_list = cf_recommender_model.recommend_items(user_id = user_id, 
                                                  items_to_ignore=[], 
                                                  topn=topn, 
                                                  verbose=True)
  rec_list = rec_list[rec_list.score > 0]

  print(rec_list)
  rec_list.to_csv(output, index=False)


if __name__ == '__main__':
    run()  