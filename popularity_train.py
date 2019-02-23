import click
import pandas as pd
import math
from util import *
from sklearn.model_selection import train_test_split
from evaluation.model_evaluator import *
import mlflow

# Const
METRICS_LOG_PATH    = './artefacts/metrics.png'
SCORE_VALUES_PATH   = './artefacts/score_plot.png'

class PopularityRecommender:
  MODEL_NAME = 'Popularity'
  
  def __init__(self, popularity_df, items_df=None):
      self.popularity_df = popularity_df
      self.items_df = items_df
      
  def get_model_name(self):
      return self.MODEL_NAME
      
  def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
      # Recommend the more popular items that the user hasn't seen yet.
      recommendations_df = self.popularity_df[~self.popularity_df['content_id'].isin(items_to_ignore)] \
                             .sort_values('view', ascending = False) \
                             .head(topn)
      return recommendations_df

def run():
  print("run()")
  # Load Dataset
  articles_df, interactions_full_df, \
    interactions_train_df, interactions_test_df, \
      cf_preds_df = load_dataset()

  #interactions_full_df['eventStrength'] = interactions_full_df['eventStrength'].apply(lambda x: 1 if x > 0 else 0)
  print('# interactions on Train set: %d' % len(interactions_train_df))
  print('# interactions on Test set: %d' % len(interactions_test_df))

  # Train 
  ##  Computes the most popular items
  item_popularity_df = interactions_full_df.groupby('content_id')['view'].sum()\
                          .sort_values(ascending=False).reset_index()

  popularity_model   = PopularityRecommender(item_popularity_df, articles_df)


  # Evaluation
  model_evaluator    = ModelEvaluator(articles_df, interactions_full_df, 
                                       interactions_train_df, interactions_test_df)    
  
  print('Evaluating Popularity recommendation model...')
  pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
  print('\nGlobal metrics:\n%s' % pop_global_metrics)

  # Plot Metrics
  plot_metrics_disc(pop_global_metrics, METRICS_LOG_PATH)

  # Tracking
  with mlflow.start_run():
    for metric in ModelEvaluator.METRICS:
      mlflow.log_metric(metric, pop_global_metrics[metric])
      mlflow.log_artifact(METRICS_LOG_PATH, "evaluation")


if __name__ == '__main__':
    run()  