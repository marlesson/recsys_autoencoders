# Evaluation 
#
# These evaluation are derived from the
# https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101

import pandas as pd
import random
import numpy as np
from . import metrics

class ModelEvaluator:
  METRICS = ['ndcg_at.5','ndcg_at.10', 'converge','personalization']
  
  def __init__(self, articles_df, interactions_full_df, 
                      interactions_train_df, interactions_test_df):

    #Indexing by user_id to speed up the searches during evaluation
    self.articles_df                   = articles_df
    self.interactions_full_indexed_df  = interactions_full_df[interactions_full_df.view > 0].set_index('user_id')
    self.interactions_train_indexed_df = interactions_train_df[interactions_train_df.view > 0].set_index('user_id')
    self.interactions_test_indexed_df  = interactions_test_df.set_index('user_id')


  def get_items_interacted(self, person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[person_id]['content_id']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

  def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
    interacted_items = self.get_items_interacted(person_id, self.interactions_full_indexed_df)
    all_items = set(self.articles_df['content_id'])
    non_interacted_items = all_items - interacted_items

    random.seed(seed)
    non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
    return set(non_interacted_items_sample)

  def _verify_hit_top_n(self, item_id, recommended_items, topn):        
    try:
        index = next(i for i, c in enumerate(recommended_items) if c == item_id)
    except:
        index = -1
    hit = int(index in range(0, topn))
    return hit, index

  def recommender_model_for_user(self, model, person_id):
    #Getting the items in test set
    interacted_values_testset = self.interactions_test_indexed_df.loc[person_id]
    if type(interacted_values_testset['content_id']) == pd.Series:
        person_interacted_items_testset = np.array(interacted_values_testset['content_id'])
    else:
        person_interacted_items_testset = np.array([int(interacted_values_testset['content_id'])])  
    interacted_items_count_testset = len(person_interacted_items_testset) 

    #Getting a ranked recommendation list from a model for a given user
    person_recs_df = model.recommend_items(person_id, 
                                           items_to_ignore=self.get_items_interacted(person_id, 
                                                                                self.interactions_train_indexed_df), 
                                           topn=20)
    # Recommender Content_iD
    #recs_content_id   = person_recs_df['content_id'].values[:10]


    person_metrics = {'recommender':  person_recs_df['content_id'].values,
                      'labels':       person_interacted_items_testset}

    return person_metrics

  def evaluate_model(self, model):
    print('Running evaluation for users')
    people_recs     = []
    recs_content_id  = []
    users           = list(self.interactions_test_indexed_df.sample(frac=1).index.unique().values)[:500]
    len_users       = len(users)

    for idx, person_id in enumerate(users):
        if idx % 500 == 0 and idx > 0:
           print('%.2f users processed' % (idx/len_users))

        person_recs = self.recommender_model_for_user(model, person_id)  
        person_recs['person_id'] = person_id
        people_recs.append(person_recs)

    print('%d users processed' % idx)
    print("evaluations...")

    detailed_results_df = pd.DataFrame(people_recs)
    predictions = detailed_results_df['recommender'].values
    labels      = detailed_results_df['labels'].values

    ndcg_at_5       = metrics.ndcg_at(predictions, labels, k=5)
    ndcg_at_10      = metrics.ndcg_at(predictions, labels, k=10)
    mean_average_precision = metrics.mean_average_precision(predictions, labels)
    coverage        = metrics.coverage(predictions, self.articles_df['content_id'].values)/100
    personalization = metrics.personalization(predictions)

    global_metrics = {'ndcg_at.5':       ndcg_at_5,
                      'ndcg_at.10':      ndcg_at_10,
                      'MAP':             mean_average_precision,
                      'converge':        coverage,
                      'personalization': personalization }    
    
    

    return global_metrics, detailed_results_df
      

class CFRecommender:
  '''

  '''
  
  MODEL_NAME = 'Collaborative Filtering'
  
  def __init__(self, cf_predictions_df, items_df=None):
      self.cf_predictions_df = cf_predictions_df
      self.items_df = items_df
      
  def get_model_name(self):
      return self.MODEL_NAME
      
  def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
      # Get and sort the user's predictions
      sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                                  .reset_index().rename(columns={user_id: 'score'})

      # Recommend the highest predicted rating movies that the user hasn't seen yet.
      recommendations_df = sorted_user_predictions[~sorted_user_predictions['content_id'].isin(items_to_ignore)] \
                             .sort_values('score', ascending = False) \
                             .head(topn)

      if verbose:
        if self.items_df is None:
            raise Exception('"items_df" is required in verbose mode')

        recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                      left_on = 'content_id', 
                                                      right_on = 'content_id')[['score', 'content_id', 'game']]


      return recommendations_df    