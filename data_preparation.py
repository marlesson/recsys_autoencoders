# Dataset - Steam Video Games
# https://www.kaggle.com/tamber/steam-video-games
#

import click
import pandas as pd
from util import *
from sklearn.model_selection import train_test_split

# event_type_strength = {
#    'purchase': 1,
#    'play': 1 
# }

@click.command(help="Processes the source data to create new data for the model")
@click.option("--min_interactions", type=click.INT, default=5, 
                  help="Minimun number of interactions for user.")
@click.option("--test_size", type=click.FLOAT, default=0.2)
@click.option("--factor_negative_sample", type=click.INT, default=0)
def run(min_interactions, test_size, factor_negative_sample):
  base_path = './data/raw/'

  # Contains logs of user interactions on shared articles
  interactions_df = pd.read_csv(base_path+'/rating.csv', index_col=None, header=None)
  interactions_df.columns=['user_id', 'game', 'type', 'hours', 'none']
  
  # Group interations by user_id and game
  interactions_full_df = interactions_df.groupby(['user_id', 'game'])\
                                        .sum()['hours'].reset_index()
                                        
  interactions_full_df['view'] = 1 # define 

  # Filter interactions
  interactions_full_df = filter_interactions(interactions_full_df, min_interactions)

  # Define dummy ID
  interactions_full_df['content_id'] = interactions_full_df['game'].astype('category').cat.codes
  interactions_full_df['user_id']    = interactions_full_df['user_id'].astype('category').cat.codes

  # Create a DataFrame with Content Information
  articles_df = interactions_full_df.groupby(['game', 'content_id'])\
                                      .agg({'user_id': 'count', 'hours': np.sum})[['user_id','hours']]\
                                      .reset_index()\
                                      .rename(columns={'user_id': 'total_users', 'hours': 'total_hours'})

  print('# of unique user/item interactions: %d' % len(interactions_full_df))

  # Split dataset in Train/Test
  interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                 stratify=interactions_full_df['user_id'], 
                                                 test_size=test_size,
                                                 random_state=42)  


  print("# size of train dataset before negative sample: %d" % len(interactions_train_df))
  
  # If use negative sample then create
  interactions_train_df =  interactions_with_negative_sample(interactions_train_df, 
                                                              factor_negative_sample=factor_negative_sample)
  
  print("# size of train dataset after negative sample: %d" % len(interactions_train_df))
  print("# size of test dataset: %d" % len(interactions_test_df))


  interactions_full_df[['user_id','content_id','game','hours','view']].to_csv('./data/interactions_full_df.csv', index = False)
  interactions_train_df[['user_id','content_id','game','hours','view']].to_csv('./data/interactions_train_df.csv', index = False)
  interactions_test_df[['user_id','content_id','game','hours','view']].to_csv('./data/interactions_test_df.csv', index = False)
  articles_df[['content_id', 'game','total_users','total_hours']].to_csv('./data/articles_df.csv', index = False)

def filter_interactions(interactions_df, min_interactions):
  '''
  Filter interactions of users with at least {min_interactions} interactions
  '''
  users_interactions_count_df = interactions_df.groupby('user_id').size()
  print('# users: %d' % len(users_interactions_count_df))
  
  users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= min_interactions]\
                                        .reset_index()[['user_id']]
  
  print('# users with at least %d interactions: %d' % (min_interactions, len(users_with_enough_interactions_df)))  

  print('# of interactions: %d' % len(interactions_df))
  interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
                                                               how = 'right',
                                                               left_on = 'user_id',
                                                               right_on = 'user_id')

  print('# of interactions from users with at least %d interactions: %d' % (min_interactions, len(interactions_from_selected_users_df)))

  return interactions_from_selected_users_df


def interactions_with_negative_sample(interactions_train_df, factor_negative_sample=3):
  '''
  Create a negative interactions, on the user not view content
  
  factor_negative_sample: Kx no-negative interactions 
  '''

  if factor_negative_sample == 0:
      return interactions_train_df
  
  # Top content Views
  top_content_views     = interactions_train_df.groupby('content_id').count()['user_id']\
                              .reset_index().sort_values('user_id', ascending=False).head(1000)
  
  interactions_train_df = interactions_train_df.set_index('user_id')

  all_df = []
  for user_id in np.unique(interactions_train_df.index.values):
    content_views    = interactions_train_df.loc[user_id].content_id.unique()
    content_not_view = top_content_views[~top_content_views.content_id.isin(content_views)]\
                            .content_id.values[:int(len(content_views)*factor_negative_sample)]

    df_view     = pd.DataFrame(data={'user_id': [user_id]*len(content_views), 
                                       'content_id': content_views, 
                                       'view': [1]*len(content_views)})

    df_not_view = pd.DataFrame(data={'user_id': [user_id]*len(content_not_view), 
                                       'content_id': content_not_view, 
                                       'view': [-1]*len(content_not_view)})

    df = pd.concat([df_view,df_not_view]).sample(frac=1)
    all_df.append(df)
  
  return pd.concat(all_df)    

if __name__ == '__main__':
    run()
