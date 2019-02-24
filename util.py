import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns

def load_dataset(base_path = './data/', with_cartesian = False):
  articles_df           = pd.read_csv(base_path+'articles_df.csv')
  interactions_full_df  = pd.read_csv(base_path+'interactions_full_df.csv')
  interactions_train_df = pd.read_csv(base_path+'interactions_train_df.csv')
  interactions_test_df  = pd.read_csv(base_path+'interactions_test_df.csv')
  
  if with_cartesian:
    cartesian_product_df  = pd.read_csv(base_path+'cartesian_product_df.csv')
  else:
    cartesian_product_df  = None

  return articles_df, interactions_full_df, interactions_train_df, interactions_test_df, cartesian_product_df

def export_figure_matplotlib(arr, f_name, dpi=200, resize_fact=1, plt_show=False):
  """
  Export array as figure in original resolution
  :param arr: array of image to save in original resolution
  :param f_name: name of file where to save figure
  :param resize_fact: resize facter wrt shape of arr, in (0, np.infty)
  :param dpi: dpi of your screen
  :param plt_show: show plot or not
  """
  fig = plt.figure(frameon=False)
  fig.set_size_inches(arr.shape[1]/dpi, arr.shape[0]/dpi)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(arr, cmap='hot')
  plt.savefig(f_name, dpi=(dpi * resize_fact))
  if plt_show:
      plt.show()
  else:
      plt.close()


def plot_scores_values(values, f_name, plt_show=False):

  fig = plt.figure(figsize=(10,4))

  #ax = sns.boxplot(x=values)
  ax = sns.distplot(values)
  #ax = sns.swarmplot(x=values, color=".25")
  plt.savefig(f_name, dpi=(200))
  if plt_show:
      plt.show()
  else:
      plt.close()

def plot_metrics_disc(metrics, f_name, plt_show=False):
  '''
  Plot Metrics in Angle 
  '''
  labels = list(metrics.keys())
  stats  = list(metrics.values())
  
  angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
  # close the plot
  stats  = np.concatenate((stats,[stats[0]]))
  angles = np.concatenate((angles,[angles[0]]))

  fig = plt.figure(figsize=(6,6))
  ax  = fig.add_subplot(111, polar=True)
  ax.plot(angles, stats, linewidth=1, linestyle='solid')
  ax.fill(angles, stats, 'b', alpha=0.1)
  ax.set_thetagrids(angles * 180/np.pi, labels)
  plt.yticks([0.25,0.5, 0.75], ["0,25","0,5", "0,75"], color="grey", size=7)
  plt.ylim(0,1)
  
  plt.savefig(f_name, dpi=(200))
  if plt_show:
      plt.show()
  else:
      plt.close()

def plot_hist(hist):
  # summarize history for loss
  fig, ax = plt.subplots()  # create figure & 1 axis

  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  return fig   

def smooth_user_preference(x):
  return math.log(1+x, 2)
