from . import ssl
from .acpl_fsl import  acpl_fsl_select
import numpy as np


def get_indices(selection_method, episode_dict, support_size_max=None):
    # random 
    if selection_method == "random":
        ind = np.random.choice(episode_dict["unlabeled"]["samples"].shape[0], 1, replace=False)
    
    # random  imbalanced
    if selection_method == "random_imbalanced":
        ind = np.random.choice(episode_dict["unlabeled"]["samples"].shape[0], 1, replace=False)
    
    # ssl
    if selection_method == "ssl":
        ind = ssl.ssl_get_next_best_indices(episode_dict)

    # ACPL-FSL
    if selection_method == 'ACPL':
        # ind = acpl_fsl_select(episode_dict)
        ind = acpl_fsl_select(episode_dict)



    return ind
