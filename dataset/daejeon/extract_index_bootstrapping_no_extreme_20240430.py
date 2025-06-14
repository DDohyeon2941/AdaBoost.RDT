# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:14:55 2024

@author: dohyeon
"""

import pandas as pd
import numpy as np


if __name__ == "__main__":

    trn_df = pd.read_csv(r'2018_train_daejeon_weekdays_20240418.csv')
    trn_df1 = pd.read_csv(r'2018_train_daejeon_weekends_20240418.csv')

    pd.DataFrame(data=np.where(trn_df.y<10)[0]).to_csv(r'bootstrap_index_daejeon_weekdays_20240430.csv', index=False)
    pd.DataFrame(data=np.where(trn_df1.y<10)[0]).to_csv(r'bootstrap_index_daejeon_weekends_20240430.csv', index=False)






