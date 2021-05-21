#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:39:43 2021

@author: mallen
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

os.chdir("/home/vegveg/suhi_predictor")
seed = 1
# =============================================================================
# 
# =============================================================================
# import data
b = pd.read_csv("./data/bubble_cleaned_v2.csv")

# drop unmodeled columns
bf = b.drop(['Unnamed: 0', 'datetime', 'decdate', 
             'uthem', 'rtbright',
             'utair2.6', 'utair13.9', 'utair17.5', 'utair21.5', 'utair25.5', 'utair31.5',
             'urh2.6', 'urh13.9', 'urh17.5', 'urh21.5', 'urh25.5', 
             'uwvc2.6', 'uwvc13.9', 'uwvc17.5', 'uwvc21.5', 'uwvc25.5',
             'ulup', 'rlup', 'rwd',
             'season'], axis = 1)

bfg = bf.groupby(['month', 'time']).mean().reset_index()

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(np.array(bf.loc[:,'utair':'rwv']), np.array(bf.loc[:,'suhi']), random_state = seed)
# note: using the whole set

# set model hyperparameters
model = RandomForestRegressor(n_estimators = 100, 
                              verbose = 1, 
                              oob_score = True,
                              random_state = seed,
                              n_jobs = -1)
# fit model
model.fit(X_train, y_train)

# feature importance
fi = model.feature_importances_
# merge with labels
cols = list(bf.loc[:,'utair':'rwv'].columns)
fic = pd.DataFrame([fi.T], columns = [cols])

# scores
oob = model.oob_score_
r2_post = model.score(X_train, y_train)

# predict
pred = model.predict(X_test)
r2_test = np.corrcoef([y_test, pred])[0,1]

### barplot
fig, ax = plt.subplots(1, 1)

ax.bar(cols, fi)