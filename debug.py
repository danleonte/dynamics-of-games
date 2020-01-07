#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:44:30 2020

@author: dan
"""
import numpy as  np
import pandas as pd
np.random.seed(0)
g = g=Game(5,3,100,100)
g.train_unconditional_regret()
[df_regrets,df_prop,df_count,df_reward,df_max_regrets] = g.history
df_regrets.to_csv('regrets.csv')
df_prop.to_csv('proportions.csv')
df_count.to_csv('action_count.csv')