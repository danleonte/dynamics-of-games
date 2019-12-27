#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 04:30:52 2019

@author: dan
"""

from Colonel_Blotto_game import *
g = Game(5,3,100000)
#print(g.actions)
#print(g.get_utility(-5,-5))

a= time.time()
p1,p2 = g.train()
b = time.time()


#p2.action_count.sum()

plt.scatter([i for i in range(1,g.nr_actions+1)],p2.action_count)
plt.scatter([i for i in range(1,g.nr_actions+1)],p1.action_count)

plt.hist(p2.last_regret)
plt.hist(p1.last_regret)
