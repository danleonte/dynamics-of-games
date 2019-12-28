import numpy as np
import pandas as pd
from auxiliary import enumerate_actions


class Game:    
    
    def __init__(self,S,N,nr_steps,multiple):
        assert S > N
        self.actions    = enumerate_actions(S,N)
        self.nr_actions = len(self.actions)
        self.utilities = self.precompute_utilities()
        self.nr_steps = nr_steps
        self.multiple = multiple
        self.history = None
        
    
    def get_utility(self,action1_index, action2_index): 
        
        positions_won  = ((self.actions[action1_index] - self.actions[action2_index])>0).sum()
        positions_lost = ((self.actions[action1_index] - self.actions[action2_index])<0).sum()
        result = positions_won - positions_lost
            
        if   result  ==  0 : return 0
        elif result <  0: return -1
        elif result >  0: return 1
        
    
    def precompute_utilities(self):
        utilities = np.empty([self.nr_actions,self.nr_actions])
        
        for i in range(self.nr_actions):
            for j in range(self.nr_actions):
                utilities[i,j] = self.get_utility(i,j)
                
        return utilities
      
    
    def train(self):
        p1 = Player(self.nr_actions)
        p2 = Player(self.nr_actions)
        
        history_regrets = []
        history_prob    = []
        history_actions = []
        history_rewards = []
        
        
        for i in range(0,self.nr_steps+1):                
                
            action1_index = p1.choose_strategy()
            action2_index = p2.choose_strategy()
            
            utility1 = self.utilities[action1_index,action2_index]
            utility2 = self.utilities[action2_index,action1_index]
            
            regret1 = self.utilities[:,action2_index] - utility1
            regret2 = self.utilities[:,action1_index] - utility2
            
            
            p1.update(regret1,action1_index,utility1)
            p2.update(regret2,action2_index,utility2)    
            
            
            ######  add the history  ####
            if i% self.multiple == 0 and i > 1 :
                
                history_regrets.append(list(p1.last_regret)+['A',i])
                history_regrets.append(list(p2.last_regret)+['B',i])

                
                #add regrets and probabilities, computing them from regrets
                proportions_1 = np.array([i if i > 0 else 0 for i in p1.last_regret]) 
                proportions_2 = np.array([i if i > 0 else 0 for i in p2.last_regret]) 
                
                history_prob.append(list(proportions_1 / proportions_1.sum())+['A',i])
                history_prob.append(list(proportions_2 / proportions_2.sum())+['B',i])   
                
                #add action counts
                history_actions.append(list(p1.action_count / i)+['A',i])
                history_actions.append(list(p2.action_count / i)+['B',i])
                
                #add action rewards
                r1 = p1.action_reward / p1.action_count
                r1[p1.action_count == 0] = 0
                
                r2 = p2.action_reward / p2.action_count
                r2[p2.action_count == 0] = 0
                
                
                history_rewards.append(list(r1)+['A',i])
                history_rewards.append(list(r2)+['B',i])

        
        #save a history of probabilities, action played counts and action rewards in a list
        columns = [str(tuple(i)) for i in self.actions] + ['player','iteration'] 
        df_regrets = pd.DataFrame(history_regrets    ,columns = columns)
        df_prop    = pd.DataFrame(history_prob    ,columns = columns)
        df_count   = pd.DataFrame(history_actions ,columns = columns)
        df_reward  = pd.DataFrame(history_rewards ,columns = columns)
        
        self.history = [df_regrets,df_prop,df_count,df_reward]

class Player:
    
    def __init__(self, nr_actions):
        self.last_regret   = [1/nr_actions] * nr_actions
        self.action_count  = np.zeros(nr_actions)
        self.action_reward = np.zeros(nr_actions)
        
        
    def choose_strategy(self):
        proportions = np.array([i if i > 0 else 0 for i in self.last_regret])
        proportions = proportions / proportions.sum()
        strategy_index = np.argmax(np.random.multinomial(1,proportions) > 0)
        return  strategy_index  

    def update(self,regret,action_index,reward):
        self.last_regret  += regret
        self.action_count[action_index]  += 1
        self.action_reward[action_index] += reward
        
    

    




