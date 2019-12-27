import numpy as np
from auxiliary import enumerate_actions


class Game:    
    
    def __init__(self,S,N,nr_steps):
        assert S > N
        self.actions    = enumerate_actions(S,N)
        self.nr_actions = len(self.actions)
        self.utilities = self.precompute_utilities()
        self.nr_steps = nr_steps
        self.history = []

    
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
        
        for i in range(self.nr_steps):
            #if i% 10000 == 0:
               # self.history.append([p1.last_regret,p2.last_regret])
                #modify this to get a history
            
            action1_index = p1.choose_strategy()
            action2_index = p2.choose_strategy()
            
            utility1 = self.utilities[action1_index,action2_index]
            utility2 = self.utilities[action2_index,action1_index]
            #check if the transpose (except diagonal) is minus the toher thing
            
            regret1 = self.utilities[:,action2_index] - utility1
            regret2 = self.utilities[:,action1_index] - utility2
            
            
            p1.update(regret1,action1_index,utility1)
            p2.update(regret2,action2_index,utility2)     
    
        return [p1,p2]

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
        
    

    




