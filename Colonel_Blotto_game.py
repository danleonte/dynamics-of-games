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
        
        soldiers = self.actions[action1_index] - self.actions[action2_index]
        positions_won  = (soldiers>0).sum()
        positions_lost = (soldiers<0).sum()
            
        return np.sign(positions_won - positions_lost)
        
    
    def precompute_utilities(self):
        utilities = np.empty([self.nr_actions,self.nr_actions])
        
        for i in range(self.nr_actions):
            for j in range(self.nr_actions):
                utilities[i,j] = self.get_utility(i,j)
                
        return utilities
      
    
    def train_unconditional_regret(self):
        p1 = Player(self.nr_actions)
        p2 = Player(self.nr_actions)
        
        p1.last_regret = [1000] * self.nr_actions
        p2.last_regret = [1000] * self.nr_actions
        
        history_regrets = []
        history_prob    = []
        history_actions = []
        history_max_regrets = []
        
        
        for i in range(0,self.nr_steps+1):                
                
            [action1_index,proportions_1] = p1.choose_strategy_unconditional()
            [action2_index,proportions_2] = p2.choose_strategy_unconditional()
            
            utility1 = self.utilities[action1_index,action2_index]
            utility2 = self.utilities[action2_index,action1_index]
            
            regret1 = self.utilities[:,action2_index] - utility1
            regret2 = self.utilities[:,action1_index] - utility2            
            
            p1.update_unconditional(regret1,action1_index)
            p2.update_unconditional(regret2,action2_index)    
            
            ######  add the history  ####
        
            
            if i% self.multiple == 0 and i > 1 :
                
                #add regrets and probabilities, computing them from regrets                
                history_regrets.append(list(p1.last_regret)+['A',i])
                history_regrets.append(list(p2.last_regret)+['B',i])
                
                history_max_regrets.append([p1.last_regret.max() / (i+1),'A',i])
                history_max_regrets.append([p2.last_regret.max() / (i+1),'B',i])
                
                history_prob.append(list(proportions_1)+['A',i])
                history_prob.append(list(proportions_2)+['B',i])   
                
                #add action counts
                history_actions.append(list(p1.action_count / (i+1))+['A',i])
                history_actions.append(list(p2.action_count / (i+1))+['B',i])
                
        
        #save a history of probabilities, action played counts in a list
        columns = [str(tuple(i)) for i in self.actions] + ['player','iteration'] 
        df_regrets     = pd.DataFrame(history_regrets    ,columns = columns)
        df_prop        = pd.DataFrame(history_prob    ,columns = columns)
        df_count       = pd.DataFrame(history_actions ,columns = columns)
        df_max_regrets = pd.DataFrame(history_max_regrets, columns = ['max_regrets','player','iteration'] )

        
        self.history = [df_regrets,df_prop,df_count,df_max_regrets]

    
    def train_conditional_regret(self,mu1,mu2,type_):
        p1 = Player(self.nr_actions)
        p2 = Player(self.nr_actions)
        p1.mu,p2.mu = mu1,mu2

        
        p1.last_regret = np.zeros([self.nr_actions,self.nr_actions])
        p2.last_regret = np.zeros([self.nr_actions,self.nr_actions])
        
        history_regrets = []
        history_prob    = []
        history_actions = []
        history_swaps   = [0,0]
        
        for i in range(0,self.nr_steps+1):  
            
            history_regrets.append(list(p1.last_regret)+['A',i])
            history_regrets.append(list(p2.last_regret)+['B',i])
            
            [action1_index,proportions_1] = p1.choose_strategy_conditional(type_)
            [action2_index,proportions_2] = p2.choose_strategy_conditional(type_)
            
            if action1_index != p1.last_action : history_swaps[0] += 1
            if action2_index != p2.last_action : history_swaps[1] += 1
            
            utility1 = self.utilities[action1_index,action2_index]
            utility2 = self.utilities[action2_index,action1_index]
            
            regret1 = self.utilities[:,action2_index] - utility1
            regret2 = self.utilities[:,action1_index] - utility2
    
            p1.update_conditional(regret1,action1_index,utility1,i+1)
            p2.update_conditional(regret2,action2_index,utility2,i+1) 
            
            if i % self.multiple == 0:
                
                print(p1.last_regret.max() / (i+1),p1.last_regret.max() / (i+1))

                
                history_prob.append(list(proportions_1 / proportions_1.sum())+['A',i])
                history_prob.append(list(proportions_2 / proportions_2.sum())+['B',i])
                
                #add action counts
                history_actions.append(list(p1.action_count / (i+1))+['A',i])
                history_actions.append(list(p2.action_count / (i+1))+['B',i])
                

                
        #save a history of probabilities, action played counts in a list
        columns = [str(tuple(i)) for i in self.actions] + ['player','iteration'] 
        df_regrets = pd.DataFrame(history_regrets    ,columns = columns)
        df_prop    = pd.DataFrame(history_prob    ,columns = columns)
        df_count   = pd.DataFrame(history_actions ,columns = columns)
        
        self.history = [df_regrets,df_prop,df_count,history_swaps]

        
class Player:
    
    def __init__(self, nr_actions):
        self.action_count  = np.zeros(nr_actions)
        self.nr_actions = nr_actions
        
        #this will be sett to  vector [0] *nr_actions or  matrix [0] *nr_actions depending on which training method we choose       
        self.last_regret   = None
        #needed only for collel conditional training
        self.last_action = None
        self.mu = None
        
        
    def choose_strategy_unconditional(self):
        proportions = np.array([i if i > 0 else 0 for i in self.last_regret])
        if proportions.sum() > 0:
            proportions = proportions / proportions.sum()
        elif proportions.sum() == 0:
            proportions = [1/self.nr_actions] * self.nr_actions
        
        strategy_index = np.random.choice(self.nr_actions, p=proportions)
        return  [strategy_index,np.array(proportions)]        


    def update_unconditional(self,regret,action_index):
        self.last_regret  += regret
        self.action_count[action_index]  += 1
        
        
        
    def choose_strategy_conditional(self,type_):
        if self.action_count.sum() == 0:
            proportions = [1/self.nr_actions] * self.nr_actions
        else:
            if  type_  == 'stationary':
                A = np.array([[i if i>0 else 0 for i in arr] for arr in self.last_regret]) / self.mu()
                for i in self.nr_actions: 
                    A[i,i] = 0
                    A[i,i] = 1 - A[i,:].sum()
                
                # A is a stochastic matrix (row sums =1 )
                proportions = evec_with_eval1(A)
                
            elif type_ == 'collel':
                proportions = np.array([i if i >0 else 0 for i in self.last_regret[self.last_action,:]]) / self.mu
                proportions[self.last_action] = 0
                proportions[self.last_action] = 1 - proportions.sum() 
                
            
        strategy_index = np.argmax(np.random.multinomial(1,proportions) > 0)
        return  [strategy_index,np.array(proportions)]    #update the last action in update method


    
    def update_conditional(self,regret,action_index,iteration):        
        self.last_regret[action_index,:]  = (self.last_regret[action_index,:] * iteration + regret) / (iteration + 1)
        self.action_count[action_index]  += 1
        
        #update the last action with the new action played
        self.last_action = action_index
        
    

    




