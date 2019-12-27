import numpy as np

def enumerate_actions(n,k):
    """ list all length k tuples with non-negative entries which sum up to n 
    """
    result = []
    
    def helper(partial_sol,n,k):
        if k == 1:
            result.append(partial_sol + [n,])
        elif k > 1:
            for i in range(n+1):
                helper(partial_sol+[i],n-i,k-1)
    helper([],n,k)
                
    return np.array(result)