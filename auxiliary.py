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

def evec_with_eval1(A):
    """ A is a stochastic matrix (np.array) which has a stationary distribution (left evector with evalue 1)
    """
    e_val,e_vec = np.linalg.eig(np.transpose(A))
    index_eval_1 = np.argmin(np.abs(e_val-1))
    return e_vec[:,index_eval_1]