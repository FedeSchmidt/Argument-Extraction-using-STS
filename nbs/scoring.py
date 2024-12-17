import math

"""
NSCORE based on error penalization
"""
def get_N_score(M, PM, MU, UNR, similarity_values, gold, pred, lambda1 = 1, lambda2 = 1, lambda3 = 1):
    if pred == gold == 0:
        return 1
    elif pred == 0 and gold > 0:
        return (1/pred)-1
    elif pred > 0 and gold == 0:
        return (1/gold)-1
    else:
        weighted_PM = sum([(1-ri) for ri in similarity_values])
        return 1- (((lambda1*weighted_PM + lambda2*MU)/pred) + (lambda3*UNR/gold))