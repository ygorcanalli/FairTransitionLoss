import numpy as np

def paraboloid_parity(metrics):
    acc = metrics['overall_acc']
    par = metrics['stat_par_diff']
    return -par**2 - (acc-1)**2

def paraboloid_odds(metrics):
    acc = metrics['overall_acc']
    odds = metrics['avg_odds_diff']
    return -odds**2 - (acc-1)**2

def paraboloid_opportunity(metrics):
    acc = metrics['overall_acc']
    opp = metrics['eq_opp_diff']
    return -opp**2 - (acc-1)**2

def acc_parity(metrics):
    acc = metrics['overall_acc']
    par = metrics['stat_par_diff']
    return acc - abs(par)

def acc_odds(metrics):
    acc = metrics['overall_acc']
    odds = metrics['avg_odds_diff']
    return acc - abs(odds)

def acc_opportunity(metrics):
    acc = metrics['overall_acc']
    opp = metrics['eq_opp_diff']
    return acc - abs(opp)

def mcc_parity(metrics):
    mcc = metrics['MCC']
    par = metrics['stat_par_diff']
    return mcc - par

def mcc_odds(metrics):
    mcc = metrics['MCC']
    odds = metrics['avg_odds_diff']
    return mcc - odds

def mcc_opportunity(metrics):
    mcc = metrics['MCC']
    opp = metrics['eq_opp_diff']
    return mcc - opp

def bal_acc_linear_parity(metrics):
    acc = metrics['bal_acc']
    par = metrics['stat_par_diff']
    return acc - abs(par)

def bal_acc_linear_odds(metrics):
    acc = metrics['bal_acc']
    odds = metrics['avg_odds_diff']
    return acc - abs(odds)

def bal_acc_linear_opportunity(metrics):
    acc = metrics['bal_acc']
    opp = metrics['eq_opp_diff']
    return acc - abs(opp)
def linear_all(metrics):
    acc = metrics['overall_acc']
    opp = metrics['eq_opp_diff']
    odds = metrics['avg_odds_diff']
    par = metrics['stat_par_diff']
    unfairness = np.sqrt(opp**2 + odds**2 + par**2)
    return acc - unfairness

def accuracy_only(metrics):
    return metrics['overall_acc']