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
