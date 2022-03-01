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

def linear_parity(metrics):
    acc = metrics['overall_acc']
    par = metrics['stat_par_diff']
    return acc - abs(par)

def linear_odds(metrics):
    acc = metrics['overall_acc']
    odds = metrics['avg_odds_diff']
    return acc - abs(odds)

def linear_opportunity(metrics):
    acc = metrics['overall_acc']
    opp = metrics['eq_opp_diff']
    return acc - abs(opp)