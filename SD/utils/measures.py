from scipy.stats import entropy, trim_mean
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import math
from . import subgroup

# Get an array with true elements when an instance of the dataset has attribute:value
def covers_selector(data, attribute, value):
    row = data[attribute].to_numpy()
    if pd.isnull(value):
        return pd.isnull(row)
    return row == value

# Get an array with true elements when instances have true label after checking attribute:value
def covers_subgroup(data, pattern):
    return np.all([covers_selector(data,sel[0],sel[1]) for sel in pattern] , axis=0)


def calculate_statistics_dataset(data, target):
    positives = covers_selector(data, target[0], target[1])
    return positives

def calculate_statistics_pattern(data, pattern, instances_positives_dataset):
    instances_covered = covers_subgroup(data, pattern)
    IS = np.count_nonzero(instances_covered)
    PS =  np.count_nonzero(np.all([instances_covered, instances_positives_dataset] , axis=0))
    return IS, PS

def calculate_measures(selectors, ID, PD, IS, PS, alpha):
    pattern = subgroup.Pattern(selectors=selectors)
    pattern.PS = PS
    pattern.IS = IS
    pattern.wracc = wracc(ID,PD,IS,PS,1-alpha)
    pattern.coverage = coverage(ID,IS)
    OR, OR_interval = calculate_odd_value(ID,IS,PD,PS)
    pattern.odd = OR
    pattern.odd_range = OR_interval
    pattern.confidence = confidence(IS,PS)
    return pattern

def wracc(ID,PD,IS,PS,a):
    if IS == 0:
        return -1
    #a = 1
    p_subgroup = PS / IS
    p_dataset = PD / ID
    return (IS / ID) ** a * (p_subgroup - p_dataset)

def coverage(ID, IS):
    return IS / ID

def confidence(IS, PS):
    return PS / IS

# Function that calculate information gained for a subgroup
def calculate_info_gained(ID,PD,IS,PS):
    a = ID - IS
    b = PD - PS
    x1 = IS / ID
    x2 = a / ID
    p1 = PD / ID
    p2 = PS / IS
    p3 = 0
    if a != 0:
        p3 = b / a
    return entropy([p1,1-p1],base=2) - (x1)*entropy([p2,1-p2],base=2) - (x2)*entropy([p3,1-p3],base=2)

def threshold2(l):
    n = len(l)
    if n == 1:
        return l[0]
    m = np.mean(l)
    cont = 0
    for elem in l:
        cont += (elem**2 - m)**2
    #return np.mean(l)
    return math.sqrt(cont/(n))

def threshold_og(l):
    # WE CAN COMPUTE DE MEDIAN AND THE THRESHOLD. THE MAX VALUE IS CONSIDERED
    n = len(l)
    if n == 1:
        return l[0]
    c1 = np.sum(np.square(l))
    c2 = n*np.mean(l)
    a = n*c1
    b = c2**2
    c = n*(n-1)
    s = math.sqrt((a-b)/c)
    #return min(s,np.mean(l))
    return s

def threshold(l):
# mean > median -> positive skewed, concentracion en menores valores
# mean < median -> negative skewed, concentracion en mayores valores
  
    # mean = np.mean(l)
    # median = np.median(l)
    # if mean >= median:
    #     return median
    # return np.percentile(l,75)
    return np.median(l)
    
def tritmean(l):
    if len(set(l)) == 1:
        return l[0]
    #return np.percentile(l, 75)
    q3 = np.percentile(l, 75)
    return q3

# Function that calculate odd value for a subgroup
def calculate_odd_value(ID,IS,PD,PS):
    a = PS
    b = IS - PS
    c = PD - PS
    d = (ID - PD) - b
    if b == 0 or c == 0 or d == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5
    w = 1.96*math.sqrt((1/a)+(1/b)+(1/c)+(1/d))
    #w = 1.645*math.sqrt((1/a)+(1/b)+(1/c)+(1/d))
    OR = (a*d) / (b*c)
    OR_interval = (OR*math.exp(-w), OR*math.exp(+w))
    return OR, OR_interval

def evaluate_OR_range_overlap(IS1,PS1,IS2,PS2):
    a = PS1 / IS1
    b = PS2 / IS2
    stderror1 = math.sqrt((a*(1-a))/IS1)
    stderror2 = math.sqrt((b*(1-b))/IS2)
    diff_between = abs(a - b)
    std_error_diff = math.sqrt(stderror1**2 + stderror2**2)
    if (diff_between + 1.96*std_error_diff) > 0 and (diff_between + 1.96*std_error_diff) > 0:
        return True
    return False
        
    

def redundancy(pattern, list_patterns, beta):
    list_patterns_copy = list_patterns.copy()
    total_selectors = len(pattern.selectors)
    list_patterns_copy.remove(pattern)
    pattern_attrs = pattern.getAttrs()
    cont_attrs = 0
    cont_values = 0
    for elem in list_patterns_copy:
        cont_attrs += (len(list(set(pattern_attrs).intersection(elem.getAttrs()))) / total_selectors )
        cont_values += (len(list(set(pattern.selectors).intersection(elem.selectors))) / total_selectors)
    redundancy = (cont_attrs/len(list_patterns_copy))*beta + (cont_values/len(list_patterns_copy))*(1-beta)
    return redundancy + 0.01

def threshold_test(l, number_groups, avg):
    #print("Number of total patterns {}, Number of possible children {}".format(number_groups,len(l)))
    print(avg)
    print(len(l))
    exit(0)
    if len(l) <= number_groups:
        return len(l)
    if number_groups > 0:
        top_k_mean = np.mean(l[:number_groups])
        if top_k_mean <= avg:
            return number_groups
    for i in range(number_groups+1,len(l)):
        if np.mean(l[:i]) <= avg:
            break
    return i
        
def mrmr(patterns, beta,a=0):
    odds_norm = normalize([[elem.wracc for elem in patterns]],norm="max")[0]
    odds_norm = [elem.wracc for elem in patterns]
    
    ratios = []
    for i, pattern in enumerate(patterns):   
        redund = redundancy(pattern,patterns,beta)
        ratio = odds_norm[i] / redund
        ratios.append(ratio)
        pattern.redundancy = redund
        pattern.ratio = ratio
    results = [pattern for pattern, ratio in zip(patterns,ratios) if ratio >= threshold2(ratios)]
    if len(results) == 0:
        return [pattern for pattern, ratio in zip(patterns,ratios) if ratio >= threshold(ratios)]
    return results