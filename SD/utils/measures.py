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

def calculate_measures(selectors, ID, PD, IS, PS, alpha, IP, PP):
    pattern = subgroup.Pattern(selectors=selectors)
    pattern.PS = PS
    pattern.IS = IS
    pattern.wracc = wracc(ID,PD,IS,PS,1-alpha)
    pattern.ig = calculate_info_gained(IP,PP,IS,PS)
    pattern.coverage = coverage(ID,IS)
    OR, OR_interval = calculate_odd_value(ID,IS,PD,PS)
    pattern.odd = OR
    pattern.odd_range = OR_interval
    pattern.confidence = confidence(IS,PS)
    return pattern

def wracc(ID,PD,IS,PS,a):
    if IS == 0:
        return -1
    p_subgroup = PS / IS
    p_dataset = PD / ID
    return (IS / ID) * (p_subgroup - p_dataset)

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
    return np.mean([math.sqrt(cont/(n)),np.median(l)]) 

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
    #w = 1.96*math.sqrt((1/a)+(1/b)+(1/c)+(1/d))
    w = 1.39*math.sqrt((1/a)+(1/b)+(1/c)+(1/d))
    OR = (a*d) / (b*c)
    OR_interval = (OR*math.exp(-w), OR*math.exp(+w))
    return OR, OR_interval

def odd_equivalent(odd):
    #OR: <1.68, 1.68 - 3.47, 3.47 - 6.71, >6.71
    if odd < 1.68:
        return 1
    elif odd >= 1.68 and odd < 3.47:
        return 2
    elif odd >= 3.47 and odd < 6.71:
        return 3
    return 4

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

def redundancy(pattern, list_patterns):
    if len(list_patterns) == 1:
        return 0.01
    list_patterns_copy = list_patterns.copy()
    total_selectors = len(pattern.selectors)
    list_patterns_copy.remove(pattern)
    pattern_attrs = pattern.getAttrs()
    cont_attrs = 0
    cont_values = 0
    for elem in list_patterns_copy:
        cont_attrs += (len(list(set(pattern_attrs).intersection(elem.getAttrs()))) / total_selectors )
        cont_values += (len(list(set(pattern.selectors).intersection(elem.selectors))) / total_selectors)
    redundancy = (cont_attrs/len(list_patterns_copy))*0.5 + (cont_values/len(list_patterns_copy))*0.5
    return redundancy + 0.01

def mrmr(patterns,t):

    ratios = []
    for i, pattern in enumerate(patterns):   
        redund = redundancy(pattern,patterns)
        ratio = pattern.wracc / redund
        ratios.append(ratio)
        pattern.redundancy = redund
        pattern.ratio = ratio
    if t == True:
        results = [pattern for pattern, ratio in zip(patterns,ratios) if ratio >= threshold2(ratios)]
        # if len(results) == 0:
        #     return [pattern for pattern, ratio in zip(patterns,ratios) if ratio >= threshold(ratios)]
        return results
    return patterns