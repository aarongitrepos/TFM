import os, sys, re
import numpy as np
import pandas as pd
import subprocess
import time
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(parentdir+"/SOTA_algorithms/SSD_Algo")
from SOTA_algorithms.SSD_Algo.RuleList import algo_test
from SD.utils import measures

def repr_pattern(pattern):
    result = ""
    for selector in pattern:
        result += selector[0] + "==" + selector[1] + " AND "
    return result[:-5]

def getAttrs(pattern):
        return [selector[0] for selector in pattern]

def redundancy(pattern, list_patterns):
    list_patterns_copy = list_patterns.copy()
    total_selectors = len(pattern)
    list_patterns_copy.remove(pattern)
    pattern_attrs = getAttrs(pattern)
    cont_attrs = 0
    cont_values = 0
    for elem in list_patterns_copy:
        cont_attrs += (len(list(set(pattern_attrs).intersection(getAttrs(elem)))) / total_selectors )
        cont_values += (len(list(set(pattern).intersection(elem))) / total_selectors)
    redundancy = (cont_attrs/len(list_patterns_copy))*(0.3) + (cont_values/len(list_patterns_copy))*(0.7)
    return redundancy 
        
def mrmr(patterns):
    l = []
    for _, pattern in enumerate(patterns):
        #relev = calculate_odd_value(ID,pattern.IS,PD,pattern.PS)
        redund = redundancy(pattern,patterns)
        #ratio = relev / redund
        l.append(redund)
    return l
    
    
def getStats(method,route_csv,name_df,FS):
    df = pd.read_csv(route_csv)
    df = df.astype(str)
    ID = df.shape[0]
    dict_targets_positives = {target: (measures.calculate_statistics_dataset(df, ["Class",target]),np.count_nonzero(measures.calculate_statistics_dataset(df, ["Class",target]))) for target in df["Class"].unique()}
    # SSD
    if method == 1:
        stats = {"Pattern":[],"Class":[],"Ratios":[],"Coverage":[],"Confidence":[],"Odds":[]}
        datasetRute = currentdir+"/results/"+name_df+"/SSD.txt"
        with open(datasetRute) as f:
            lines = f.read().splitlines()
            for line in lines[:-1]:
                pattern = re.search(".*[IF|If]\s(.+)\s.THEN",line).group(1)
                pattern = [(elem.split(" = ")[0],elem.split(" = ")[1]) for elem in pattern.split("  AND  ")]
                spl = line.split("Class")
                targets = spl[1].split(";")
                for target in targets[:-1]:
                    m = re.search(r"Pr\((.+)\)\s=\s(.+)",target)
                    if m.group(2) != "0.0":
                        IS, PS = measures.calculate_statistics_pattern(df, pattern, dict_targets_positives[m.group(1)][0])
                        cov = measures.coverage(ID, IS)
                        conf = measures.confidence(IS, PS)
                        odd = measures.calculate_odd_value(ID,IS,dict_targets_positives[m.group(1)][1],PS)
                        stats["Pattern"].append(repr_pattern(pattern))
                        stats["Class"].append(m.group(1))
                        stats["Coverage"].append(cov)
                        stats["Confidence"].append(conf)
                        stats["Odds"].append(odd)
                        stats["Ratios"].append(0)
            pd.DataFrame(data = stats).to_csv("./results/"+name_df+"/stats_SSD.csv")
    # FSSD
    elif method == 2:
        data = []
        for target in dict_targets_positives.keys():
            list_patterns = []
            stats = {"time":[],"target":[],"size":[],"length":[],"redundancy":[],"wracc":[],"coverage":[],"confidence":[],"odd":[]}
            #stats = {"Pattern":[],"Class":[],"Redundancy":[],"Coverage":[],"Confidence":[],"Odds":[]}
            with open(currentdir+"/results/"+name_df+"/FSSD_"+target+".txt") as f:
                lines = f.read().splitlines()
                for line in lines[1:-2]:
                    pattern = re.search(".+(\[.+\])\s(\[.+\])",line)
                    pattern = [(attr,value) for attr, value in zip(eval(pattern.group(1)),eval(pattern.group(2))) if value != "*"] 
                    IS, PS = measures.calculate_statistics_pattern(df, pattern, dict_targets_positives[target][0])
                    cov = measures.coverage(ID, IS)
                    conf = measures.confidence(IS, PS)
                    odd,_ = measures.calculate_odd_value(ID,IS,dict_targets_positives[target][1],PS)
                    wracc = measures.wracc(ID,dict_targets_positives[target][1],IS,PS,1)
                    list_patterns.append(pattern)
                    stats["length"].append(len(pattern))
                    stats["wracc"].append(wracc)
                    stats["coverage"].append(cov)
                    stats["confidence"].append(conf)
                    stats["odd"].append(odd) 
                    # stats["Pattern"].append(repr_pattern(pattern))
                    # stats["Class"].append(target)
                    # stats["Coverage"].append(cov)
                    # stats["Confidence"].append(conf)
                    # stats["Odds"].append(odd) 
            redundancy = mrmr(list_patterns)      
            stats["redundancy"] = redundancy
            data.append([lines[-1],target, len(lines[1:-2]), np.mean(stats["length"]), np.mean(redundancy),np.mean(stats["wracc"]),np.mean(stats["coverage"]),np.mean(stats["confidence"]),np.mean(stats["odd"])   ])
            # print(data)
            # exit(0)
        if FS == True:
            pd.DataFrame(data = data,columns=list(stats.keys())).to_csv("./results/"+name_df+"/FSSD_FS.csv")
        else:
            pd.DataFrame(data = data,columns=list(stats.keys())).to_csv("./results/"+name_df+"/FSSD_non_FS.csv")
        
                
            
def execute(method,route_csv,name_df):
    df = pd.read_csv(route_csv)
    df = df.astype(str)
    ## SSD
    if method == 1:    
        y = pd.Series(df.Class)
        df.drop(['Class'],axis=1,inplace=True)
        model = algo_test.execute(df,y)
        f = open(currentdir+"/results/"+name_df+"/SSD.txt","a")
        f.write(model.__str__())

    ## FSSD
    elif method == 2:
        route_py = currentdir + "/SOTA_algorithms/FSSD/FSSD/FSSD/main_topk_SD.py"
        #route_csv = ("./datasets_fs/"+str(name_df)+"_filter.csv")
        class_attr = "Class"
        nb_attr = df.shape[1] - 1
        for target_value in list(df["Class"].unique()):
            route_result = "./results/"+name_df+"/FSSD_"+str(target_value)+".txt"
            start = time.perf_counter()
            subprocess.run(["py", route_py, "--USE_ALGO", "--file", route_csv, "--delimiter", ",", "--class_attribute", class_attr, "--wanted_label", str(target_value), "--nb_attributes", str(nb_attr), "--top_k", "5", "--method", "fssd", "--results_file", route_result])
            #subprocess.run(["py", route_py, "--Q3", "--file", route_csv, "--class_attribute", class_attr, "--wanted_label", str(target_value), "--results_perf", route_result, "--offset", str(0), "--nb_attributes", str(nb_attr), "--top_k", "5", "--method", "DSSD","--delimiter", ",", "--depthmax" , str(4)])
            end = time.perf_counter()
            with open(currentdir+"/results/"+name_df+"/FSSD_"+str(target_value)+".txt","a") as f:
                f.write(str(end-start))                                                                                                                                                        

# name_df = "TUANDROMD"
# route_csv = "./datasets_fs/"+name_df+"_filter.csv"
# getStats(2,route_csv,name_df,True)
