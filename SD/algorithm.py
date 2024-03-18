import pandas as pd
import numpy as np
from .utils import measures
from .utils import subgroup

    
def create_nominal_selectors(data, ignore=None):
    if ignore is None:
        ignore = []
    nominal_selectors = []
    for attribute in [attribute for attribute in data.columns.values if attribute != "Class" and attribute not in ignore ]:
        nominal_selectors.extend(create_nominal_selectors_for_attribute(data, attribute))
    return nominal_selectors
    
def create_nominal_selectors_for_attribute(data, attribute, dtypes=None):
    nominal_selectors = []
    for value in pd.unique(data[attribute]):
        nominal_selectors.append((attribute, value))
    return nominal_selectors


class SubgroupDiscoveryTask:
    '''
    Capsulates all parameters required to perform standard subgroup discovery
    '''
    def __init__(self, data, target, search_space, name_df, depth=3, alpha=0, beta=0,timeout=3600, filter_vars=None, constraints=None):
        
        self.data = data
        self.target = target
        self.search_space = search_space
        self.depth = depth
        self.alpha = alpha
        self.beta = beta
        self.name_df = name_df
        self.timeout = timeout
        self.average_ratio = 0
        if constraints is None:
            constraints = []
        if filter_vars is None:
            filter_vars = []
        self.filter_vars = filter_vars
        
        
    def calculate_constant_statistics(self, data, target):
        self.instances_positives_dataset = measures.calculate_statistics_dataset(data, target)
        self.PD = np.count_nonzero(self.instances_positives_dataset)
        self.ID = data.shape[0]
        

class SubgoupDiscoverySearch:
    
    def generate_df(self,results,name_df,target,beta):
        #ratios = measures.mrmr(results)
        data = {"pattern":[str(elem) for elem in results],"length":[len(elem.selectors) for elem in results],"redundancy":[elem.redundancy for elem in results],"wracc":[elem.wracc for elem in results],"coverage":[elem.coverage for elem in results],
                "confidence":[elem.confidence for elem in results], "odd":[elem.odd for elem in results],"IS":[elem.IS for elem in results],
                "PS":[elem.PS for elem in results]}
        df = pd.DataFrame(data=data)
        df.loc[len(df.index)] = [len(df.index), np.mean(df["length"]), np.mean(df["redundancy"]), np.mean(df["wracc"]),np.mean(df["coverage"]),np.mean(df["confidence"]),np.mean(df["odd"]),"-","-"]  
        return [str(target[1]),len(results), np.mean(df["length"]), np.mean(df["redundancy"]), np.mean(df["wracc"]),np.mean(df["coverage"]),np.mean(df["confidence"]),np.mean(df["odd"])]  
        #df.to_csv("../results/"+name_df+"/stats_IGSD_"+str(target[1])+".csv")
   

    def evaluate(self, list_patterns, depth, parent):
        list_patterns = list(filter(lambda a: a.wracc > 0, list_patterns))
        if len(list_patterns) == 0:
            return list_patterns
        ## Si es iteración 0 solo aplicamos threshold wracc
        if depth == 0:
            thre = measures.threshold_og([pattern.wracc for pattern in list_patterns])        
            list_patterns = list(filter(lambda a: a.wracc >= thre, list_patterns))
        elif depth == 1:
            #print(len(list_patterns))
            list_patterns = list(filter(lambda a: a.odd > parent.odd and a.odd_range[0] > parent.odd_range[1], list_patterns))
            #print(len(list_patterns))
        ## Si es iteración > 1, hay que descartar aquellos patrones que no cumplan el gradiente.
        else:
            list_patterns = list(filter(lambda a: (abs(a.wracc / parent.wracc) / parent.wracc) >= parent.wracc_gradient 
                                        and (a.odd > parent.odd and a.odd_range[0] > parent.odd_range[1]),list_patterns))
            # list_patterns = list(filter(lambda a: (abs(a.wracc / parent.wracc) / parent.wracc) >= parent.wracc_gradient 
            #                             and (a.odd > parent.odd and (a.odd_range[0] > parent.odd_range[1]) or measures.evaluate_OR_range_overlap(parent.IS,parent.PS,a.IS,a.PS)),list_patterns))
        for pattern in list_patterns:
            if depth > 0:
                g1 = abs(pattern.wracc / parent.wracc) / parent.wracc
                pattern.wracc_gradient = g1
        #print(list_patterns[0].gradient)
        return list_patterns
    
    def evaluate_if_append(self,results,pattern):
        if len(results) > 0:
            for patt in results:
                redund = len(list(set(pattern.selectors).intersection(patt.selectors)))
                if redund >= (len(pattern.selectors) - 1) and pattern.wracc <= patt.wracc:
                    return results
        results.append(pattern)
        return results       
                
    def check_redundancy(self,results):
        r = []
        for i, pattern in enumerate(results):
            f = False
            for patt in results[i+1:]:
                #print(pattern,patt)
                if all(elem in patt.selectors for elem in pattern.selectors):
                    f = True
                    break
            if f == False:
                r.append(pattern)
        return r
                    
    def execute(self, task):
        task.calculate_constant_statistics(task.data, task.target)
        patterns = [subgroup.Pattern(IS=task.ID, PS=task.PD)]
        patterns_aux = []
        list_aux = []
        patterns_aux_attr = []
        patterns_aux_vals = []
        results = []
        depth = 0
        while depth < task.depth and len(patterns) > 0:
            alpha = task.alpha * depth
            print("Iteration {} of {}. Number of patterns: {}".format(depth,task.depth,len(patterns)))
            smt = False
            for pattern in patterns:
                patter_attrs = pattern.getAttrs()
                for selector in list(filter(lambda a: a[0] not in patter_attrs, task.search_space)):
                    attributes = patter_attrs + [selector[0]]
                    values = pattern.getVals() + [selector[1]]
                    if smt==False or (sorted(attributes) not in patterns_aux_attr or sorted(values) not in patterns_aux_vals):
                        IS, PS = measures.calculate_statistics_pattern(task.data, list(zip(attributes, values)), task.instances_positives_dataset)
                        if IS > 0 and PS > 0:
                            new_pattern = measures.calculate_measures(pattern.selectors + [(selector[0],selector[1])], task.ID, task.PD, IS, PS, alpha)
                            #list_aux = self.evaluate(new_pattern, list_aux, alpha)
                            list_aux.append(new_pattern)
                #list_aux = self.evaluate(list_aux, depth, pattern.wracc, pattern.odd, pattern.odd_range, pattern.wracc_gradient)
                list_aux = self.evaluate(list_aux, depth, pattern)
                if len(list_aux) == 0:
                    #print(results)
                    results = self.evaluate_if_append(results,pattern)
                    #print("Number of patterns accumulated in results {}".format(len(results)))
                else:
                    for pattern in list_aux:
                        patterns_aux_attr.append(sorted(pattern.getAttrs()))
                        patterns_aux_vals.append(sorted(pattern.getVals()))
                        patterns_aux.append(pattern)
                    smt = True
                    list_aux.clear()
            #print(len(patterns_aux))
            # if depth > 0 and len(patterns_aux) > 1:
            #     patterns_aux = measures.mrmr(patterns_aux, task.beta)
            #print(len(patterns_aux))
            #print("Number of patterns after filter {}".format(len(patterns_aux)))
            patterns = patterns_aux.copy()
            depth += 1
            patterns_aux.clear()
            patterns_aux_attr.clear()
            patterns_aux_vals.clear()
            #results.clear()
        if len(patterns) > 0:
            results = self.evaluate_if_append(results,pattern)
        results = self.check_redundancy(results)
        print(len(results))
        results = measures.mrmr(results, task.beta)
        print(len(results))
        #results = measures.mrmr(results, task.beta)
       
        return self.generate_df(results,task.name_df,task.target,task.beta)


if __name__ == "__main__":

    a = []
    
    name_df = "mushrooms"
    df = pd.read_csv("../datasets_fs/"+name_df+"_filter.csv")
    #df = pd.read_csv("../datasets/"+name_df+".csv")
    df = df.astype(str)
    
    searchspace = create_nominal_selectors(df)
    a = []
    for target in df["Class"].unique():
        target = ["Class", target]
        print("Class: {}".format(target[1]))
        task = SubgroupDiscoveryTask (
                df, 
                target, 
                searchspace,
                name_df, 
                depth=df.shape[1] -1,
                alpha = 1/(df.shape[1]-1),
                beta = 0.1 )
        r = SubgoupDiscoverySearch().execute(task)
        a.append(r)
    df = pd.DataFrame(a,columns = ["target","size","length","redundancy","wracc","coverage","confidence","odd"])
    df = pd.DataFrame(df.mean().to_dict(),index=[df.index.values[-1]])
    print(df)