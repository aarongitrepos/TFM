class Pattern():
    def __init__(self, selectors=[], IS=0, PS=0, value=0):
        self.selectors = selectors
        self.PS = PS
        self.IS = IS
        self.wracc = value
        self.coverage = 0
        self.confidence = 0
        self.odd = 0
        self.odd_range = (0,0)
        self.ratio = 0
        self.wracc_gradient = 0
        self.redundancy = 0
        self.ig = 0
        self.ig_gradient = 0
        
    def getAttrs(self):
        return [selector[0] for selector in self.selectors]
    
    def getVals(self):
        return [selector[1] for selector in self.selectors]
        
    def __repr__(self) -> str:
        pattern = ""
        for selector in self.selectors:
            pattern += selector[0] + "==" + selector[1] + " AND "
        return pattern[:-5]