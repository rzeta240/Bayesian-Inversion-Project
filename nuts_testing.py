import numpy as np
import pymc as pm
import pytensor
pytensor.config.cxx = ""

with pm.Model() as model:
    x = pm.Normal("x", mu = 0, sigma = 1)

    trace = pm.sample(
        draws=2500,  
        tune=1250,    
        chains=4,  
        cores=1,      
    )

print(pm.summary(trace, var_names=["x"]))