
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
palette = 'muted'
sns.set_palette("summer"); sns.set_color_codes(palette)

def posterior(grid_points=100, heads=5, tosses=20): 
    #Defining a grid for the coin flip problem. 
    #The underlying assumption is that we make 20 tosses and we observe 5 heads.
    
    grid = np.linspace(0, 1, grid_points)

    prior = np.repeat(5, grid_points)  
   

    likelihood = stats.binom.pmf(heads, tosses, grid)

    unstd_posterior = likelihood * prior

    posterior = unstd_posterior / unstd_posterior.sum()
    return grid, posterior

points = 15
heads, tosses = 5, 20 # We make 20 tosses and observe 5 heads.  
grid, posterior = posterior(points, heads, tosses)
plt.plot(grid, posterior, 'o-')
plt.plot(0, 0, label='heads = {}\ntosses = {}'.format(heads, tosses), alpha=0)
plt.xlabel(r'$\theta$', fontsize=12)
plt.legend(loc=0, fontsize=12)
plt.savefig('B04958_02_01.png', dpi=300, figsize=(5.5, 5.5));

