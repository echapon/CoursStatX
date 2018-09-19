#!/bin/python

# imports
from scipy import optimize, math
import numpy
import ROOT
from array import array

###############
# DEFINITIONS #
###############

# Poisson prob
def prob_poisson(n,mu):
    if mu<=0 or n<0: 
        return 0
    else:
        p = 1
        for i in range(0,len(n)):
            nn = int(n[i])
            p *= (mu**nn)*math.exp(-mu)/math.factorial(nn) 
        return p

# Poisson LH
def lh_poisson(mu, *args):
    n = args[0]
    return -2*numpy.log(prob_poisson(n,mu))

################
# MAIN PROGRAM #
################

# constants
mu0=3.5
ntrials=1000

# lists to be used to plot the graphs
ntoys = []
dntoys = []
means = []
dmeans = []
variances = []
dvariances = []
biases = []

rnd = ROOT.TRandom3()

# loop on the number of toys
for nt in [1,2,3,4,5,6,7,8,9,10]:
    ntoys.append(nt)
    dntoys.append(0)

    # 1000 trials
    estimates = []
    for i in range(0,ntrials):
        # toy generation
        toys = []
        for it in range(0,nt):
            toys.append(rnd.Poisson(mu0))

        # LH estimate
        result = optimize.fmin(lh_poisson,mu0,args=(toys,),disp=False)
        estimates.append(result)

    # compute the mean, variance, bias
    means.append(numpy.mean(estimates))
    variances.append(numpy.var(estimates))
    biases.append(numpy.mean(estimates)-mu0)
    dmeans.append(math.sqrt(variances[-1]/ntrials))
    dvariances.append(math.sqrt(2*variances[-1]**2/(ntrials-1)))

# plot results
nx = len(ntoys)
c1 = ROOT.TCanvas("c1","Mean")
gmean = ROOT.TGraphErrors(nx,array('d',ntoys),array('d',means),array('d',dntoys),array('d',dmeans))
gmean.Draw("ALP")
c1.Draw()
c1.SaveAs("c1.pdf")

c2 = ROOT.TCanvas("c2","Variance")
gvar = ROOT.TGraphErrors(nx,array('d',ntoys),array('d',variances),array('d',dntoys),array('d',dvariances))
gvar.Draw("ALP")
c2.Draw()
c2.SaveAs("c2.pdf")

c3 = ROOT.TCanvas("c3","Biases")
gbias = ROOT.TGraphErrors(nx,array('d',ntoys),array('d',biases),array('d',dntoys),array('d',dmeans))
gbias.Draw("ALP")
c3.Draw()
c3.SaveAs("c3.pdf")
