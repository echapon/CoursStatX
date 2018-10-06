# imports
from scipy import optimize, math
import numpy
import ROOT
from array import array


ROOT.gStyle.SetOptStat(1100)
ROOT.gStyle.SetOptTitle(0)

###############
# DEFINITIONS #
###############

# Poisson prob
def prob_poisson(n,mu):
    if mu<=0 or n<0: 
        return 0
    else:
        p = 1.0
        for i in range(0,len(n)):
            nn = int(n[i])
            p = p * (mu**nn)*math.exp(-mu)/math.factorial(nn) if nn>0 else 0
        return p

# Poisson LH
def lh_poisson(mu, *args):
    n = args[0]
    lh = prob_poisson(n,mu)
    if lh>0:
        return -2.0*math.log(lh)
    else:
        return 1e99
    
# exp prob
def prob_exp(t,tau):
    if tau<=0:
        return 0
    else:
        p = 1.0
        for i in range(0,len(t)):
            tt = t[i]
            p = p * (1/tau)*math.exp(-tt/tau) if tt>=0 else 0
        return p

# exp LH
def lh_exp(tau, *args):
    t = args[0]
    lh = prob_exp(t,tau)
    if (lh>0):
        return -2.0*math.log(lh)
    else:
        return 1e99

# binned chi2, expected uncertainties
def chi2_exp(tau, *args):
    if (tau<=0):
        return 1e99
    hist = args[0]
    q = 0
    #  print(tau)
    ntot = hist.GetEntries()
    for i in range(1,hist.GetNbinsX()+1):
        bc = hist.GetBinCenter(i)
        width = hist.GetBinWidth(i)
        tmin = bc-width/2.
        tmax = bc+width/2.
        fi = (math.exp(-tmin/tau)-math.exp(-tmax/tau))*ntot
        if fi>0:
            di = hist.GetBinContent(i)
            q = q + (di-fi)**2/fi**2
            #  print(di,fi)
    return q


##############
# PARAMETERS #
##############

mu0=3.5
tau0=0.5
ntrials=100

################
# MAIN PROGRAM #
################


# derived constants
xmin=0
xmax=5*tau0
nbins=10

# lists to be used to plot the graphs
ntoys = []
dntoys = []

# list of methods
means = {}
variances = {}
methods = ["moments","MLE","chi2"]

for meth in methods:
    means[meth] = 0
    variances[meth] = 0

rnd = ROOT.TRandom3()

nt = 100
ntoys.append(nt)
dntoys.append(0)

# trials
estimates = {}
for meth in methods:
    estimates[meth] = []
    
for i in range(0,ntrials):
    # toy generation
    toys = []
    h = ROOT.TH1F("h","",10,0,10)
    for it in range(0,nt):
        toy = rnd.Exp(tau0)
        toys.append(toy)
        h.Fill(toy)

    for meth in methods:
        result=0
        if meth=="MLE":    
            # maximum likelihood estimate
            result = optimize.fmin(lh_exp,tau0,args=(toys,),disp=False)
        elif meth=="moments":
            result = numpy.mean(toys)
        elif meth=="chi2":
            result = optimize.fmin(chi2_exp,tau0,args=(h,),disp=False)
            
        #  print(meth,result)
        estimates[meth].append(result)
        
    del h

            
# check the mean and variance of the estimator for the different methods            
for meth in methods:
    means[meth] = numpy.mean(estimates[meth])
    variances[meth] = numpy.var(estimates[meth])
    print ("mean for method %s: %f" % (meth , means[meth]))
    print ("variance for method %s: %f" % (meth, variances[meth]))



# draw the distribition of estimator values
c = ROOT.TCanvas("c0","estimators")
c.Divide(len(methods))
i=0
h = []
for meth in methods:
    i=i+1
    c.cd(i)
    h.append(ROOT.TH1F("h%d" % i, "",nbins,xmin,xmax))
    for j in range(0,len(estimates[meth])):
        h[-1].Fill(estimates[meth][j])
    h[-1].Draw()
c.Draw()


#  raw_input("Press Enter to continue ...")
