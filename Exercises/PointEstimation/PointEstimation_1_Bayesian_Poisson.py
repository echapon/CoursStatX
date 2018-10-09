import ROOT
import math
from scipy import optimize
from array import array

# 1. likelihood
def lh(D, Q, s, b, k):
    if (D<=0 or Q<=0 or s<0 or b<0):
        return 1e99
    return ((s+b)**D * math.exp(-(s+b)) / math.gamma(D+1)) * ((b*k)**Q * math.exp(-b*k) / math.gamma(Q+1))

def lnlh(D, Q, s, b, k):
    l = lh(D, Q, s, b, k)
    if (l<=0):
        return 1e99
    return -2*math.log(l)

def lh_b(x,*args):
    b = x[0]
    s = args[3]
    D = args[0]
    Q = args[1]
    k = args[2]
    return lh(D,Q,s,b,k)

def lnlh_b(x,*args):
    b = x[0]
    s = args[3]
    D = args[0]
    Q = args[1]
    k = args[2]
    return lnlh(D,Q,s,b,k)

# 2. draw the profile likelihood
D = 13
B = 2.6
dB = 0.7

Q = (B/dB)**2
k = B/(dB**2)

s_vals = []
lh_vals = []
proflh_vals = []

# global minimum
bhat = Q/k
shat = D-bhat
lhmax = lh(D,Q,shat,bhat,k)
lnlhmin = lnlh(D,Q,shat,bhat,k)

ds = 0.1
for i_s in range(50,200):
    s = i_s * ds
    bhathat = optimize.fmin(lnlh_b,bhat,args=(D,Q,k,s),disp=False)
    bhathat = bhathat[0]
    s_vals.append(s)
    lh_vals.append(lnlh(D,Q,s,bhat,k)-lnlhmin)
    proflh_vals.append(lnlh(D,Q,s,bhathat,k)-lnlhmin)

glh = ROOT.TGraph(len(s_vals), array('d',s_vals), array('d',lh_vals))
gproflh = ROOT.TGraph(len(s_vals), array('d',s_vals), array('d',proflh_vals))

c1 = ROOT.TCanvas()
gproflh.SetLineColor(ROOT.kRed)
gproflh.Draw("AL")
glh.SetLineColor(ROOT.kBlue)
glh.Draw("L")
c1.Draw()

print ("MLE: shat = %f, bhat = %f" % (shat, bhat))

raw_input("Press Enter to continue...")

# 3. marginalisation

# define priors
def prior_flat_positive(x):
    if (x>=0):
        return 1
    else:
        return 0

def prior_flat(x):
    return 1

def prior_Jeffreys(s,b):
    if (s<0 or b<0): 
        return 0
    return(1/math.sqrt(s+b))

# define posteriors using Bayes' theorem
def posterior_flatprior(D,Q,s,b,k):
    return lh(D,Q,s,b,k)*prior_flat_positive(s)*prior_flat_positive(b)

def posterior_Jeffreys(D,Q,s,b,k):
    return lh(D,Q,s,b,k)*prior_Jeffreys(s,0)*prior_flat_positive(b)

def posterior_Jeffreys_b(D,Q,s,b,k):
    return lh(D,Q,s,b,k)*prior_Jeffreys(s,bhat)*prior_flat_positive(b)

# MCMC
def mcmc(fun, n, D,Q,s0,b0,k):
    vs = [s0,]
    vb = [b0,]

    for i in range(0,n):
        s = ROOT.gRandom.Gaus(s0,4*math.sqrt(s0))
        b = ROOT.gRandom.Gaus(b0,4*math.sqrt(b0))
        alpha = min(1,fun(D,Q,s,b,k)/fun(D,Q,s0,b0,k))
        u = ROOT.gRandom.Uniform(1)
        if (u <= alpha):
            s1 = s
            b1 = b
        else:
            s1 = s0
            b1 = b0
        if (i>1000): # burn-in: ignore the first iterations
            vs.append(s1)
            vb.append(b1)
        s0 = s1
        b0 = b1

    return (vs,vb)

c2 = ROOT.TCanvas()
c2.Divide(2,2)

# 3.a flat prior
vals_flat = mcmc(posterior_flatprior, 100000, D, Q, shat, bhat, k)
gflat = ROOT.TGraph(len(vals_flat[0]), array('d',vals_flat[0]), array('d',vals_flat[1]))
c2.cd(1)
gflat.Draw("APL")

# 3.b flat prior
vals_Jeffreys = mcmc(posterior_Jeffreys, 100000, D, Q, shat, bhat, k)
gJeffreys = ROOT.TGraph(len(vals_Jeffreys[0]), array('d',vals_Jeffreys[0]), array('d',vals_Jeffreys[1]))
c2.cd(2)
gJeffreys.Draw("APL")

# 3.c flat prior
vals_Jeffreys_b = mcmc(posterior_Jeffreys_b, 100000, D, Q, shat, bhat, k)
gJeffreys_b = ROOT.TGraph(len(vals_Jeffreys_b[0]), array('d',vals_Jeffreys_b[0]), array('d',vals_Jeffreys_b[1]))
c2.cd(3)
gJeffreys_b.Draw("APL")

# compare the marginalised posteriors
hflat = ROOT.TH1D("hflat","Flat prior",60,0,30)
hJeffreys = ROOT.TH1D("hJeffreys","Jeffrey's prior (b=0)",60,0,30)
hJeffreys_b = ROOT.TH1D("hJeffreys_b","Jeffrey's prior (b=#hat{b})",60,0,30)

for i in range(0,len(vals_flat[0])):
    hflat.Fill(vals_flat[0][i])
    hJeffreys.Fill(vals_Jeffreys[0][i])
    hJeffreys_b.Fill(vals_Jeffreys_b[0][i])

c2.cd(4)
hflat.DrawNormalized("hist")
hJeffreys.SetLineColor(ROOT.kRed)
hJeffreys.DrawNormalized("hist same")
hJeffreys_b.SetLineColor(ROOT.kBlue)
hJeffreys_b.DrawNormalized("hist same")
c2.Draw()
c2.SaveAs("file.pdf")

print("Flat prior: mean = %f, RMS = %f" % (hflat.GetMean(), hflat.GetRMS()))
print("Jeffrey's prior (b=0): mean = %f, RMS = %f" % (hJeffreys.GetMean(), hJeffreys.GetRMS()))
print("Jeffrey's prior (b=bhat): mean = %f, RMS = %f" % (hJeffreys_b.GetMean(), hJeffreys_b.GetRMS()))
print ("MLE: shat = %f, bhat = %f" % (shat, bhat))

raw_input("Press Enter to continue...")

