#!/bin/python

from scipy import optimize

def func(x, *args):
    mu = args[0]
    return (x-mu)**2

x0=2
mu=5
result = optimize.minimize_scalar(func,args=(mu))
print(result.x)

result = optimize.minimize(func, x0, method="CG", args=(mu))
print(result.x)

thelist = [mu]
print (len(thelist))
