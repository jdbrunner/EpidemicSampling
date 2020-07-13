import numpy as np
from scipy.integrate import cumtrapz
from scipy.integrate import ode
from numpy.random import exponential as exp
from numpy.random import rand

import subprocess
import sys
import shlex
# from scipy.ndimage import gaussian_filter1d as smooth
import json

import matplotlib.pyplot as plt



import matplotlib.pyplot as plt

def gen_dynamics(end_time,initial_condtions,dynamics,symptomaticVar,asymptomaticVar,noninfectedVar,params,tst = 0.01):

    sir_sol = ode(dynamics)
    sir_sol.set_f_params(params)
    sir_sol.set_initial_value(initial_condtions,0)

    time_array = np.array([0])
    sol_list = [list(initial_condtions)]
    for tpt in np.arange(0,end_time,tst):
        solt = list(sir_sol.integrate(sir_sol.t + tst))
        sol_list = sol_list + [solt]
        time_array = np.append(time_array,sir_sol.t)
    sol_array = np.array(sol_list)

    if not np.isscalar(symptomaticVar):
        Symptomatic = np.sum(sol_array[:,symptomaticVar],axis = 1)
    else:
        Symptomatic = sol_array[:,symptomaticVar]

    if not np.isscalar(asymptomaticVar):
        Asymptomatic = np.sum(sol_array[:,asymptomaticVar],axis = 1)
    else:
        if asymptomaticVar != -1:
            Asymptomatic = sol_array[:,asymptomaticVar]
        else:
            Asymptomatic = np.zeros(len(time_array))

    if not np.isscalar(noninfectedVar):
        NonInfected = np.sum(sol_array[:,noninfectedVar],axis = 1)
    else:
        NonInfected = sol_array[:,noninfectedVar]

    dynamic_map = {"TimePoints":list(time_array), "Symptomatic":list(Symptomatic),"Asymptomatic":list(Asymptomatic),"NonInfected":list(NonInfected)}

    return dynamic_map

def gen_jsons(folder,dynamic_map,Bias,maxcapacity,capacityfun = False):

    time_array = dynamic_map["TimePoints"]

    if callable(Bias):
        biasarr = [Bias(t) for t in time_array]
    elif np.isscalar(Bias):
        biasarr = [Bias]*len(time_array)
    else:
        biasarr = Bias

    if np.isscalar(maxcapacity):
        maxcapacity = [maxcapacity]

    if callable(capacityfun):
        capacity_map = {}
        for i in maxcapacity:
            capacity_map[str(i)] = [i*capacityfun(t) for t in time_array]
    else:
        capacity_map = {}
        for i in maxcapacity:
            capacity_map[str(i)] = [i]*len(time_array)


    with open(folder+"/dynamics.json","w") as outfile:
        json.dump(dynamic_map, outfile)

    with open(folder+"/bias.json","w") as outfile:
        json.dump(biasarr,outfile)

    with open(folder+"/capacity.json","w") as outfile:
        json.dump(capacity_map,outfile)

    return None



def fit_slope(full_y,window,*argv):

    y = full_y[window[0]:window[1]]

    if len(argv):
        x = argv[0][window[0]:window[1]]
    else:
        x = np.arange(len(y))


    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    return m

def poly_fitting(full_y,window,max_pow,*argv):

    y = np.array(full_y[window[0]:window[1]])

    if len(argv):
        x = np.array(argv[0][window[0]:window[1]])
    else:
        x = np.ones(len(y))

    coeffs,err,_,_,_ = np.polyfit(x,y,1,full = True)

    for i in np.arange(2,max_pow):
        coeffstmp,errtmp,_,_,_ = np.polyfit(x,y,i,full = True)
        if errtmp < 0.9*err:
            err = errtmp
            coeffs = coeffstmp
    # polys = np.polyfit(x,y,max_pow)
    #
    # totvar = abs(y[-1]-y[0])
    #
    # rel_pow = 0
    #
    # for i in range(max_pow):
    #     var = abs(polys[i]*(x[-1]**(max_pow-i) - x[0]**(max_pow-i)))
    #     if var/totvar > 0.25:
    #         rel_pow = max_pow - i
    #         break

    # coeffs = polys[max_pow-rel_pow:]
    rel_pow = len(coeffs) - 1

    fitted = sum([coeffs[i]*x**(rel_pow-i) for i in range(len(coeffs))])
    err = sum((y-fitted)**2)

    return coeffs,err


def exp_fitting(full_y,window,*argv):
    y = np.array(full_y[window[0]:window[1]])

    if len(argv):
        x = np.array(argv[0][window[0]:window[1]])
    else:
        x = np.ones(len(y))

    expCs = np.polyfit(x,np.log(y),1)
    expEr = sum((y - np.exp(expCs[1])*np.exp(expCs[0]*x))**2)

    return expCs,expEr

def log_fitting(full_y,window,*argv):
    y = np.array(full_y[window[0]:window[1]])

    if len(argv):
        x = np.array(argv[0][window[0]:window[1]])
    else:
        x = np.ones(len(y))

    if x[0] > 0:
        logCs = np.polyfit(np.log(x),y,1)
        logEr = sum((y - (logCs[0]*np.log(x) + logCs[1]))**2)
    else:
        logCs = np.array([0,0])
        logEr = 1000

    return logCs,logEr


def nonlinear_compare(poly_coeffs,exp_coeffs,log_coeffs,realvals,realtimes):
    #First - error of fitted poly
    pow = len(poly_coeffs) - 1
    fittedp = sum([poly_coeffs[i]*realtimes**(pow-i) for i in range(len(poly_coeffs))])
    errp = sum((realvals - fittedp)**2)/len(realvals)

    #next - error of fitted exponential
    fittede = np.exp(exp_coeffs[1])*np.exp(exp_coeffs[0]*realtimes)
    erre = sum((realvals - fittede)**2)/len(realvals)

    #and error of fitted log
    if realtimes[0]>0:
        fittedl = log_coeffs[0]*np.log(realtimes) + log_coeffs[1]
        errl = sum((realvals - fittedl)**2)/len(realvals)
    else:
        errl = -1

    return errp,erre,errl


def find_nonlinear(data,times,realvals,realtimes,twindow):
    data = np.array(data)
    times = np.array(times)
    realvals = np.array(realvals)
    realtimes = np.array(realtimes)
    #get index
    if twindow[1] < times[-1]:
        indx1 = np.argwhere(twindow[0] <= times)[0,0]
        indx2 = np.argwhere(twindow[1] <= times)[0,0]
    elif twindow[0]<times[-1]:
        indx1 = np.argwhere(twindow[0] <= times)[0,0]
        indx2 = len(times)
    else:
        print("No data in window")
        return None

    poly = poly_fitting(data,[indx1,indx2],4,times)
    exp = exp_fitting(data,[indx1,indx2],times)
    log = log_fitting(data,[indx1,indx2],times)

    if twindow[1] < realtimes[-1]:
        rindx1 = np.argwhere(twindow[0] <= realtimes)[0,0]
        rindx2 = np.argwhere(twindow[1] <= realtimes)[0,0]
    elif twindow[0]<realtimes[-1]:
        rindx1 = np.argwhere(twindow[0] <= realtimes)[0,0]
        rindx2 = len(realtimes)
    else:
        print("No dynamics in window")
        return None

    polyerr,experr,logerr = nonlinear_compare(poly[0],exp[0],log[0],realvals[rindx1:rindx2],realtimes[rindx1:rindx2])

    return len(poly[0]) - 1,polyerr,experr,logerr


def find_trend(data,times,twindow):
    data = np.array(data)
    times = np.array(times)
    #get index
    if twindow[1] < times[-1]:
        indx1 = np.argwhere(twindow[0] <= times)[0,0]
        indx2 = np.argwhere(twindow[1] <= times)[0,0]
    elif twindow[0]<times[-1]:
        indx1 = np.argwhere(twindow[0] <= times)[0,0]
        indx2 = len(times)
    else:
        print("No data in window")
        return None

    trend = fit_slope(data,[indx1,indx2],times)

    return trend


def test_all_windows(data,times,windowsize,start = 0):
    window_trend = []
    window_ending = []

    t = start

    while t<times[-windowsize]:
        window_trend += [find_trend(data,times,[t,t+windowsize])]
        window_ending += [t+windowsize]
        t += 1

    return np.array(window_trend),window_ending

def test_independent_windows(data,times,windowsize,start = 0):
    window_trend = []
    window_ending = []

    t = start

    while t<times[-1]:
        window_trend += [find_trend(data,times,[t,t+windowsize])]
        window_ending += [t+windowsize]
        t += windowsize

    return np.array(window_trend),np.array(window_ending)

def trendError(realTrends,sampleTrends):
    realTrends = realTrends[:len(sampleTrends)]
    sq_error = (np.array(realTrends) - np.array(sampleTrends))**2
    rt_mn_sq_error =np.sqrt(sum(sq_error)/len(realTrends))
    prod = np.array(realTrends)*np.array(sampleTrends)
    same_sign = [p > 0 for p in prod]
    same_sign_prob = sum(same_sign)/len(realTrends)
    return sq_error,rt_mn_sq_error,same_sign_prob,same_sign

def test_all_nonlinear(data,times,realtimes,realvals,windowsize,start = 0):


    t = start
    poly_powers = []
    poly_errs = []
    exps_errs = []
    logs_errs = []
    window_ending = []

    while t<times[-1]:
        polypow,poly,exp,log = find_nonlinear(data,times,realtimes,realvals,[t,t+windowsize])


        poly_powers += [polypow]
        poly_errs += [poly]
        exps_errs += [exp]
        logs_errs += [log]


        window_ending += [t+windowsize]
        t += 1

    return poly_powers,poly_errs,exps_errs,logs_errs, window_ending

def nonlinearConfidence(samplevals,sampletimes,realvals,realtimes,windowsize,strt = 0):
    real_nonlinear = test_all_nonlinear(realvals,realtimes,realvals,realtimes,windowsize)[0]
    poly_errors = []
    exp_errors = []
    log_errors = []
    poly_correct = []
    polydeg = []

    for i in range(len(samplevals)):
        poly_powers,poly_errs,exps_errs,logs_errs,window_end = test_all_nonlinear(samplevals[i],sampletimes[i],realvals,realtimes,windowsize,start = strt)
        real_trends_temp = real_nonlinear.copy()[:len(poly_powers)]
        poly_errors += [poly_errs]
        exp_errors += [exps_errs]
        log_errors += [logs_errs]
        polydeg += [poly_powers]
        poly_correct +=[[real_trends_temp[j]==poly_powers[j] for j in range(len(poly_powers))]]

    poly_correct_prop = np.sum(poly_correct,axis = 0)/len(samplevals)
    poly_correct_tot = sum(poly_correct_prop)/len(window_end)
    polyerSum = np.sum(poly_errors,axis = 0)/len(samplevals)
    polyerVar = np.sum((np.array(poly_errors)-polyerSum)**2,axis = 0)/(len(samplevals) - 1)

    experSum = np.sum(exp_errors,axis = 0)/len(samplevals)
    experVar = np.sum((np.array(exp_errors)-experSum)**2,axis = 0)/(len(samplevals) - 1)

    logerSum = np.sum(log_errors,axis = 0)/len(samplevals)
    logerVar = np.sum((np.array(log_errors)-logerSum)**2,axis = 0)/(len(samplevals) - 1)

    avg_deg = list(np.sum(polydeg,axis = 0).astype(float)/len(samplevals))

    return poly_correct_tot,(list(poly_correct_prop),list(window_end)),(list(avg_deg),list(window_end)),(list(polyerSum),list(polyerVar),list(window_end)),(list(experSum),list(experVar),list(window_end)),(list(logerSum),list(logerVar),list(window_end))



def trendConfidence(samplevals,sampletimes,realvals,realtimes,windowsize):
    real_trend,_ = test_all_windows(realvals,realtimes,windowsize)
    tot = 0
    confTFarr = []
    sqrerrArr  = []
    for i in range(len(samplevals)):
        samp_trend,window_end = test_all_windows(samplevals[i],sampletimes[i],windowsize)
        sqerr,_,conf,confTF = trendError(real_trend,samp_trend)
        tot += conf
        confTFarr += [confTF]
        sqrerrArr += [sqerr]
    confTFarr = np.array(confTFarr).astype(float)
    sqrerrArr = np.array(sqrerrArr)
    confTFsum = np.sum(confTFarr, axis = 0)
    sqrerrSum = np.sum(sqrerrArr, axis = 0)
    return tot/len(samplevals),(list(confTFsum/len(samplevals)),window_end),(list(sqrerrSum/len(samplevals)),window_end)

def trendConfidenceInd(samplevals,sampletimes,realvals,realtimes,windowsize):
    real_trend,_ = test_independent_windows(realvals,realtimes,windowsize)
    samp_trend,_ = test_independent_windows(samplevals,sampletimes,windowsize)
    sqr,_,conf,_ = trendError(real_trend,samp_trend)
    return conf,sqr







#from https://www.endpoint.com/blog/2015/01/28/getting-realtime-output-using-python
def run_command(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr = subprocess.STDOUT, text = False)
    outlist = []
    while True:
        output = process.stdout.readline()
        outlist += [output]
        if output.decode("utf-8") == '' and process.poll() is not None:
            break
        if output:
            if '\r' in output.decode("utf-8"):

                sys.stdout.write('\r' + output.decode("utf-8").strip())
            else:
                sys.stdout.write(output.decode("utf-8"))
    rc = process.poll()
    return rc,outlist

def SIR_model(t,X,params):
    s,i,r= X
    beta,recrate = params
    if callable(beta):
        dsdt = -(beta(X,t))*s*i
    else:
        dsdt = -beta*s*i
    didt = -dsdt - recrate*i
    drdt = recrate*i
    return [dsdt,didt,drdt]


def SIR_model_asympt(t,X,params):
    R01f,R02f,asympt,g,bet = params
    if callable(R01f):
        R01 = R01f(X,t)
    else:
        R01 = R01f
    if callable(R02f):
        R02 = R02f(X,t)
    else:
        R02 = R02f
    r01 = R01*asympt
    r02 = R01*(1-asympt)
    r03 = R02*asympt
    r04 = R02*(1-asympt)
    s,i1,i2,r= X
    dsdt = -(r01+r02)*s*i1 - (r03 + r04)*s*i2
    di1dt = r01*s*i1 + r03*s*i2 - g*i1 - bet*i1
    di2dt = r02*s*i1 + r04*s*i2 - g*i2 + bet*i1
    drdt = g*i1 + g*i2
    return [dsdt,di1dt,di2dt,drdt]
