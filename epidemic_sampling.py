import numpy as np
import json
import matplotlib.pyplot as plt
import covid_funs
import importlib
import subprocess
import sys
import shlex
import os


fldername = "Example-SAIR"
try:
    os.mkdir(fldername)
except:
    None





Bias = 10#Bias = 1 means no bias. Higher puts bias towards testing symptomatic individuals.
#Bias can also be given as a function of time.


capacities = [100,300,900,2700,8100]#These are max or average capacities.
#We can also give a function of time as the capacities. If this function of time is
# of order 1 (and so relative to some maximum or average), we can then test
# for a set of maximums/averages.
def capacityfun(t):#A Hill function
    return t/(1+t)

end_time = 200

num_trials = 10


dyn = "SAIR"


timescale = 15 #1/gamma


init_inf = 0.01

s0 = (1-init_inf)
i0 = init_inf
r0 = 0

# R0 = 2.2/timescale

def R0(X,t):#Can generate dynamics with the SIR or SAIR model,  with time-varying R0
    return (2 - np.exp(-((t-50)/15)**2))/timescale


#We can generate dynamics any way we want as long as we are left with a dictionary with lists which contains
# "TimePoints", "Symptomatic","Asymptomatic","NonInfected"
# as lists


if dyn == "given":
    time = np.arange(0,end_time,0.01)
    Symptomatic = 0.1*np.sin(time/5) + 0.2
    Asymptomatic = np.zeros(len(time))
    NonInfected = 1 - Symptomatic
    total_infected = Symptomatic + Asymptomatic


    dynamics = {"TimePoints":list(time), "Symptomatic":list(Symptomatic),"Asymptomatic":list(Asymptomatic),"NonInfected":list(NonInfected)}

#We provide a function to generate dynamics according to an ODE model. Two models (SIR and SAIR) are also provided.
elif dyn == "SIR":
    dynamics = covid_funs.gen_dynamics(end_time,[s0,i0,r0],covid_funs.SIR_model,1,-1,[0,2],[R0,1/timescale])

elif dyn == "SAIR":

    asympt = 0.7
    bet = 1
    dynamics = covid_funs.gen_dynamics(end_time,[s0,i0*asympt,i0*(1-asympt),r0],covid_funs.SIR_model_asympt,2,1,[0,3],[R0,R0,asympt,1/timescale,bet/timescale])

#gen_jsons is a helper function that will generate the .json input files for the executible in the correct format
# This Bias can be given as a function of time, and there
covid_funs.gen_jsons(fldername,dynamics,Bias,capacities,capacityfun = capacityfun)

total_infected = np.array(dynamics['Symptomatic']) + np.array(dynamics['Asymptomatic'])

svfl = fldername+"/testresults.json"
dynamicsfl = fldername+"/dynamics.json"
biasfl = fldername+"/bias.json"
capfl = fldername+"/capacity.json"
falsePos = 0.1
falseNeg = 0.1
smth = 5
peak_tol = 3
# popsize = 1000

base_command = "./disease_confidence"
opts = ["-Dynamics="+dynamicsfl]
opts +=["-TestingBias="+biasfl]
opts +=["-TestingCapacities="+capfl]
opts +=["-Trials="+str(num_trials)]
opts +=["-SaveFile="+svfl]
opts +=["-FalsePositive="+str(falsePos)]
opts +=["-FalseNegative="+str(falseNeg)]
opts +=["-Smoothing="+str(smth)]
opts +=["-PeakTol="+str(peak_tol)]


full_command = base_command + " " + " ".join(opts)
so = covid_funs.run_command(full_command)

with open(svfl) as fl:
    results = json.load(fl)




pos_prop = {}
for ky in results["SimulatedData"]:
    smps = []
    times = []
    for sample in results["SimulatedData"][ky]:
        smps += [np.array(sample["DailyPositive"])/np.maximum(1,np.array(sample["DailyTotal"]))]
        times += [np.array(sample["DayTimes"])]
    pos_prop[ky] = (smps,times)


for ky in pos_prop.keys():
    fig,ax = plt.subplots(figsize = (10,5))
    ax.plot(dynamics["TimePoints"],total_infected, label = "Total Infection Proportion", color = 'red')
    # ax.plot(dynamics["TimePoints"],dynamics['Symptomatic'], label = "Symptomatic Infection Proportion", color = 'green')
    # ax.plot(dynamics["TimePoints"],dynamics['Asymptomatic'], label = "Asymptomatic Infection Proportion", color = 'yellow')

    ax.bar(pos_prop[ky][1][0],pos_prop[ky][0][0], label = "Positive Test Proportion")
    ax.set_xlabel("Time")
    ax.legend()
    fig.savefig(fldername+"/"+ky)
    plt.close()





five_day_conf = {}
five_day_OverTime = {}
five_day_err = {}
for ky in pos_prop.keys():
    five_day_conf[ky],five_day_OverTime[ky],five_day_err[ky] = covid_funs.trendConfidence(pos_prop[ky][0],pos_prop[ky][1],total_infected,dynamics["TimePoints"],5)

with open(fldername+"/five_day_conf.json","w") as outfile:
    json.dump(five_day_conf, outfile)


with open(fldername+"/five_day_conf_overtime.json","w") as outfile:
    json.dump(five_day_OverTime, outfile)

fig,ax = plt.subplots(figsize = (10,5))
for ky in pos_prop.keys():
    ax.plot(five_day_OverTime[ky][1],five_day_OverTime[ky][0], label = "Samples/Day:" + str(ky))
    # ax[1].plot(five_day_err[ky][1],five_day_err[ky][0], label = "Samples/Day:" + str(ky))



ax.set_ylabel("Five-day confidence")
ax.set_xlabel("Time")
ax.legend()
# ax[1].set_ylabel("Mean-Square Error of Estimated Slope")
# ax[1].set_xlabel("Time")
# ax[1].legend()
fig.savefig(fldername+"/5Dovertime")
plt.close()


poly_correct = {}
poly_correct_overtime = {}
poly_err = {}
exp_err = {}
log_err = {}
avg_deg = {}
for ky in pos_prop.keys():
    poly_correct[ky],poly_correct_overtime[ky],avg_deg[ky],poly_err[ky],exp_err[ky],log_err[ky] = covid_funs.nonlinearConfidence(pos_prop[ky][0],pos_prop[ky][1],total_infected,dynamics["TimePoints"],10,strt = 15)

with open(fldername+"/correct_power.json","w") as outfile:
    json.dump(poly_correct,outfile)

with open(fldername+"/poly_error.json","w") as outfile:
    json.dump(poly_err,outfile)

with open(fldername+"/exp_error.json","w") as outfile:
    json.dump(exp_err,outfile)

with open(fldername+"/log_err.json","w") as outfile:
    json.dump(log_err,outfile)

with open(fldername+"/avg_deg.json","w") as outfile:
    json.dump(avg_deg,outfile)


fig,ax = plt.subplots(figsize = (10,5))
for ky in pos_prop.keys():
    ax.plot(poly_correct_overtime[ky][1],poly_correct_overtime[ky][0], label = "Samples/Day: " + str(ky))
ax.set_ylabel("Polynomial Power Confidence")
ax.set_xlabel("Time")
ax.legend()
fig.savefig(fldername+"/polyconf")
plt.close()

fig,ax = plt.subplots(figsize = (10,5))
for ky in pos_prop.keys():
    ax.plot(avg_deg[ky][1],avg_deg[ky][0], label = "Samples/Day: " + str(ky))
ax.set_ylabel("Average fit polynomial degree")
ax.set_xlabel("Time")
ax.legend()
fig.savefig(fldername+"/avg_deg")
plt.close()

fig,ax = plt.subplots(figsize = (10,5))
for ky in pos_prop.keys():
    ax.plot(poly_err[ky][2],poly_err[ky][0], label = "Samples/Day: " + str(ky))
ax.set_ylabel("Average polynomial fit error")
ax.set_xlabel("Time")
ax.legend()
fig.savefig(fldername+"/polyerr")
plt.close()

fig,ax = plt.subplots(figsize = (10,5))
for ky in pos_prop.keys():
    ax.plot(poly_err[ky][2],poly_err[ky][1], label = "Samples/Day: " + str(ky))
ax.set_ylabel("Polynomial fit error variance")
ax.set_xlabel("Time")
ax.legend()
fig.savefig(fldername+"/polyvar")
plt.close()

fig,ax = plt.subplots(figsize = (10,5))
for ky in pos_prop.keys():
    ax.plot(exp_err[ky][2],exp_err[ky][0], label = "Samples/Day: " + str(ky))
ax.set_ylabel("Average exponential fit error")
ax.set_xlabel("Time")
ax.legend()
fig.savefig(fldername+"/experr")
plt.close()

fig,ax = plt.subplots(figsize = (10,5))
for ky in pos_prop.keys():
    ax.plot(exp_err[ky][2],exp_err[ky][1], label = "Samples/Day: " + str(ky))
ax.set_ylabel("Exponential fit error variance")
ax.set_xlabel("Time")
ax.legend()
fig.savefig(fldername+"/expvar")
plt.close()

fig,ax = plt.subplots(figsize = (10,5))
for ky in pos_prop.keys():
    ax.plot(log_err[ky][2],log_err[ky][0], label = "Samples/Day: " + str(ky))
ax.set_ylabel("Average log fit error")
ax.set_xlabel("Time")
ax.legend()
fig.savefig(fldername+"/logerr")
plt.close()

fig,ax = plt.subplots(figsize = (10,5))
for ky in pos_prop.keys():
    ax.plot(log_err[ky][2],log_err[ky][1], label = "Samples/Day: " + str(ky))
ax.set_ylabel("Log fit error variance")
ax.set_xlabel("Time")
ax.legend()
fig.savefig(fldername+"/logvar")
plt.close()


recall = {}
for ky,val in results["Performance"].items():
    if sum([len(v["Recalls"]) for v in val]):
        recall[ky] = sum([pk["Found"] for dyn in val for  pk in dyn["Recalls"]])/sum([len(v["Recalls"]) for v in val])
    else:
        recall[ky] = 0

with open(fldername+"/recall.json","w") as outfile:
    json.dump(recall, outfile)

precision = {}
for ky,val in results["Performance"].items():
    if sum([len(v["Precisions"]) for v in val]):
        precision[ky] = sum([pk["Real"] for dyn in val for  pk in dyn["Precisions"]])/sum([len(v["Precisions"]) for v in val])
    else:
        precision[ky] = 0

with open(fldername+"/precision.json","w") as outfile:
    json.dump(precision, outfile)

mean_sq_error = {}
for ky,val in results["Performance"].items():
    if sum([len(v["Precisions"]) for v in val]):
        mean_sq_error[ky] =[(sum([(pk["SqDist"])**2 for pk in dyn["Precisions"]])/len(dyn["Precisions"]))**(1/2) for dyn in val]
    else:
        mean_sq_error[ky] = 0

with open(fldername+"/mean_sq_error.json","w") as outfile:
    json.dump(mean_sq_error, outfile)
