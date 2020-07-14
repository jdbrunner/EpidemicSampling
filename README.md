# EpidemicSampling
Generate simulated sampling data from an epidemic of known dynamics.

Given an epidemic with exact known dynamics as H(t), I1(t), I2(t) representing the number of individuals who are healthy, infected but asymptomatic,
and infected and symptomatic, this tool simulates daily biased sampling of the epidemic spread using a stochastic model. Sampling is modeled as a 
Poisson point process, and a test result is drawn from a discrete distribution based on H(t), I1(t), I2(t) and some bias and error in the testing.

This tool requires go, as well as the python libraries:
numpy
json
matplotlib.pyplot
covid_funs
importlib
subprocess
sys
shlex
os

To use, simply git clone this repository. Next use, go build disease_confidence.go to compile the executible. The included Jupyter notebook provides
a tutorial for using the tool.


The main function of the package is an executible named disease_confidence written in GoLang which generates the sample data. This function also provides the option to estimate
peaks in the simulated data. (may need to be rebuilt with go build disease_confidence.go)

This program requires dynamics, bias, and testing capacities to by input as .json files.

The following lists the options for this executible (Flag, default, description)
-Verbose,false,"whether or not to print all trials in detail."

-ComputePeaks,true,"whether or not to compute peak predictions."

-SaveFile,"json_io/out","name of save file"

-Dynamics,"",".json file with dynamics saved."

-TestingBias,"",".json file with testing bias saved."

-TestingCapacities,"",".json file with testing capacities saved."

-FalsePositive,0,"False positive test rate"

-FalseNegative,0,"False negative test rate"

-PeakTol,2,"Tolerance for peak prediction."

-Interval,1,"Length of data bucket intervals"

-Smoothing,5,"Guassian smoothing parameter for derivative estimation"

-TotalPop,0,"Total population - leave 0 for infinite population/immediate retesting"

-RetestRate, 1, "Rate of eligibility for retesting (1/time)"

-Trials,1,"Number of trials for estimation")

Also provided are python scripts which make it easy to generate the necessary .json files, and analyse the output of the executible.
