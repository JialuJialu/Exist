import matplotlib.pyplot as plt
import csv
import numpy as np

usedtime = open('invariants/used_time_alt_init', 'r')
usedtimelst = []
for line in usedtime:
    usedtimelst.append(line.strip().split(",")[:4])


totaltime = [float(row[1])/60 for row in usedtimelst]
sampletime_per = [float(row[2])/float(row[1]) for row in usedtimelst]

n, bins, patches = plt.hist(x=totaltime, bins='auto', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Time(minutes)')
plt.ylabel('Number of Benchmarks')
plt.savefig("total_time_histogram.png")

n, bins, patches = plt.hist(x=sampletime_per, bins='auto', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Percentage time for sampling')
plt.ylabel('Number of Benchmarks')
plt.savefig("sampling_perc_histogram.png")
