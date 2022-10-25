#date: 2022-10-25T17:48:06Z
#url: https://api.github.com/gists/63c14d02b0f787aced1fdaa1e06fdca8
#owner: https://api.github.com/users/NicerNewerCar

import os,sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

if __name__ == "__main__":
    # read in two csv files into dataframes
    if len(sys.argv) != 3:
        print("Usage: py analyize-times.py <CUDA file> <OpenCL file>")
        sys.exit(1)
    df1 = pd.read_csv(sys.argv[1])
    df2 = pd.read_csv(sys.argv[2])

    # set the column name to time
    df1.columns = ["time"]
    df2.columns = ["time"]

    # make sure they have the same number of rows
    if len(df1) != len(df2):
        print("Files are not the same length")
        print("cuda: ", len(df1))
        print("opencl: ", len(df2))
        sys.exit(1)

    # compute the mean, median, and standard deviation of the two dataframes
    df1_avg = df1.mean(axis=0)
    df1_med = df1.median(axis=0)
    df1_std = df1.std(axis=0)
    df2_avg = df2.mean(axis=0)
    df2_med = df2.median(axis=0)
    df2_std = df2.std(axis=0)

    # print the specs of this computer
    print("CPU: 11th Gen Intel(R) Core(TM) i7-11850H @ 2.50GHz")
    print("GPU: NVIDIA RTX A2000 Laptop GPU")
    print("RAM: 32GB")
    print("OS: Windows 11 Pro 64-bit")
    print("")

    print(f"{len(df1)} tracks were preformed for each CUDA and OpenCL\n")

    # compare the two dataframes
    print("Average")
    print("CUDA: ",df1_avg.array[0])
    print("OpenCL: ",df2_avg.array[0])
    print("Median")
    print("CUDA: ",df1_med.array[0])
    print("OpenCL: ",df2_med.array[0])
    print("Standard Deviation")
    print("CUDA: ",df1_std.array[0])
    print("OpenCL: ",df2_std.array[0])
    print("")

    # preform t test
    print("Welch's T Test:")
    ttest = stats.ttest_ind(df1, df2, axis=0, equal_var=False)
    stat = ttest.statistic[0]
    print("t-statistic: ", stat)
    pvalue = ttest.pvalue[0]
    print("p-value: ", pvalue)
    if pvalue < 0.05:
        print("There is significant difference between CUDA and OpenCL")
    else:
       print("There is no significant difference between CUDA and OpenCL")

    # plot the two dataframes
    df1.hist(grid=False, bins=20, rwidth=0.9, color='#607c8e')
    plt.title('CUDA Histogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    df2.hist(grid=False, bins=20, rwidth=0.9, color='#607c8e')
    plt.title('OpenCL Histogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    plt.show()
