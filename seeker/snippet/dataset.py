#date: 2022-03-08T16:52:23Z
#url: https://api.github.com/gists/8e2e96d22e60ed5e810116177422c330
#owner: https://api.github.com/users/vwxyzjn

        trace_names = [
            'Subject_54_Trial_128.csv',
            'Subject_45_Trial_100.csv',
            'Subject_30_Trial_57.csv',
            'Subject_88_Trial_230.csv',
            'Subject_51_Trial_120.csv',
            'Subject_33_Trial_66.csv',
            'Subject_50_Trial_116.csv',
            'Subject_70_Trial_175.csv',
            'Subject_100_Trial_267.csv',
            'Subject_60_Trial_145.csv',
            'Subject_34_Trial_69.csv',
            'Subject_98_Trial_261.csv',
            'Subject_36_Trial_75.csv',
            'Subject_78_Trial_199.csv',
            'Subject_32_Trial_61.csv',
        ]
    else:
        trace_names = list(dataset.keys())[:args.num_traces]
    training_evaluation_trace = "Subject_32_Trial_61.csv"
    testing_evaluation_trace = "Subject_96_Trial_254.csv"
