#date: 2024-08-28T16:54:46Z
#url: https://api.github.com/gists/e0e23622f18375c45a029868b40423da
#owner: https://api.github.com/users/eggaskin

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
This code provides functions for analyzing textual conversation-format data that is split turn by turn.
The receptive.R script should have been run first on conversation data and saved to utterrecepts (receptiveness of different utterances).
This was first used on the utterances.csv file from the persuasion-for-good dataset.

Interesting and helpful papers for theory of mind and question-based prompting:
https://arxiv.org/pdf/2310.01468
https://arxiv.org/pdf/2310.03051
https://aclanthology.org/2023.conll-1.25.pdf
https://arxiv.org/ftp/arxiv/papers/2309/2309.01660.pdf
https://arxiv.org/pdf/2302.02083

"""

def getconvtext(speaker,utterances,casino=True):
    # DIALOGUE ID is the conv number
    utterances = utterances[utterances['speaker'] == speaker] # first or second speaker? does it matter?
    # get unique dialogue id vals
    uniqueconv = utterances['dialogue_id'].unique()
    sents = []
    for i in uniqueconv:
        convid = i
        speakerconvtext = utterances.loc[(utterances['dialogue_id'] == convid) & (utterances['speaker'] == speaker)]
        # join all sentences in text column
        sents.append((convid,speakerconvtext['text'].tolist()))
    return sents

# other csv's that can be helpful for analysis
#charityppl = pd.read_csv('data/full_info.csv') # full info on each speaker
allfeats = pd.read_csv('testout.csv') # full output of receptiveness script if testing binary variables
qualdf = pd.read_csv('qualsort.csv') # sorted based on magnitude of some speaker quality

def shift(one,two,avgforward=None):
    # helps shifts conversation arrays to be the same length so we are comparing responses in order.
    
    if avgforward:
        if len(one) > len(two):
            # make sure to also shift down one
            one,two= one[:len(two)],two
        elif len(one) < len(two):
            one,two= one,two[:len(one)]
        
        # get the average receptive over next avgforward utterances for each utterance
        lens = len(one)-avgforward
        avg1 = []
        avg2 = []
        for i in range(lens):
            avg1.append(np.mean(two[i:i+avgforward]))
            avg2.append(np.mean(one[i:i+avgforward]))
        return avg1,avg2

    if len(one) > len(two):
        # make sure to also shift down one
        return one[:len(two)],two
    elif len(one) < len(two):
        return one,two[:len(one)]
    elif len(one) == len(two):
        return one,two

qualities = ["agreeableness_y","openness-to-experiences_x","extraversion","emotional-stability","conscientiousness"]

def receptive_v_qual(quality): 
    # plots for each level of a quality (1-7), the spread of how receptive speaker's utterances are.
    
    quals = [qualdf.loc[qualdf['speaker_id']==sp][quality] for sp in allfeats['speaker'].tolist()]
    plt.plot(allfeats['receptive'],quals,'o')
    plt.xlabel('Receptiveness')
    plt.ylabel(quality)
    plt.show()

# the main dataframe with the receptiveness scores.
ur = pd.read_csv('utterrecepts.csv')
# drop annotations col
ur = ur.drop(columns=['annotations', "speaker_internal_id","id","reply.to"])
conversations = ur['conversation_id'].unique()

def plot_1v2_recept(convs): 
    # plots the correlation of the 1st vs 2nd speaker.
    # this mostly serves to obtain general correlation scores for general analysis.

    convdict = {}
    x = np.array([])
    y = np.array([])

    indcorrs = []
    spekrs = []

    for c in convs:
        conv = ur.loc[ur['conversation_id'] == c]
        spkrs = conv['speaker'].unique()
        spekrs.append(spkrs)
        
        recs_speaker_0 = conv.loc[conv['speaker_id'] == spkrs[0], 'receptive'][:-1]
        recs_speaker_1 = conv.loc[conv['speaker_id'] == spkrs[1], 'receptive'][:-1]

        recs_speaker_0,recs_speaker_1 = shift(recs_speaker_0,recs_speaker_1,avgforward=2)
        convdict[c] = (np.array(recs_speaker_0),np.array(recs_speaker_1))
        x = np.concatenate((x,recs_speaker_0))
        y = np.concatenate((y,recs_speaker_1))

        indcorrs.append(np.corrcoef(np.array(recs_speaker_0),np.array(recs_speaker_1))[0][1])

        if len(recs_speaker_1) != len(recs_speaker_0):
            print(recs_speaker_1,recs_speaker_0)
            print(len(recs_speaker_1),len(recs_speaker_0))
            break

        # Plot the difference in derivatives
        plt.scatter(recs_speaker_0,recs_speaker_1,alpha=0.2)
        plt.xlabel('1st speaker receptiveness')
        plt.ylabel('2nd speaker receptiveness')

    plt.show()
    return indcorrs

indcorrs = plot_1v2_recept(conversations)

# kde plot to show the distirbution of correlation in conversations. 
# Typically a centered bi-modal distribution
sns.kdeplot(indcorrs,fill=True)
plt.title('correlation per conversation distribution')

def receptive_ex_case(): # a case study to show how the receptive difference works
    # for utterance_0 conversation_id, plot receptiveness of both speakers over conversation
    conv = ur.loc[ur['conversation_id'] == 'utterance_0']

    # Get the receptive scores for each speaker
    # unique speaker vals
    spkrs = conv['speaker'].unique()
    recs_speaker_0 = conv.loc[conv['speaker_id'] == spkrs[0], 'receptive'][:-1]
    recs_speaker_1 = conv.loc[conv['speaker_id'] == spkrs[1], 'receptive'][:-1]

    # Plot the lines
    plt.plot(recs_speaker_0, label='Speaker 0')
    plt.plot(recs_speaker_1, label='Speaker 1')

    # Add labels and legend
    plt.xlabel('Turn')
    plt.ylabel('Receptive Score')
    plt.legend()

    # Show the plot
    plt.show()

    difference = np.array(recs_speaker_0) - np.array(recs_speaker_1)
    return difference

difference = receptive_ex_case() # this will represent the rec.diff. in a conversation that is very correlated.

def plot_recept_overtime(convs): # 
    """
    This will plot the difference in derivatives of the two speaker's receptiveness scores,
    meaning the general trend of the speaker's changes in receptiveness. 
    Meaning, if they are changing in tandem or similarly this graph will be low in magnitude.

    For conversations whose speakers do follow each other closely in receptiveness as shown in the example above, 
    this will be a small value close to 0-0.2.
    """
    convdict = {}
    grads = []

    for c in convs:
        conv = ur.loc[ur['conversation_id'] == c]
        spkrs = conv['speaker'].unique()
        recs_speaker_0 = conv.loc[conv['speaker_id'] == spkrs[0], 'receptive'][:-1]
        recs_speaker_1 = conv.loc[conv['speaker_id'] == spkrs[1], 'receptive'][:-1]

        convdict[c] = (np.array(recs_speaker_0),np.array(recs_speaker_1))

        minlen = min(len(recs_speaker_0),len(recs_speaker_1))
        # chop both to that size
        recs_speaker_0 = recs_speaker_0[:minlen]
        recs_speaker_1 = recs_speaker_1[:minlen]

        grads.append(np.array(recs_speaker_0) - np.array(recs_speaker_1))

        # Plot the difference in derivatives
        plt.plot(abs(np.array(recs_speaker_0) - np.array(recs_speaker_1)),alpha=0.05)
        plt.xlabel('Turn')
        plt.ylabel('Diff of receptiveness plain')

    plt.plot(abs(difference),alpha=1,color='black')
    plt.show()

# a low score for the plot above means that the two speakers receptiveness follow each other closely
# turn by turn, so one changes and the other also changes accordingly.
