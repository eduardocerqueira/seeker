#date: 2022-03-21T17:10:47Z
#url: https://api.github.com/gists/0a9e795710f82f629c83bab56ed853ba
#owner: https://api.github.com/users/parthvishah


"""
df - dataframe
pred - predicited behavior column (ex: "behavior:sexual")
behavior - actual behavior column (ex: "label:sexual")
prod_pred - production behavior column (ex: "prod-behavior:sexual")
prod_behavior - actual behavior column for production metrics (ex: "label:sexual")
CONFIDENCE - confidence column for the behavior (ex: "confidence:sexual")

Mostly behavior and prod_behavior columns will be the same
"""

def print_metrics(df, pred, behavior, prod_pred, prod_behavior, CONFIDENCE):

    df['high_conf'] = np.where(df[CONFIDENCE] == "High",1,0)
    df['low_conf'] = np.where(df[CONFIDENCE] == "Low",1,0)

    print("Prod metrics:")
    cm = confusion_matrix(df[prod_behavior], df[prod_pred])
    pr = round(cm[1][1]*100/(cm[1][1]+cm[0][1]), 1)
    rc = round(cm[1][1]*100/(cm[1][1]+cm[1][0]), 1)

    print("tn\tfp\tfn\ttp\tprecision\trecall")
    print("{}\t{}\t{}\t{}\t{}\t\t{}".format(cm[0][0], cm[0][1], cm[1][0], cm[1][1], pr, rc))
    prod_metrics = [cm[0][0], cm[0][1], cm[1][0], cm[1][1], pr, rc]
    print("\n")

    cm = confusion_matrix(df[behavior], df[pred])
    pr = round(cm[1][1]*100/(cm[1][1]+cm[0][1]), 1)
    rc = round(cm[1][1]*100/(cm[1][1]+cm[1][0]), 1)
    print("Iteration metrics:")
    print("tn\tfp\tfn\ttp\tprecision\trecall")
    print("{}\t{}\t{}\t{}\t{}\t\t{}".format(cm[0][0], cm[0][1], cm[1][0], cm[1][1], pr, rc))
    iteration_metrics = [cm[0][0], cm[0][1], cm[1][0], cm[1][1], pr, rc]
    print("\n")

    print("High Conf:")
    cm = confusion_matrix(df[behavior], df["high_conf"])
    pr = round(cm[1][1]*100/(cm[1][1]+cm[0][1]), 1)
    rc = round(cm[1][1]*100/(cm[1][1]+cm[1][0]), 1)
    print("tn\tfp\tfn\ttp\tprecision\trecall")
    print("{}\t{}\t{}\t{}\t{}\t\t{}".format(cm[0][0], cm[0][1], cm[1][0], cm[1][1], pr, rc))
    print("\n")
    
    
    print("Low Conf:")
    cm = confusion_matrix(df[behavior], df["low_conf"])
    pr = round(cm[1][1]*100/(cm[1][1]+cm[0][1]), 1)
    rc = round(cm[1][1]*100/(cm[1][1]+cm[1][0]), 1)
    print("tn\tfp\tfn\ttp\tprecision\trecall")
    print("{}\t{}\t{}\t{}\t{}\t\t{}".format(cm[0][0], cm[0][1], cm[1][0], cm[1][1], pr, rc))    
    
    difference = []
    zip_object = zip(iteration_metrics, prod_metrics)
    for list1_i, list2_i in zip_object:
        difference.append(list1_i-list2_i)

    print("Delta: (current - prod)")
    print("tn\tfp\tfn\ttp\tprecision\trecall")
    print("{}\t{}\t{}\t{}\t{}\t\t{}".format(difference[0],difference[1],difference[2],difference[3],difference[4],difference[5]))