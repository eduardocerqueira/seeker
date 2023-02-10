#date: 2023-02-10T16:52:25Z
#url: https://api.github.com/gists/99a9867afce3e21a3e67837ab55c795b
#owner: https://api.github.com/users/casanave

def viz_numeric_vars_to_target(pivot_table, labels_dict, target_column):
    
    """
    HELPER FUNCTION TO: output_viz(X, y, num_cols, target_column)
    
    DOCSTRING: inputs 'pivot_table' (output from pivot_table_mean, a pivot table
    of the numeric features means as distributed by the categorical outcomes of 
    the target,) 'labels_dict' (a dictionary of labels for plotting), and 
    'target_column'(the name of the target column.)
    
    Displays a barplot for each numeric feature, with each categorical target outcome
    as a bar on the X axis. 
    """
    
    try: 

        # itterate to show relationships between the target and numeric dependant variables
        # for each column in the pivot table, make a barplot with the target outcome on the x axis
        # and the numeric feature on the y axis and make sure the plots are properly labeled with 'Title Case' labels

        for col in pivot_table.columns:
            fig, ax = plt.subplots()
            figure = sns.barplot(data = pivot_table,
                                 x = pivot_table.index,
                                 y = pivot_table[col])

            figure.set_title(f"{labels_dict[col]} Averaged by Class")
            figure.set_ylabel(f"{labels_dict[col]}")
            figure.set_xlabel(target_column.replace("_", " ").title())
            plt.xticks(rotation = 45)

        
    except Exception as e_9:
        print('ERROR at viz_numeric_vars_to_target', {e_9})
        # returns NOTHING 
