#date: 2021-12-09T17:10:56Z
#url: https://api.github.com/gists/f2046a942efd4172f00c7ed6c0a72ded
#owner: https://api.github.com/users/sain1905

def features_f5(X):
    ''' This function generates the Dimensional Reduction features and adds two and three way features '''
    
    X1 = features_f2(X)
    
    X_pca = pca.transform(X1)
    X_pca_df = pd.DataFrame(X_pca, index=list(X1.index))
    X_pca_df.columns = [('pca_'+str(i)) for i in range(20)]
    
    X_tSVD = tSVD.transform(X1)
    X_tSVD_df = pd.DataFrame(X_tSVD, index=list(X1.index))
    X_tSVD_df.columns = [('tSVD_'+str(i)) for i in range(20)]
    
    X_Fica = F_ica.transform(X1)
    X_Fica_df = pd.DataFrame(X_Fica, index=list(X1.index))
    X_Fica_df.columns = [('ica_'+str(i)) for i in range(20)]
    
    # Generating dataframe for selected 2 feature combinations 
    two_feat_dict = dict()
    for feat in two_feat_selc:
        ft_list = feat.split('_')
        two_feat_dict[feat] = X[ft_list[0]] + X[ft_list[1]]
    two_feat_df = pd.DataFrame(two_feat_dict)
    
    # Generating dataframe for selected 3 feature combinations 
    three_feat_dict = dict()
    for feat in three_feat_selc:
        ft_list = feat.split('_')
        three_feat_dict[feat] = X[ft_list[0]] + X[ft_list[1]] + X[ft_list[2]]
    three_feat_df = pd.DataFrame(three_feat_dict)
    
    X_mod = pd.concat([X_pca_df,
                       X_tSVD_df,
                       X_Fica_df,
                       two_feat_df,
                       three_feat_df],axis=1)
    
    X_mod = X_mod.drop(Dim_redn_corr_feat,axis=1)
    
    return X_mod