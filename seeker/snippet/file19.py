#date: 2022-04-15T16:50:58Z
#url: https://api.github.com/gists/e4c08e5aae0dbea84e2704d0e5e38ae8
#owner: https://api.github.com/users/ShubhashreeSur

def custom_ensembler(X_train,y_train,X_test,n_estimators,models):
    
    #splitting the data into D1 and D2 datasets (50-50 proportion)
    D1,D2,D1_y,D2_y=train_test_split(X_train,y_train,test_size=0.5,random_state=42)
    
    D1=pd.DataFrame(np.hstack((D1,D1_y.values.reshape(-1,1))))
    
    trained_models=[]
    
    print("Training the base models...\n")        #training the base models 
    for i in range(n_estimators):
        
        #getting the samples with replacement
        d=D1.sample(n=len(D1),replace=True,random_state=np.random.randint(1,100))
        d_y=d.iloc[:,-1]
        d=d.iloc[:,:-1]
        
        #fitting the base model and then fitting a calibrated classifier since we want probabilities
        models[i].fit(d,d_y)
        sig_clf = CalibratedClassifierCV(models[i], method="sigmoid")
        sig_clf.fit(d, d_y)
        
        #prediciting using the trained base model for the sample we created
        predict_y=sig_clf.predict_proba(d)
        print("The train log loss for model ",i,": ",log_loss(d_y, predict_y, eps=1e-15))
        print("--"*30)
        
        #predicting for D2 dataset
        predictions_D2=sig_clf.predict_proba(pd.DataFrame(D2))
        
        
        #stacking the predictions for D2 dataset that we get using the trained base models to create another dataset
        if i==0:
            predictions_D2_array=predictions_D2
            
        else:
            predictions_D2_array=np.hstack((predictions_D2_array,predictions_D2))
        
        #getting the trained moels in a list
        trained_models.append(sig_clf)
            
        
    predictions_D2_df=pd.DataFrame(predictions_D2_array)
    
    
    
    print("\nTraining the meta classifier...")    #training the meta classifier

    meta_clf = SGDClassifier(alpha=1,loss='log',class_weight='balanced')   #using logistic regression as meta model
    meta_clf.fit(predictions_D2_df, D2_y)
    meta_sig_clf = CalibratedClassifierCV(meta_clf, method="sigmoid")
    meta_sig_clf.fit(predictions_D2_df, D2_y)

    #predicting using the meta model
    predict_D2_y = meta_sig_clf.predict_proba(predictions_D2_df)
    print("\nThe log loss is:",log_loss(D2_y, predict_D2_y, eps=1e-15))    
    
    
    
    #predictions for test dataset using the trained base models so that to create a new dataset
    for i in range(n_estimators):
        
        predictions_test=trained_models[i].predict_proba(pd.DataFrame(X_test))
        
        if i==0:
            predictions_test_array=predictions_test
            
        else:
            predictions_test_array=np.hstack((predictions_test_array,predictions_test))
        
    
    #predictions for test dataset using the meta model
    predict_test_y=meta_sig_clf.predict_proba(predictions_test_array)
    
    
    return predict_test_y,trained_models,meta_sig_clf