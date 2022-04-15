#date: 2022-04-15T16:52:13Z
#url: https://api.github.com/gists/885dcbdead1f082faa5cbd6b3e0e6c7c
#owner: https://api.github.com/users/ShubhashreeSur

def predict(X_test,trained_models,meta_model):
    
    
    #predictions for test dataset using the trained base models so that to create a new dataset
    for i in range(len(trained_models)):
        
        predictions_test=trained_models[i].predict_proba(pd.DataFrame(X_test))
        
        if i==0:
            predictions_test_array=predictions_test
            
        else:
            predictions_test_array=np.hstack((predictions_test_array,predictions_test))
    
    
    #predictions for test dataset using the meta model
    predict_test_y=meta_model.predict_proba(predictions_test_array)
    
    return predict_test_y