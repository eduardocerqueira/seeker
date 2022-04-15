#date: 2022-04-15T16:54:07Z
#url: https://api.github.com/gists/81f29483a5b7e25c0aec49cbcbb9fb10
#owner: https://api.github.com/users/ShubhashreeSur

#predicting
final_predict_test_y=predict(test,trained_models,meta_model)

#getting the top 5 countries based on probability scores
id=[]
country=[]
for i in range(len(test_df.id.values)):
    country.extend(np.argsort(final_predict_test_y[i])[-1:-6:-1])
    for j in range(5):
        id.append(test_df.id.values[i])
        
predicted_df=pd.DataFrame({'id':id,'country':country})

#getting the destination country names by label encoder object
predicted_df.country=le.inverse_transform(predicted_df.country)

#saving the output file in csv format
predicted_df.to_csv('custom_ensembler_output.csv',index=False)