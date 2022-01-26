#date: 2022-01-26T16:59:20Z
#url: https://api.github.com/gists/38f5a80b292cc97f1afef73bbfb5f1ef
#owner: https://api.github.com/users/vishal-aiml164

# create Weekday/weekend feature and drop redundant features

test_df['weekday']=test_df['Date'].apply(weekday_fet)
def isweekend(x):
  if x==5 or x==6:
    return 1
  else:
    return 0
test_df['weekend']=test_df['weekday'].apply(isweekend)
test_df.drop('weekday',axis=1,inplace=True)

# categorical encoding using pandas pkg and drop redundant features
test_df_new=pd.get_dummies(test_df,columns=['Store_Type','Location_Type','Region_Code','Discount','Store_id'])
test_df_new.drop(['ID','Date'],axis=1,inplace=True)

# Use the trained model to predict
y_ans=model_ran.predict(test_df_new)
y_inv_ans=10**y_ans

# writing predictions to file
res=pd.DataFrame({'ID':df_test['ID'],'Sales':y_inv_ans})
res.to_csv('/content/SALES_PRED.csv', index=False)