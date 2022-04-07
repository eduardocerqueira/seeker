#date: 2022-04-07T17:04:12Z
#url: https://api.github.com/gists/52a66012aa515cc75e82a8faba984167
#owner: https://api.github.com/users/maniezhilan

#!/usr/bin/env python

"""
Social_Metrics_Tutorial_Script_User_Timeline_All_Pages_MongoDB.py - DOWNLOADS ALL AVAILABLE RECENT
TWEETS FROM 5 MLB ACCOUNTS INTO MONGODB DATABASE

BEFORE RUNNING THIS SCRIPT, YOU WILL NEED TO:
  1. HAVE ANACONDA PYTHON 2.7 INSTALLED
  2. HAVE CREATED CSV FILE (E.G., IN EXCEL) CONTAINING TWITTER HANDLES YOU 
     WISH TO DOWNLOAD (SEE TUTORIAL FOR DETAILS)
  3. HAVE MONGODB INSTALLED AND RUNNING


THE CODE IS DIVIDED INTO SEVEN PARTS:
  1. Importing necessary Python packages 
  2. Importing Twython and Twitter app key and access token
       - YOU NEED TO MODIFY THIS SECTION IN ORDER TO GET SCRIPT TO WORK (LINES 39-41)
  3. Defining function for getting Twitter data
  4. Set up MongoDB database and collections (tables)
  5. Read in Twitter accounts (and add to MongoDB database if first run)
  6. Main loop over each of the Twitter handles in the accounts table of the database.  
  7. Print out number of tweets in database per account    
"""


###### PART I: IMPORT PYTHON PACKAGES (ALL BUT TWYTHON ARE INSTALLED W/ ANACONDA PYTHON ###### 
import sys
import time
import json
import pandas as pd
from twython import Twython #NEEDS TO BE INSTALLED SEPARATELY ONCE: pip install Twython





###### PART II: IMPORT TWYTHON, ADD TWITTER APP KEY & ACCESS TOKEN (TO ACCESS API) ###### 

#REPLACE 'APP_KEY' AND 'ACCESS_TOKEN' WITH YOUR APP KEY & ACCESS TOKEN IN THE NEXT 2 LINES
APP_KEY = ' '    
ACCESS_TOKEN = ' '
twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)





###### PART III: DEFINE TWYTHON FUNCTION FOR GETTING ALL AVAILABLE PAGES OF TWEETS PER USER ###### 

def get_data_user_timeline_all_pages(kid, page):
    try:        
        '''
        'count' specifies the number of tweets to try and retrieve, up to a maximum of 200
        per distinct request. The value of count is best thought of as a limit to 
        the number of tweets to return because suspended or deleted content is removed 
        after the count has been applied. We include retweets in the count, even if 
        include_rts is not supplied. It is recommended you always send include_rts=1 when 
        using this API method.
        '''        
        d = twitter.get_user_timeline(screen_name=kid, count="200", page=page, include_entities="true", include_rts="1")          
    except Exception, e:
        print "Error reading id %s, exception: %s" % (kid, e)
        return None
    print len(d) #NUMBER OF ENTRIES RETURNED
    #print "d.keys(): ", d[0].keys()
    return d





###### PART IV: SET UP MONGODB DATABASE AND ACCOUNTS AND TWEETS TABLES ###### 

#MAKE CONNECTION TO MONGODB
import pymongo
from pymongo import MongoClient
client = MongoClient()


# DEFINE YOUR MONGODB DATABASE
db = client['MLB']

# CREATE ACCOUNTS COLLECTION (TABLE) IN YOUR DATABASE FOR TWITTER ACCOUNT-LEVEL DETAILS
accounts = db['accounts']

# CREATE AN INDEX ON THE COLLECTION TO AVOID INSERTION OF DUPLICATES
db.accounts.create_index([('Twitter_handle', pymongo.ASCENDING)], unique=True)

# SHOW INDEX ON ACCOUNTS TABLE
#list(db.accounts.index_information())

#SHOW NUMBER OF ACCOUNTS IN TABLE
#accounts.count()

# DEFINE COLLECTION (TABLE) WHERE YOU'LL INSERT THE TWEETS
tweets = db['tweets']

# CREATE UNIQUE INDEX FOR TABLE (TO AVOID DUPLICATES)
db.tweets.create_index([('id_str', pymongo.ASCENDING)], unique=True)

#SHOW INDEX ON TWEETS COLLECTION
#list(db.tweets.index_information())

#SHOW NUMBER OF TWEETS IN TABLE
#tweets.count()

#TO SEE LIST OF CURRENT MONGODB DATABASES
#client.database_names()

#TO SEE LIST OF COLLECTIONS IN THE *MLGB* DATABASE
#db.collection_names()





###### PART V: READ IN TWITTER ACCOUNTS (AND ADD TO MONGODB IF FIRST RUN)

# IF ACCOUNTS COLLECTION IS EMPTY READ IN CSV FILE AND ADD TO MONGODB
if accounts.count() < 1:
    df = pd.read_csv('accounts.csv')
    records = json.loads(df.T.to_json()).values()
    print "No account data in MongoDB, attempting to insert", len(records), "records"
    try:
        accounts.insert_many(records)
    except pymongo.errors.BulkWriteError, e:
        print e, '\n'
        #pass  
else:
    print "There are already", accounts.count(), "records in the *accounts* table"


#LIST ROWS IN ACCOUNTS COLLECTION
#list(accounts.find())[:1]

# CREATE LIST OF TWITTER HANDLES FOR DOWNLOADING TWEETS
twitter_accounts = accounts.distinct('Twitter_handle')
#print len(twitter_accounts)
#twitter_accounts[:5]





###### PART VI: LOOP OVER TWITTER HANDLES & DOWNLOAD TWEETS INTO MONGODB COLLECTION ######  

import timeit
start_time = timeit.default_timer()

starting_count = tweets.count()

for s in twitter_accounts[:1]:
    
    #SET THE DUPLICATES COUNTER FOR THIS TWITTER ACCOUNT TO ZERO
    duplicates = 0
    
    #CHECK FOR TWITTER API RATE LIMIT (450 CALLS/15-MINUTE WINDOW)
    rate_limit = twitter.get_application_rate_limit_status()['resources']['statuses']['/statuses/user_timeline']['remaining']
    print '\n\n', '# of remaining API calls: ', rate_limit

    #tweet_id = str(mentions.find_one( { "query_screen_name": s}, sort=[("id_str", 1)])["id_str"])
    print 'Grabbing tweets sent by: ', s, '-- index number: ', twitter_accounts.index(s)
    
    page = 1
    
    #WE CAN GET 200 TWEETS PER CALL AND UP TO 3,200 TWEETS TOTAL, MEANING 16 PAGES' PER ACCOUNT 
    while page < 17:
        print "------XXXXXX------ STARTING PAGE", page, '...estimated remaining API calls:', rate_limit

        d = get_data_user_timeline_all_pages(s, page)          
        if not d:
            print "THERE WERE NO STATUSES RETURNED........MOVING TO NEXT ID"
            break      
        if len(d)==0:    #THIS ROW IS DIFFERENT FROM THE MENTIONS AND DMS FILES
            print "THERE WERE NO STATUSES RETURNED........MOVING TO NEXT ID"
            break
        #if not d['statuses']:
        #    break
        
                
        #DECREASE rate_limit TRACKER VARIABLE BY 1
        rate_limit -= 1
        print '.......estimated remaining API rate_limit: ', rate_limit    
    

        ##### WRITE THE DATA INTO MONGODB -- LOOP OVER EACH TWEET
        for entry in d:
            #ADD THE FOLLOWING THREE VARIABLES TO THOSE RETURNED BY TWITTER API
            entry['date_inserted'] = time.strftime("%d/%m/%Y")
            entry['time_date_inserted'] = time.strftime("%H:%M:%S_%d/%m/%Y")
            entry['screen_name'] = entry['user']['screen_name']
        
            #CONVERT TWITTER DATA TO PREP FOR INSERTION INTO MONGO DB
            t = json.dumps(entry)
            #print 'type(t)', type(t)                   #<type 'str'>
            loaded_entry = json.loads(t)
            #print type(loaded_entry) , loaded_entry    #<type 'dict'>
        
            #INSERT THE TWEET INTO THE DATABASE -- UNLESS IT IS ALREADY IN THE DB
            try:
                 tweets.insert_one(loaded_entry)
            except pymongo.errors.DuplicateKeyError, e:
                #print e, '\n'
                duplicates += 1
                pass     
        
        
        print '------XXXXXX------ FINISHED PAGE', page, 'FOR ORGANIZATION', s, "--", len(d), "TWEETS"
    
        #IF THERE ARE TOO MANY DUPLICATES THEN SKIP TO NEXT ACCOUNT 
        if duplicates > 20:
            print '\n********************There are %s' % duplicates, 'duplicates....moving to next ID********************\n'
            #continue        
            break
                    
        page += 1
        if page > 16:
            print "WE'RE AT THE END OF PAGE 16"
            break    
                
        #THIS IS A SOMEWHAT CRUDE METHOD OF PUTTING IN AN API RATE LIMIT CHECK
        #THE RATE LIMIT FOR CHECKING HOW MANY API CALLS REMAIN IS 180, WHICH MEANS WE CANNOT
        if rate_limit < 5:
            print 'Estimated fewer than 5 API calls remaining...check then pause 5 minutes if necessary'
            rate_limit_check = twitter.get_application_rate_limit_status()['resources']['statuses']['/statuses/user_timeline']['remaining']
            print '.......and here is our ACTUAL remaining API rate_limit: ', rate_limit_check
            if rate_limit_check<5:
                print 'Fewer than 5 API calls remaining...pausing for 5 minutes'
                time.sleep(300) #PAUSE FOR 300 SECONDS
                rate_limit = twitter.get_application_rate_limit_status()['resources']['statuses']['/statuses/user_timeline']['remaining']
                print '.......here is our remaining API rate_limit after pausing for 5 minutes: ', rate_limit
                #if rate_limit_check == 450:
                #    rate_limit = 450

    #if twitter.get_application_rate_limit_status()['resources']['search']['/search/tweets']['remaining']<5:
    if rate_limit < 5:
        print 'Fewer than 5 estimated API calls remaining...pausing for 5 minutes'
        time.sleep(300) #PAUSE FOR 900 SECONDS
      
        
elapsed = timeit.default_timer() - start_time
print '# of minutes: ', elapsed/60
print "Number of new tweets added this run: ", tweets.count() - starting_count
print "Number of tweets now in DB: ", tweets.count(), '\n', '\n'            
            
                        



###### PART VII: PRINT OUT NUMBER OF TWEETS IN DATABASE FOR EACH ACCOUNT ######  

for org in db.tweets.aggregate([
    {"$group":{"_id":"$screen_name", "sum":{"$sum":1}}} 
    ]):
    print org['_id'], org['sum']


