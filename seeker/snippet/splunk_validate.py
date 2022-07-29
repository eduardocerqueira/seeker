#date: 2022-07-29T17:01:22Z
#url: https://api.github.com/gists/abab0bc0d5208985b1642c62a1951107
#owner: https://api.github.com/users/rakeshdevineni123

#!/usr/bin/python
import subprocess
import re
import csv
import time

encoding = 'ascii'
start = '<sid>'
end = '</sid>'

query = ' | tstats count where (index=na* OR index=cs* earliest=-4h)  by sourcetype | sort -num(count)"'

print("Beginning query for new cluster\n")
cmd_for_sid= 'curl -u admin:' + "'password'" + ' -k https://splunk-api.hio.data.sfdc.net:8214/services/search/jobs -d search="' + query

process1 = subprocess.Popen(cmd_for_sid, shell=True, stdout=subprocess.PIPE)
output1 = process1.communicate()[0]
print("printing output with search id for new cluster\n")
print(output1)

sid = re.search('%s(.*)%s' % (start, end), output1).group(1)

cmd_for_output = 'curl -u admin:' + "'password'" + ' -k https://splunk-api.hio.data.sfdc.net:8214/services/search/jobs/' + sid + '/results/ --get -d output_mode=csv'
print("sleeping for 120 seconds for the query to complete\n")
time.sleep(120)

process2 = subprocess.Popen(cmd_for_output, shell=True, stdout=subprocess.PIPE)
output2 = process2.communicate()[0].decode(encoding)

newsourcetype_count = csv.reader(output2.splitlines(), delimiter=",")
newclustersrctypes = {rows[0]:rows[1] for rows in newsourcetype_count}

try:
    del newclustersrctypes['sourcetype']
except KeyError as ex:
    print("No such key: '%s'" % ex.message)

#print("----------Begin Sourcetypes and COUNT for the NEW cluster \n")
#print(newclustersrctypes)
#print("-----------End of Sourcetypes and COUNT for the NEW cluster\n")

print("Beginning query for old cluster\n")
cmd_for_sid_old= 'curl -u admin:' + "'password'" + ' -k https://shared1-logsearch1-1-hio.ops.sfdc.net:8214/services/search/jobs -d search="' + query

process3 = subprocess.Popen(cmd_for_sid_old, shell=True, stdout=subprocess.PIPE)
output3 = process3.communicate()[0]
print("printing output with search id in old cluster\n")
print(output3)

sid_old_cluster = re.search('%s(.*)%s' % (start, end), output3).group(1)

cmd_for_output_old = 'curl -u admin:' + "'password'" + ' -k https://shared1-logsearch1-1-hio.ops.sfdc.net:8214/services/search/jobs/' + sid_old_cluster + '/results/ --get -d output_mode=csv'
print("sleeping for 120 seconds for the query to complete\n")
time.sleep(120)

process4 = subprocess.Popen(cmd_for_output_old, shell=True, stdout=subprocess.PIPE)
output4 = process4.communicate()[0].decode(encoding)

oldsourcetype_count = csv.reader(output4.splitlines(), delimiter=",")
oldclustersrctypes = {row[0]:row[1] for row in oldsourcetype_count}

try:
    del oldclustersrctypes['sourcetype']
except KeyError as ex:
    print("No such key: '%s'" % ex.message)

#print("----------Begin Sourcetypes and COUNT for the OLD cluster \n")
#print(oldclustersrctypes)
#print("-----------End of Sourcetypes and COUNT for the OLD cluster\n")


########-------------------------

oldcount = 0
newcount = 0
for oldsum in oldclustersrctypes:
    oldcount = oldcount + int(oldclustersrctypes[oldsum])
for newsum in newclustersrctypes:
    newcount = newcount + int(newclustersrctypes[newsum])

compared_matches = {}
for key1 in oldclustersrctypes:
    temp_list_common = []
    temp_list = []
    if key1 in newclustersrctypes:
        old=float(oldclustersrctypes[key1])
        new=float(newclustersrctypes[key1])
        diff = old-new
        percent_diff = (abs(diff/old)) * 100
        #               print(diff/old)
        #               print(diff)
        #               print(percent_diff)
        temp_list_common.append(oldclustersrctypes[key1])
        temp_list_common.append(newclustersrctypes[key1])
        temp_list_common.append("difference " + str(diff))
        temp_list_common.append("percentage missing " + str(percent_diff))
        compared_matches[key1]=temp_list_common
    else:
        temp_list.append(oldclustersrctypes[key1])
        temp_list.append(" - ")
        compared_matches[key1]=temp_list
print("\nPrinting SOURCETYPE and Counts in OLD Cluster and NEW Cluster respectively.................\n")
print(compared_matches)
print("Total Count for OLD cluster is : " + str(oldcount))
print("Total Count for NEW cluster is : " + str(newcount))