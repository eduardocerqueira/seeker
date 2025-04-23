#date: 2025-04-23T16:35:45Z
#url: https://api.github.com/gists/95637bacfd3e741cf96f41f535f7e4bf
#owner: https://api.github.com/users/gpmidi

from dialpad import DialpadClient
from pprint import pprint
import json
import os,os.path
import uuid
import requests
import time,datetime

# Fill in your company level API token here
TOKEN= "**********"

# Client
dp = "**********"=False, token=TOKEN)

# Keep track of all call IDs
call_ids = set()

# Funcs
def dump_json(path="",fname=None,data={}):
    if not fname:
        fname=uuid.uuid4().hex
    dpath=os.path.join(os.getcwd(),"dialpad-export",path)
    fpath=os.path.join(dpath,fname+".json")
    os.makedirs(dpath,exist_ok =True)
    with open(fpath,'w') as f:
        json.dump(data,f)

def dump_file(path="",fname=None,data=b'',ext=".json"):
    if not fname:
        fname=uuid.uuid4().hex
    dpath=os.path.join(os.getcwd(),"dialpad-export",path)
    fpath=os.path.join(dpath,fname+ext)
    os.makedirs(dpath,exist_ok =True)
    with open(fpath,'wb') as f:
        f.write(data)
        
# Company
company=dp.company.get()
dump_json(path="company",fname=company['name'],data=company.copy())
# Offices
for office in dp.office.list():
    # pprint(office)
    dump_json(path="office",fname=office['name'],data=office.copy())
# Global Contacts
for contact in dp.contact.list(limit=None):
    dump_json(path="contact/global",fname=contact['id'],data=contact.copy())   
# Dept
for dept in dp.department.request(["/",]):
    dump_json(path="department",fname=dept['name'],data=dept.copy())          
    # Calls
    for call in dp.call.request(["/",],data=dict(target_id=dept['id'],target_type="department")):
        call_ids.add(call['call_id'])
        dump_json(path=os.path.join('calls','by_department',dept['id']),fname=call['call_id'],data=call.copy())     
# Calls
for call in dp.call.request(["/",]):
    call_ids.add(call['call_id'])
    dump_json(path=os.path.join('calls',"global"),fname=call['call_id'],data=call.copy())      
    
# Per User
for user in dp.user.list():
    uid=user['id']
    name=user['display_name']
    dump_json(path="user",fname=user['id'],data=user.copy())
    
    # User Contacts
    for contact in dp.contact.list(limit=100000,owner_id=uid):
        dump_json(path=os.path.join('contact',uid),fname=contact['id'],data=contact.copy())

    # User Devices
    for device in dp.userdevice.list(user_id=uid):
        dump_json(path=os.path.join('device',uid),fname=device['id'],data=device.copy())

    # Calls
    for call in dp.call.request(["/",],data=dict(target_id=uid,target_type="user")):
        call_ids.add(call['call_id'])
        dump_json(path=os.path.join('calls','by_user',uid),fname=call['call_id'],data=call.copy())      

# Stats collection
ftypes=dict(
    csv=".csv",
    json=".json",
)
targets={
    "user":[],
    "office":[],
    "department":[],
}
for user in dp.user.list():
    targets["user"].append(user['id'])
for office in dp.office.list():
    targets["office"].append(office['id'])
for dept in dp.department.request(["/",]):
    targets["department"].append(dept['id'])

for office in dp.office.list():
    print(f"On office {office['id']}")
    for export_type in ['records','stats']:
        print(f"On Export Type {export_type}")
        for stat_type in ["calls", "texts", "voicemails", "recordings", "onduty"]:
            print(f"On Stat Type {stat_type}")
            for type_name,tgts in targets.items():
                print(f"On Type {type_name}")
                for target in tgts:           
                    print(f"On Target {target}")         
                    req=dp.stats.post(
                        days_ago_start=0,
                        days_ago_end=9999,
                        export_type=export_type,
                        stat_type=stat_type,
                        office_id=office['id'],
                        timezone="UTC",
                        target_id=target,
                        target_type=type_name,
                    )
                    rid=req.get('request_id')
                    data=None
                    while data is None and rid is not None:                
                        res=dp.stats.get(export_id=rid)
                        if res['status']=="failed":
                            # TODO: Retry
                            data="failed"
                        elif res['status']=="complete":
                            dl=requests.get(res['download_url'], allow_redirects=True)
                            dl.raise_for_status()
                            data=dl
                        elif res['status']=="processing":
                            time.sleep(5)
                            
                    if data=="failed":
                        continue
                        
                    dump_file(
                        path=os.path.join('stats',office['id'],export_type,stat_type,type_name),
                        fname=f"{type_name}_{target}",
                        data=data.content,
                        ext=ftypes[res['file_type']],
                    )

# Download all call recordings and transcripts
for call_id in call_ids:
    # Transcript
    res=dp.transcript.get(call_id=call_id)
    dump_json(path=os.path.join('calls','by_id',call_id),fname="transcript",data=res.copy())   
    # Deets
    res=dp.call.get_info(call_id=call_id)
    dump_json(path=os.path.join('calls','by_id',call_id),fname="info",data=res.copy())   
    
    # Recording
    for rec_deet in res.get('recording_details',[]):
        dl= "**********"="+TOKEN, allow_redirects=True,headers=dict(bearer=TOKEN))
        dl.raise_for_status()
        dump_file(
            path=os.path.join('calls','by_id',call_id,"recordings"),
            fname=f"{rec_deet['id']}",
            data=dl.content,
            ext=".mp3",
        )

print("Done!")
   ext=".mp3",
        )

print("Done!")
