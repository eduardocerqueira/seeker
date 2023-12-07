//date: 2023-12-07T17:08:37Z
//url: https://api.github.com/gists/d1d9f3c977910a7de994564a1719b6b4
//owner: https://api.github.com/users/manjot0074

global CaseBackupJob implements BackupJobInterface {
  global Database.QueryLocator getQuery(){
    return Database.QueryLocator([select id from case where lastmodifeddate > Last_N_Days:365 and status = 'resolved' and lastMessageDate__c > Last_N_Days:6 ])
  }
}