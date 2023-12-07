//date: 2023-12-07T16:55:48Z
//url: https://api.github.com/gists/0a11994744876448a01c892352a7f657
//owner: https://api.github.com/users/manjot0074

global Database.QueryLocator start(Database.BatchableContext BC){
    Backup_Custom_metadata__mdt backup = getbackupSetting();
    if(String.isNotblank(backup.DIClassName__c){
      Type obType = Type.forName(backup.DIClassName__c);
      BackupJobInterface job = (obType == null) ? null : (BackupJobInterface)obType.newInstance();
      return job.getQuery();
    }
    Date lastmodifieddate = backup.lastmodifiedDate__c;
    return Database.getQueryLocator('select id from '+backup.objectName__c+ ' where lastmodifieddate >: lastmodifieddate');    
}