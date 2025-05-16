//date: 2025-05-16T16:59:28Z
//url: https://api.github.com/gists/0d4093e3e34153434054b26ce7b4a5d5
//owner: https://api.github.com/users/sfdevmaster

public class AccountUpdateBatch implements Database.Batchable<sObject>, Database.Stateful {
    
    // Stateful variables to track results
    private List<String> successRecords;
    private List<String> errorRecords;
    
    public AccountUpdateBatch() {
        successRecords = new List<String>();
        errorRecords = new List<String>();
    }
    
    public Database.QueryLocator start(Database.BatchableContext BC) {
        return Database.getQueryLocator('SELECT Id, Name, Description FROM Account WHERE CreatedDate = TODAY');
    }
    
    public void execute(Database.BatchableContext BC, List<Account> scope) {
        List<Account> accountsToUpdate = new List<Account>();
        
        for(Account acc : scope) {
            try {
                acc.Description = 'Updated by batch - ' + System.today();
                accountsToUpdate.add(acc);
            } catch(Exception e) {
                errorRecords.add(acc.Id + ',' + acc.Name + ',' + e.getMessage());
            }
        }
        
        // Update with partial success allowed
        List<Database.SaveResult> srList = Database.update(accountsToUpdate, false);
        
        for(Integer i = 0; i < srList.size(); i++) {
            if(srList[i].isSuccess()) {
                successRecords.add(accountsToUpdate[i].Id + ',' + accountsToUpdate[i].Name);
            } else {
                errorRecords.add(accountsToUpdate[i].Id + ',' + accountsToUpdate[i].Name + ',' + srList[i].getErrors()[0].getMessage());
            }
        }
    }
    
    public void finish(Database.BatchableContext BC) {
        // Create CSV content
        String successCSV = 'Id,Name\n' + String.join(successRecords, '\n');
        String errorCSV = 'Id,Name,Error\n' + String.join(errorRecords, '\n');
        
        // Create email with attachments
        Messaging.SingleEmailMessage mail = new Messaging.SingleEmailMessage();
        mail.setToAddresses(new String[] {'admin@example.com'});
        mail.setSubject('Account Update Batch Results');
        mail.setPlainTextBody('Please find attached the results of the Account Update Batch job.');
        
        // Add attachments
        List<Messaging.EmailFileAttachment> attachments = new List<Messaging.EmailFileAttachment>();
        
        Messaging.EmailFileAttachment successAttach = new Messaging.EmailFileAttachment();
        successAttach.setFileName('success_records.csv');
        successAttach.setBody(Blob.valueOf(successCSV));
        attachments.add(successAttach);
        
        Messaging.EmailFileAttachment errorAttach = new Messaging.EmailFileAttachment();
        errorAttach.setFileName('error_records.csv');
        errorAttach.setBody(Blob.valueOf(errorCSV));
        attachments.add(errorAttach);
        
        mail.setFileAttachments(attachments);
        Messaging.sendEmail(new Messaging.SingleEmailMessage[] { mail });
    }
}