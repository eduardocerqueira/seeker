//date: 2025-05-16T17:00:12Z
//url: https://api.github.com/gists/c8ff2f5a3ec456b008a03f4088a29d76
//owner: https://api.github.com/users/sfdevmaster

public class ContactCreationBatch implements Database.Batchable<sObject>, Database.Stateful {
    
    // Stateful variables for tracking
    private List<ContactWrapper> successList;
    private List<ContactWrapper> failureList;
    
    // Wrapper class to store contact details
    private class ContactWrapper {
        String accountId;
        String contactName;
        String status;
        String errorMessage;
        
        public ContactWrapper(String accId, String name, String sts, String err) {
            accountId = accId;
            contactName = name;
            status = sts;
            errorMessage = err;
        }
        
        public String toCSVString() {
            return String.join(new List<String>{accountId, contactName, status, errorMessage}, ',');
        }
    }
    
    public ContactCreationBatch() {
        successList = new List<ContactWrapper>();
        failureList = new List<ContactWrapper>();
    }
    
    public Database.QueryLocator start(Database.BatchableContext BC) {
        return Database.getQueryLocator('SELECT Id, Name FROM Account WHERE CreatedDate = LAST_N_DAYS:7');
    }
    
    public void execute(Database.BatchableContext BC, List<Account> scope) {
        List<Contact> contactsToInsert = new List<Contact>();
        
        for(Account acc : scope) {
            Contact con = new Contact(
                AccountId = acc.Id,
                LastName = acc.Name + ' Contact',
                Email = 'contact_' + acc.Id + '@example.com'
            );
            contactsToInsert.add(con);
        }
        
        // Insert with partial success allowed
        List<Database.SaveResult> srList = Database.insert(contactsToInsert, false);
        
        for(Integer i = 0; i < srList.size(); i++) {
            if(srList[i].isSuccess()) {
                successList.add(new ContactWrapper(
                    scope[i].Id,
                    contactsToInsert[i].LastName,
                    'Success',
                    ''
                ));
            } else {
                failureList.add(new ContactWrapper(
                    scope[i].Id,
                    contactsToInsert[i].LastName,
                    'Failed',
                    srList[i].getErrors()[0].getMessage()
                ));
            }
        }
    }
    
    public void finish(Database.BatchableContext BC) {
        // Create CSV headers
        String csvHeader = 'AccountId,ContactName,Status,ErrorMessage\n';
        
        // Create CSV content
        List<String> successCSVRows = new List<String>();
        List<String> failureCSVRows = new List<String>();
        
        for(ContactWrapper sw : successList) {
            successCSVRows.add(sw.toCSVString());
        }
        
        for(ContactWrapper fw : failureList) {
            failureCSVRows.add(fw.toCSVString());
        }
        
        String successCSV = csvHeader + String.join(successCSVRows, '\n');
        String failureCSV = csvHeader + String.join(failureCSVRows, '\n');
        
        // Send email with results
        Messaging.SingleEmailMessage email = new Messaging.SingleEmailMessage();
        email.setToAddresses(new String[] {'admin@example.com'});
        email.setSubject('Contact Creation Batch Results');
        email.setPlainTextBody('Batch job completed. Please find the results attached.');
        
        // Create attachments
        List<Messaging.EmailFileAttachment> attachments = new List<Messaging.EmailFileAttachment>();
        
        if(!successList.isEmpty()) {
            Messaging.EmailFileAttachment successAttach = new Messaging.EmailFileAttachment();
            successAttach.setFileName('success_contacts.csv');
            successAttach.setBody(Blob.valueOf(successCSV));
            attachments.add(successAttach);
        }
        
        if(!failureList.isEmpty()) {
            Messaging.EmailFileAttachment failureAttach = new Messaging.EmailFileAttachment();
            failureAttach.setFileName('failed_contacts.csv');
            failureAttach.setBody(Blob.valueOf(failureCSV));
            attachments.add(failureAttach);
        }
        
        email.setFileAttachments(attachments);
        Messaging.sendEmail(new Messaging.SingleEmailMessage[] { email });
    }
}