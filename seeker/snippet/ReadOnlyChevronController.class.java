//date: 2022-01-21T16:41:52Z
//url: https://api.github.com/gists/f568fd7c71e6714bc6a6dd8856d2ffc6
//owner: https://api.github.com/users/Marley227

/**
 * @Author	:	Jitendra Zaa
 * @Date	:	17 Sep 2017
 * @Desc	:	Controller for Lightning Component "ReadOnlyChevron"
 * 
 * */
public class ReadOnlyChevronController {
    @AuraEnabled
    public static String getChevronData(String recId,String fieldName){ 
        //For Demo purpose
        if(recId == null){
            recId = '00Q3600000cSOFwEAO';
        }        
        // Logic as per Q 112 : Dynamic Apex
        // http://www.jitendrazaa.com/blog/salesforce/salesforce-interview-question-part-12/         
        List<Schema.SObjectType> gd = Schema.getGlobalDescribe().Values();
        Map<String,String> objectMap = new Map<String,String>();
        for(Schema.SObjectType f : gd)
        {
             objectMap.put(f.getDescribe().getKeyPrefix(), f.getDescribe().getName());
        }
        String prefix =  recId.left(3); 
        String objectName = objectMap.get(prefix);
        String query = 'SELECT '+fieldName+' FROM '+objectName+' WHERE Id =: recId';        
        List<SOBject> lstObj = Database.query(query);        
        String selVal =  String.valueOf(lstObj[0].get(fieldName)) ;  
        Schema.SObjectField sobjField = Schema.getGlobalDescribe().get(objectName).getDescribe().Fields.getMap().get(fieldName) ;
        Schema.DescribeFieldResult fieldResult = sobjField.getDescribe() ;
        List<Schema.PicklistEntry> ple = fieldResult.getPicklistValues();        
        Boolean curValMatched = false;
        Integer widthPerItem = 100/ple.size() ;
        List<chevronData> lstRet = new List<chevronData>();        
        for( Schema.PicklistEntry f : ple)
        {
            chevronData obj = new chevronData();
            obj.val = f.getLabel();
            obj.width = widthPerItem+'%';            
            if(obj.val == selVal){
                obj.cssClass = 'active';
                curValMatched = true;
            }
            else if(curValMatched){
                obj.cssClass = '';
            }else{
                obj.cssClass = 'visited'; 
            } 
            lstRet.add(obj);
        } 
        return JSON.serialize(lstRet);
    }    
    public class chevronData{
        public String val{get;set;}
        public String cssClass{get;set;}
        public String width {get;set;}
    }
}