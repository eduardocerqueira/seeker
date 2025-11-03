//date: 2025-11-03T16:54:42Z
//url: https://api.github.com/gists/269eb2b3a5b8ac3b583ae100e7d02d0c
//owner: https://api.github.com/users/amitastreait

global with sharing class MySelectionOptions implements Callable {

    global Object call(String methodName, Map<String, Object> args) {
        // input
        // output
        // options
        Map<String, Object> input = (Map<String, Object>)args.get('input');
        Map<String, Object> output = (Map<String, Object>)args.get('output');
        Map<String, Object> options = (Map<String, Object>)args.get('options');
        // output.put('message','Welcome to OmniScript!');
        
        return invokeMethod(methodName, input, output);
    }
    
    private Object invokeMethod(String methodName, Map<String, Object> input, Map<String, Object> output){
        if(methodName == 'parentAccount'){
            
            List< Map <String, String>> UIoptions = new List< Map <String, String>>();
            for(Account acc : [Select Id, Name FROM Account LIMIT 100]){
                Map<String,String> tempMap = new Map<String,String>();
                tempMap.put('name', acc.Id); // API backend
                tempMap.put('value', acc.Name);// Value - Front UI
                UIoptions.add(tempMap);
            }
            output.put('options', UIoptions);
            
        } else if(methodName == 'industryOptions'){
            
            List<Map <String, String> > picklistValues = new List< Map <String, String> >();
            
            Map<String, Schema.SObjectField> fieldMap = Schema.getGlobalDescribe()
                .get('Account')
                .getDescribe()
                .fields.getMap();
            
            Schema.DescribeFieldResult fieldDescribe = fieldMap.get('Industry').getDescribe();
            for (Schema.PicklistEntry ple : fieldDescribe.getPicklistValues()) {
                Map<String,String> tempMap = new Map<String,String>();
                tempMap.put('name', ple.getValue());
                tempMap.put('value', ple.getLabel());
                picklistValues.add(tempMap);
            }
            output.put('options', picklistValues);
        }
        return true;
    }
}