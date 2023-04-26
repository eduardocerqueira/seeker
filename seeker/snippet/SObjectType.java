//date: 2023-04-26T16:56:26Z
//url: https://api.github.com/gists/70374dbf0b94bb882770ad6ce56bda46
//owner: https://api.github.com/users/Vergil333

public enum SObjectType {
    @JsonProperty("Account") ACCOUNT("Account"),
    @JsonProperty("TestObject__c") TEST_OBJECT__C("TestObject__c"),
    ;

    SObjectType(String jsonName) {}
}