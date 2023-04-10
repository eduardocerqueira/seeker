//date: 2023-04-10T16:51:46Z
//url: https://api.github.com/gists/731de61ff8e238264915d05e8cc58322
//owner: https://api.github.com/users/Vergil333

public enum SObjectType {
    @JsonProperty("Account") ACCOUNT("Account"),
    @JsonProperty("TestObject__c") TEST_OBJECT__C("TestObject__c"),
    ;

    SObjectType(String jsonName) {}
}