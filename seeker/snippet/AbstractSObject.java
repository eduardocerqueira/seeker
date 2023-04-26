//date: 2023-04-26T16:56:26Z
//url: https://api.github.com/gists/70374dbf0b94bb882770ad6ce56bda46
//owner: https://api.github.com/users/Vergil333

public abstract class AbstractSObject<T extends SObjectType> implements SObjectInterface {
    public AbstractSObject(T type) {
        this.attributes = new SObjectAttributes<>(type, null);
    }

    @JsonProperty("Id") public String id;
    SObjectAttributes<T> attributes;
}