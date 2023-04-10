//date: 2023-04-10T16:51:46Z
//url: https://api.github.com/gists/731de61ff8e238264915d05e8cc58322
//owner: https://api.github.com/users/Vergil333

public abstract class AbstractSObject<T extends SObjectType> implements SObjectInterface {
    public AbstractSObject(T type) {
        this.attributes = new SObjectAttributes<>(type, null);
    }

    @JsonProperty("Id") public String id;
    SObjectAttributes<T> attributes;
}