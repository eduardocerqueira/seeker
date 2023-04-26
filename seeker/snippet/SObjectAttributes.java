//date: 2023-04-26T16:56:26Z
//url: https://api.github.com/gists/70374dbf0b94bb882770ad6ce56bda46
//owner: https://api.github.com/users/Vergil333

public class SObjectAttributes<T extends SObjectType> {
    public SObjectAttributes(T type, String url) {
        this.type = type;
        this.url = url;
    }
    T type;
    String url;
}