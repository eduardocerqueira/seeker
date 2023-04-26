//date: 2023-04-26T16:56:26Z
//url: https://api.github.com/gists/70374dbf0b94bb882770ad6ce56bda46
//owner: https://api.github.com/users/Vergil333

public class SfResponse<T extends SObjectInterface> {
    Integer totalSize;
    Boolean done;
    List<T> records;

    public Integer getTotalSize() {
        return totalSize;
    }

    public Boolean getDone() {
        return done;
    }

    public List<T> getRecords() {
        return records;
    }
}