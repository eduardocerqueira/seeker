//date: 2023-04-10T16:56:27Z
//url: https://api.github.com/gists/b62c2823cf157c7a25862c2788a149e3
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