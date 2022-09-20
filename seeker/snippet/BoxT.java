//date: 2022-09-20T17:12:19Z
//url: https://api.github.com/gists/b6fd5e9ceef44d350557a091e90383fb
//owner: https://api.github.com/users/BetterProgramming

public class Box<T> {
    private T mItem;

    public Box(T item) {
        this.mItem = item;
    }

    public T get() {
        return mItem;
    }

    public void setItem(T value) {
        this.mItem = value;
    }
}