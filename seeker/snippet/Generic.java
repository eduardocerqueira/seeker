//date: 2023-07-24T16:46:36Z
//url: https://api.github.com/gists/4ac35f6386534ae80821b46a0a4e1673
//owner: https://api.github.com/users/jairoArh

package generic;

public class Generic <T>{
    private T info;

    public Generic(T info ){
        this.info = info;
    }

    public T getInfo() {
        return info;
    }

    public void setInfo(T info) {
        this.info = info;
    }

    public String toString (){
        return info.getClass().getName();
    }
}

