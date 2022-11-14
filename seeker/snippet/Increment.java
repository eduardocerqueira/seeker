//date: 2022-11-14T17:20:13Z
//url: https://api.github.com/gists/2ec6f2827b6a8f42f3bf93e8f44d9448
//owner: https://api.github.com/users/nprokofiev

class Counter{
  private volatile int count;
  public void increment(){
    count++;
  }
  public int getCount(){
    return count;
  }
}