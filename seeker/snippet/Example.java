//date: 2025-12-02T17:03:03Z
//url: https://api.github.com/gists/cc10b2fd5b5b447c6df1ec90bbddd87a
//owner: https://api.github.com/users/Alex-st

public class TestExample {
    
  @Autowired
    TestClock clock;
  
  public void test() {
        //when: "protocol end date will come in 1 hour"
        clock.setTime(oneHourBeforeEndDate);
        //...
        //and: "performing session event with delay"
        clock.plusHours(1);
        //...
        clock.reset();
}