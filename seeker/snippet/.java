//date: 2021-11-24T17:14:33Z
//url: https://api.github.com/gists/6fa17c5a2ded458fca3ffa882c54f2e8
//owner: https://api.github.com/users/jaimemin

@Slf4j
public abstract class AbstractTemplate {

    public void execute() {
        long startTime = System.currentTimeMillis();

        // 비즈니스 로직 실행
        step1(); // 상속
        step2();
        step3();
        // 비즈니스 로직 종료

        long endTime = System.currentTimeMillis();
        long resultTime = endTime - startTime;
        log.info("resultTime={}", resultTime);
    }

    protected abstract void step1();
    
    protected abstract void step2();
  
    protected abstract void step3();
    
}