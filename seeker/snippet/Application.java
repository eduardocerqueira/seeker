//date: 2023-03-10T16:46:45Z
//url: https://api.github.com/gists/9186dac2eda267140d9e4ef163f2de2b
//owner: https://api.github.com/users/Davide453

package hello;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.servlet.ServletComponentScan;

@SpringBootApplication
@ServletComponentScan
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}