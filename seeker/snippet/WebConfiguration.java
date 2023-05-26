//date: 2023-05-26T16:43:47Z
//url: https://api.github.com/gists/5132c6d1daf2b5b32ae3fbab2e9c0d8c
//owner: https://api.github.com/users/tomatophobia

import com.linecorp.armeria.server.tomcat.TomcatService;
import com.linecorp.armeria.spring.ArmeriaServerConfigurator;
import org.apache.catalina.connector.Connector;
import org.springframework.boot.web.embedded.tomcat.TomcatWebServer;
import org.springframework.boot.web.servlet.context.ServletWebServerApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class WebConfiguration {
   public static Connector getConnector(ServletWebServerApplicationContext applicationContext) {
       final TomcatWebServer container = (TomcatWebServer) applicationContext.getWebServer();
       container.start();
       return container.getTomcat().getConnector();
   }

   @Bean
   public TomcatService tomcatService(ServletWebServerApplicationContext applicationContext) {
       return TomcatService.of(getConnector(applicationContext));
   }
   @Bean
   public ArmeriaServerConfigurator armeriaServerConfigurator(TomcatService tomcatService) {
       return sb -> sb.serviceUnder("/", tomcatService);
   }
}