//date: 2023-06-01T17:03:19Z
//url: https://api.github.com/gists/24cb1f16c7c34613adfeac9abb6caed8
//owner: https://api.github.com/users/shavkatnazarov

package it.ul.restaranserverbackend2.config;

import org.springframework.core.io.ClassPathResource;

import javax.swing.*;
import java.io.IOException;
import java.util.Properties;

public class InitConfig {
    public static boolean isStart() {
        Properties properties = new Properties();
        try {
            properties.load(new ClassPathResource("/application.properties").getInputStream());
            if (properties.getProperty("spring.jpa.hibernate.ddl-auto").equals("update")) {
                return true;
            } else {
                String code = JOptionPane.showInputDialog("Ma'lumotlarni o'chirib yuborma agar o'chirib yuborsang bilmay qoldim dema. burchakda o'lib ketasan ochish kodi ustozingdan sora");
                if (code != null && code.equals("QOZI")) {
                    return true;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return false;
    }
}
