//date: 2022-11-09T17:17:26Z
//url: https://api.github.com/gists/1854d970a284b984cafceb639af2d89c
//owner: https://api.github.com/users/MrGlass42

package hornetq;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * HornetQExampleBase
 * 
 * @author monzou
 */
class HornetQExampleBase {

    static {
        InputStream in = HornetQExampleBase.class.getClassLoader().getResourceAsStream("system.properties");
        try {
            if (in != null) {
                Properties properties = new Properties();
                properties.load(in);
                System.getProperties().putAll(properties);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
