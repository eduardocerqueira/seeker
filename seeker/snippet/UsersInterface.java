//date: 2024-01-03T16:48:53Z
//url: https://api.github.com/gists/185bbe21c97189b72cfa01587539dfdb
//owner: https://api.github.com/users/SA-JackMax

import com.sun.xml.internal.fastinfoset.util.CharArray;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;

import org.alfresco.jlan.server.auth.UserAccount;
import org.alfresco.jlan.server.auth.UsersInterface;
import org.alfresco.jlan.server.config.InvalidConfigurationException;
import org.alfresco.jlan.server.config.ServerConfiguration;
import org.alfresco.jlan.util.HexDump;
import org.apache.commons.codec.DecoderException;
import org.apache.commons.codec.binary.Hex;
import org.bouncycastle.jce.provider.BouncyCastleProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.extensions.config.ConfigElement;

import java.security.Security;

/**
 * Created by dkopel on 3/15/16.
 */
public class UsersInterface implements UsersInterface {
    private Logger logger = LoggerFactory.getLogger(getClass());

    private UserAuthenticationService userAuthenticationService;

    public UsersInterface() {
        Security.addProvider(new BouncyCastleProvider());
        userAuthenticationService = BeanAccessor.getBean(UserAuthenticationService.class);
    }

    @Override
    public void initializeUsers(ServerConfiguration serverConfiguration, ConfigElement configElement) throws InvalidConfigurationException {

    }

    @Override
    public UserAccount getUserAccount(String s) {
        MD4User u = findAccount(s);

        if(u != null) {
            logger.info("User exists with the username {}", s);
            logger.info("MD4 Password {}", HexDump.hexString(u.getPasswordMD4()));

            UserAccount us = new UserAccount(
                u.getUsername(),
                null
            );

            try {
                us.setMD4Password(Hex.decodeHex(u.getPasswordMD4String().toCharArray()));
            } catch (DecoderException e) {
                e.printStackTrace();
            }

            return us;
        } else {
            logger.info("User does not exist with the username {}", s);
            return null;
        }
    }

    public MD4User findAccount(String s) {
        return userAuthenticationService.getUser(s);
    }

    interface UserAuthenticationService {
        boolean authenticate(String username, String password);

        MD4User getUser(String username);
    }

    interface MD4User extends UserDetails {
        byte[] getPasswordMD4();

        String getPasswordMD4String();
    }
}
