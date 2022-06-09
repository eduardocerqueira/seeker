//date: 2022-06-09T17:05:10Z
//url: https://api.github.com/gists/396909edefa307f64e870281640184d2
//owner: https://api.github.com/users/jcoon97

import org.apache.commons.codec.binary.Hex;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;

public class Validator {
    private final String body;
    private final String secret;
    private final String signature;

    Validator(final String body, final String secret, final String signature) {
        this.body = body;
        this.secret = secret;
        this.signature = signature;
    }

    public boolean validate() throws NoSuchAlgorithmException, InvalidKeyException {
        Mac shaHmac = Mac.getInstance("HmacSHA256");
        SecretKeySpec keySpec = new SecretKeySpec(secret.getBytes(StandardCharsets.UTF_8), "HmacSHA256");
        shaHmac.init(keySpec);

        String hexString = Hex.encodeHexString(shaHmac.doFinal(body.getBytes(StandardCharsets.UTF_8)));
        return hexString.equals(signature);
    }
}