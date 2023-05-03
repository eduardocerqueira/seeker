//date: 2023-05-03T16:49:47Z
//url: https://api.github.com/gists/888015a56689fcaf88f6d54c070a03bc
//owner: https://api.github.com/users/thomasdarimont

package demo;

import java.nio.charset.StandardCharsets;
import java.security.InvalidAlgorithmParameterException;
import java.security.KeyFactory;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.Signature;
import java.security.spec.ECGenParameterSpec;
import java.security.spec.PKCS8EncodedKeySpec;
import java.security.spec.X509EncodedKeySpec;
import java.time.Clock;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Base64;

public class ReferenceTokenEncoder {

    private final Clock clock;

    private final PrivateKey privateKey;

    private final PublicKey publicKey;

    private final String signatureAlgorithm;

    public ReferenceTokenEncoder(Clock clock, PrivateKey privateKey, PublicKey publicKey, String signatureAlgorithm) {
        this.clock = clock;
        this.privateKey = privateKey;
        this.publicKey = publicKey;
        this.signatureAlgorithm = signatureAlgorithm;
    }

    public String encodeToken(String referenceTokenId, long expirationTimestamp) throws Exception {
        String referenceTokenString = referenceTokenId + ": "**********"
        byte[] signature = "**********"
        String signatureString = Base64.getEncoder().encodeToString(signature);
        return Base64.getEncoder().withoutPadding().encodeToString((referenceTokenString + ": "**********"
    }

    public ReferenceToken decodeToken(String input) throws Exception {

        String referenceTokenString = "**********"

        String[] parts = referenceTokenString.split(": "**********"
        if (parts.length != 3) {
            throw new IllegalArgumentException("Invalid reference token string");
        }
        String referenceTokenId = "**********"
        long timestamp = Long.parseLong(parts[1]);
        String signatureString = parts[2];
        byte[] signature = Base64.getDecoder().decode(signatureString);
        String referenceTokenData = referenceTokenId + ": "**********"
        boolean verified = "**********"
        boolean expired = timestamp < clock.millis();
        return new ReferenceToken(referenceTokenId, timestamp, verified && !expired);
    }

    private byte[] signData(String data) throws Exception {
        Signature sig = Signature.getInstance(signatureAlgorithm);
        sig.initSign(privateKey);
        sig.update(data.getBytes(StandardCharsets.UTF_8));
        return sig.sign();
    }

    private boolean verifyData(String data, byte[] signature) throws Exception {
        Signature sig = Signature.getInstance(signatureAlgorithm);
        sig.initVerify(publicKey);
        sig.update(data.getBytes(StandardCharsets.UTF_8));
        return sig.verify(signature);
    }


    public record ReferenceToken(String referenceTokenId, long timestamp, boolean verified) {
        public boolean isExpired(Clock clock) {
            return timestamp < clock.millis();
        }
    }

    public static void main(String[] args) throws Exception {

        String signatureAlgorithm = "SHA256withECDSA";

        KeyPair keyPair = Keys.generateKeyPair();
        String privateKeyString = Keys.privateKeyToString(keyPair.getPrivate());
        String publicKeyString = Keys.publicKeyToString(keyPair.getPublic());

        Clock clock = Clock.systemDefaultZone();
        ReferenceTokenEncoder tokenEncoder = "**********"
                clock, //
                Keys.getPrivateKey(privateKeyString), //
                Keys.getPublicKey(publicKeyString), //
                signatureAlgorithm //
        );

        String referenceTokenId = "**********"
        long expiresAtTimestamp = Instant.now().plus(5, ChronoUnit.MINUTES).toEpochMilli();

        String referenceTokenString = "**********"

        System.out.println("Reference Token: "**********"

        ReferenceToken referenceToken = "**********"

        System.out.println("Reference token ID: "**********"
        System.out.println("Timestamp: "**********"
        System.out.println("Verified: "**********"
        System.out.println("Expired: "**********"
    }

    static class Keys {

        public static final String KEY_ALGORITHM = "EC";

        private static final String CURVE_NAME = "secp256r1";

        public static KeyPair generateKeyPair() throws NoSuchAlgorithmException, InvalidAlgorithmParameterException {
            KeyPairGenerator keyGen = KeyPairGenerator.getInstance(KEY_ALGORITHM);
            ECGenParameterSpec ecSpec = new ECGenParameterSpec(CURVE_NAME);
            keyGen.initialize(ecSpec);
            return keyGen.generateKeyPair();
        }

        public static String publicKeyToString(PublicKey publicKey) {
            return Base64.getEncoder().encodeToString(publicKey.getEncoded());
        }

        public static String privateKeyToString(PrivateKey privateKey) {
            return Base64.getEncoder().encodeToString(privateKey.getEncoded());
        }

        public static PrivateKey getPrivateKey(String privateKey) throws Exception {
            byte[] keyBytes = Base64.getDecoder().decode(privateKey);
            PKCS8EncodedKeySpec keySpec = new PKCS8EncodedKeySpec(keyBytes);
            KeyFactory keyFactory = KeyFactory.getInstance(Keys.KEY_ALGORITHM);
            return keyFactory.generatePrivate(keySpec);
        }

        public static PublicKey getPublicKey(String publicKey) throws Exception {
            byte[] keyBytes = Base64.getDecoder().decode(publicKey);
            X509EncodedKeySpec keySpec = new X509EncodedKeySpec(keyBytes);
            KeyFactory keyFactory = KeyFactory.getInstance(Keys.KEY_ALGORITHM);
            return keyFactory.generatePublic(keySpec);
        }
    }
} PublicKey getPublicKey(String publicKey) throws Exception {
            byte[] keyBytes = Base64.getDecoder().decode(publicKey);
            X509EncodedKeySpec keySpec = new X509EncodedKeySpec(keyBytes);
            KeyFactory keyFactory = KeyFactory.getInstance(Keys.KEY_ALGORITHM);
            return keyFactory.generatePublic(keySpec);
        }
    }
}