//date: 2024-06-07T16:52:30Z
//url: https://api.github.com/gists/2843257bb5f848d82d1b59406b3610be
//owner: https://api.github.com/users/AyushPorwal10

package com.example.let;

import android.util.Log;

import com.google.api.client.util.Lists;
import com.google.auth.oauth2.GoogleCredentials;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Collections;

public class AccessToken {
    private static final String firebaseMessagingScope = "https://www.googleapis.com/auth/firebase.messaging";

    public String getAccessToken() {
        try {
            String jsonString = "{\n" +
                    "  \"type\": \"service_account\",\n" +
                    "  \"project_id\": \"fir-learn-90292\",\n" +
                    "  \"private_key_id\": \"84e79060b99baa93c342b938383fbdadaaa36104\",\n" +
                    "  \"private_key\": \"-----BEGIN PRIVATE KEY-----\\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCIJZIjrCCBfg9o\\nUa6YGL+FQ+RHrPddcX8uyz+bUhqWVYiZuSvLb5V8bkX8R/EbVX93gqlhmA4cSlse\\nlCTSaGuDx+H1b20/Jpf6P2NGZAnPE7yptDlNEBiT3XdJ7hWmHwo8ckzeHqP4pkF6\\nwYka8SgM0lz8MkK8I6EP7zEwXnWfahOzCQqjHKG03XIaTx4ofnILaEYYI2SRafIJ\\nIStKwcamVXucrPfB58Q82eh8SWZSI7pKBZ8b7DWpSlMmMkaj3UCMf8284VXZ4Wer\\nhwAN9KzHsm9vfdG+FHxDZakhASvfKnSIJdijO9uAtZ6fppDEK+qv/iFryqzZSU4p\\nlCfCeJixAgMBAAECggEAAaN4HotKCdzvSQlgoU588ZjnVLCBeqOszO6fyZoDnWcS\\noYz/uK9TXH/EQlzIS1SsV88gBD9s/gysC5JrXgfpMVkUwFwys6WegyHDq1t1XcNm\\nTCGR5fpJXXA3KRHfO1RYznDuuWajgRDZLWXKZWNdIMhgW52sPXDtVZCCpksgvcYn\\nz5+p4glQM5rUvqRq3CD0zj/S8DbJS2WmUsBg8ap8YY2PM+3k8vi+nJulNk9ofr12\\nKXqgCgpXfbI6mnAd2fNIMPJ36J2vOU2tXML3A85xvvW95jC5v5v4vo2zk0p4A0Hs\\ndscVFS7wigxPVrHta3ciHmkMtTijW11LsqyhVabetQKBgQC7N5Tj63svP23MIwFW\\nwJ0lj0rxAnzZR1MA4KF7b473VOYv+Kg1N+cgq5M1mCjBLBpnbjnaHH+Vp7XzscKU\\njBk/xskgar0jog6P9fDu9RR8AJu3GZjtY5gQuWWgmaU8XFXRthfH2CHGafTPRTYC\\nML+sqnDgkfKmZaRjH7ew1u2w5QKBgQC6KqU9Sm6oHprIdatAsliY7yFo3uN6bwbi\\nv0ZBu70PFRXLhc12EQoa+UvzQBo3j18fHCD2MlagJ/gUQ5ljsFlBsSISzQozrroz\\n/I8+CtoBvzjYU75zdW892TYc0QVXN41nR16vGC26NwPehmm0LdEniA2UgCbecsiE\\nz37TKsQn3QKBgQC2h4JKlRQNBLJwDNEJW9HbBNH0GJDQ5pEukdPfHO0uhz/GFZEq\\nEc7uM1nbLvbNH8q+fOE6nf5mUpU7e1xSqCUV4SHG0UqGq0G3afn1gEzweUdYRUSs\\nbiWcaKWE50gKiZvCUt7soPSNFlDwpHH7wLugBKz4xlLlmMOlQQ8/As3LYQKBgAu2\\n+0bsFCKIKn1Kykf78Q9OnO+YdwARVIGYP7eLNM5qKUDxXoh7cgNYhKr98ahlYTr3\\n7isP59uUKEw+JLzdMACuQNKmDGpMKHN1BR6GWEmb2tviCS4Cyck+jeUqUge2+zLw\\njsi94MLDC39JPgWUjIDMUu5xUgDVgEC7PePT4RwRAoGABK9GwD5kikrjAz7Y0MHc\\nco2yq4CDvPH3wKu6twnP6orOsi2jZ5Jo8LTgb7CbVdUmKxCr3N/u3achAHwMBfPa\\n0vlElsJv9DDMTryBj7wuSCTYqvC+Zy3Plvsjr8tFrOi5FOeyBYRPy5M4IqlYHqiM\\nVdusfYyf5csW5znDRWF2J9c=\\n-----END PRIVATE KEY-----\\n\",\n" +
                    "  \"client_email\": \"firebase-adminsdk-mzd4u@fir-learn-90292.iam.gserviceaccount.com\",\n" +
                    "  \"client_id\": \"117078028300879419094\",\n" +
                    "  \"auth_uri\": \"https://accounts.google.com/o/oauth2/auth\",\n" +
                    "  \"token_uri\": "**********"://oauth2.googleapis.com/token\",\n" +
                    "  \"auth_provider_x509_cert_url\": \"https://www.googleapis.com/oauth2/v1/certs\",\n" +
                    "  \"client_x509_cert_url\": \"https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-mzd4u%40fir-learn-90292.iam.gserviceaccount.com\",\n" +
                    "  \"universe_domain\": \"googleapis.com\"\n" +
                    "}\n}";

            InputStream stream = new ByteArrayInputStream(jsonString.getBytes(StandardCharsets.UTF_8));


            GoogleCredentials googleCredentials = GoogleCredentials.fromStream(stream).createScoped(Collections.singletonList(firebaseMessagingScope));
            googleCredentials.refresh();
            return googleCredentials.getAccessToken().getTokenValue();
        }
        catch (IOException exception){
            Log.e("error",""+exception.getMessage());
            return null;
        }

    }
}
