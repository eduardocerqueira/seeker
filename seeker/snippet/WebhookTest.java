//date: 2022-06-09T17:05:10Z
//url: https://api.github.com/gists/396909edefa307f64e870281640184d2
//owner: https://api.github.com/users/jcoon97

import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;

public class WebhookTest {
    public static void main(String[] args) throws NoSuchAlgorithmException, InvalidKeyException {
        Validator validator = new Validator("{\"id\":\"e09ad359-91f1-43d0-adc5-c0b1282cdffd\",\"resourceId\":\"061236ff-232b-4d9f-810e-f3ebc12c0f09\",\"topic\":\"customer_created\",\"timestamp\":\"2022-06-09T16:58:07.767Z\",\"_links\":{\"self\":{\"href\":\"https://api-sandbox.dwolla.com/events/e09ad359-91f1-43d0-adc5-c0b1282cdffd\"},\"account\":{\"href\":\"https://api-sandbox.dwolla.com/accounts/78498ad7-9bbb-4eb1-b16c-f3027d7ec821\"},\"resource\":{\"href\":\"https://api-sandbox.dwolla.com/customers/061236ff-232b-4d9f-810e-f3ebc12c0f09\"},\"customer\":{\"href\":\"https://api-sandbox.dwolla.com/customers/061236ff-232b-4d9f-810e-f3ebc12c0f09\"}},\"created\":\"2022-06-09T16:58:07.767Z\"}", "ThisIsASecret", "66dba6225855be1665f00e63768cf76506c692fe7b443183b2741342ef496d36");
        System.out.println("Is Valid? " + validator.validate());
    }
}
