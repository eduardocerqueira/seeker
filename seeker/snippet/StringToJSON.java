//date: 2023-02-17T17:11:25Z
//url: https://api.github.com/gists/ceaada070a8cdc5b9c2d801103a743a2
//owner: https://api.github.com/users/asingh403

public class StringToJSON {
    public static void main(String[] args) {
        String customerId = "10224";
        String transactionId = "ejyutreshfg";

        String result = stringToJson(customerId,transactionId);
        System.out.println(result);
    }
    private static String stringToJson(String customerId, String transactionId) {
        String res = "{\"customerId\": "+customerId+",\"transactionId\": "+transactionId+"}";
        return res;
    }
}