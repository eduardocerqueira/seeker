//date: 2023-04-10T16:51:46Z
//url: https://api.github.com/gists/731de61ff8e238264915d05e8cc58322
//owner: https://api.github.com/users/Vergil333

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.lang.reflect.Field;
import java.net.*;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @credits to mjg123 https://github.com/mjg123/java-http-clients/blob/master/src/main/java/com/twilio/JavaHttpURLConnectionDemo.java
 * @credits to madmax1028, broot https://discuss.kotlinlang.org/t/how-to-specify-generic-output-type-to-be-subclass-of-generic-type/24637/12
 */
public class SalesforceClient {

    /**
    * @param accessToken has to have [web, chatter_api, api] permissions for querying
    * @param instanceUrl acquired in the same response as accessToken
    */
    public SalesforceClient(String accessToken, String instanceUrl) {
        this.accessToken = "**********"
        this.instanceUrl = instanceUrl;
    }

    private String accessToken;
    private String instanceUrl;
    private String apiVersion = "v53.0";
    public ObjectMapper mapper = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    public <T extends SObjectInterface> SfResponse<T> getAll(Class<T> clazz) throws IOException, URISyntaxException {
        String joinedFields = String.join(",", this.getFields(clazz));
        String query = "SELECT " + joinedFields + " FROM " + clazz.getSimpleName();
        return getResponse(query, clazz);
    }

    public <T extends SObjectInterface> Integer count(Class<T> clazz) throws IOException, URISyntaxException {
        String query = "SELECT COUNT() FROM " + clazz.getSimpleName();
        return getResponse(query, clazz).getTotalSize();
    }

    private <T extends SObjectInterface> SfResponse<T> getResponse(String query, Class<T> clazz) throws IOException, URISyntaxException {
        URL url = constructUrl(query);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();

        connection.setRequestProperty("accept", "application/json");
        connection.setRequestProperty("Authorization", "Bearer " + accessToken);

        JavaType wrappedResponseType = mapper.getTypeFactory().constructParametricType(SfResponse.class, clazz);

        return mapper.readValue(connection.getInputStream(), wrappedResponseType);
    }

    private URL constructUrl(String query) throws MalformedURLException, URISyntaxException {
        URL url = new URL(instanceUrl + "/services/data/" + apiVersion + "/query?q=" + query);
        URI uri = toUri(url);
        return uri.toURL();
    }

    /**
     * URI is used for URL encoding.
     */
    private URI toUri(URL url) throws URISyntaxException {
        return new URI(
                url.getProtocol(),
                url.getUserInfo(),
                IDN.toASCII(url.getHost()),
                url.getPort(),
                url.getPath(),
                url.getQuery(),
                url.getRef()
        );
    }

    private <T extends SObjectInterface> List<String> getFields(Class<T> clazz) {
        List<Field> fields = Arrays.stream(clazz.getFields()).filter(field -> Arrays.asList("java.util", "java.lang")
                        .contains(field.getType().getPackageName())
                        && !field.getName().equals("attributes")
                )
                .collect(Collectors.toList());

        List<String> jsonFields = fields.stream()
                .map(field -> field.getAnnotation(JsonProperty.class).value())
                .collect(Collectors.toList());

        return jsonFields;
    }
}