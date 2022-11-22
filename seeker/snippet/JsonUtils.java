//date: 2022-11-22T17:06:36Z
//url: https://api.github.com/gists/debab0ad6db20b01d05863c84b480e90
//owner: https://api.github.com/users/petercao

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.*;
import com.meituan.conch.account.exception.JsonException;
import org.apache.log4j.Logger;

import java.io.InputStream;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.Map;

/**
 * JSON Serialization & Deserialization Utils Powered by Jackson
 */
public final class JsonUtils {

    private static final Logger LOGGER = Logger.getLogger(JsonUtils.class);
    private static ObjectMapper mapper = new ObjectMapper();
    private static ObjectMapper mapperSnake = new ObjectMapper();

    static {
        mapper.configure(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS, false);
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        mapper.configure(JsonParser.Feature.ALLOW_UNQUOTED_CONTROL_CHARS, true);
        mapper.configure(JsonParser.Feature.ALLOW_SINGLE_QUOTES, true);
        mapper.setSerializationInclusion(JsonInclude.Include.NON_NULL);

        mapperSnake.configure(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS, false);
        mapperSnake.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        mapperSnake.configure(JsonParser.Feature.ALLOW_UNQUOTED_CONTROL_CHARS, true);
        mapperSnake.configure(JsonParser.Feature.ALLOW_SINGLE_QUOTES, true);
        mapperSnake.setSerializationInclusion(JsonInclude.Include.NON_NULL);
        mapperSnake.setPropertyNamingStrategy(PropertyNamingStrategy.SNAKE_CASE);

    }

    private JsonUtils() {
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> readValue(String str) {
        try {
            return mapper.readValue(str, HashMap.class);
        } catch (Exception e) {
            LOGGER.error(e.getMessage());
            throw new JsonException(e);
        }
    }

    public static <T> T readValue(String str, Class<T> cls) {
        try {
            return mapper.readValue(str, cls);
        } catch (Exception e) {
            LOGGER.error(e.getMessage());
            throw new JsonException(e);
        }
    }

    @SuppressWarnings("unchecked")
    public static <T> T readValue(String str, TypeReference<T> t) {
        try {
            return mapper.readValue(str, t);
        } catch (Exception e) {
            LOGGER.error(e.getMessage());
            throw new JsonException(e);
        }
    }

    @SuppressWarnings("unchecked")
    public static <T> T readValue(String str, JavaType t) {
        try {
            return mapper.readValue(str, t);
        } catch (Exception e) {
            LOGGER.error(e.getMessage());
            throw new JsonException(e);
        }
    }

    public static <T> T readValue(InputStream is, Class<T> cls) {
        try {
            return mapper.readValue(is, cls);
        } catch (Exception e) {
            LOGGER.error(e.getMessage());
            throw new JsonException(e);
        }
    }

    @SuppressWarnings("unchecked")
    public static <T> T readValue(Object object, Class<T> cls) {
        try {
            return mapper.convertValue(object, cls);
        } catch (Exception e) {
            LOGGER.error(e.getMessage());
            throw new JsonException(e);
        }
    }

    public static <T> String writeValue(T o) {
        try {
            return mapper.writeValueAsString(o);
        } catch (Exception e) {
            throw new JsonException(e);
        }
    }

    public static <T> String writeValue(T o, boolean pretty) {
        try {
            StringWriter sw = new StringWriter();
            JsonGenerator generator = getFactory().createGenerator(sw);
            if (pretty) {
                generator.useDefaultPrettyPrinter();
            }
            mapper.writeValue(generator, o);
            return sw.toString();
        } catch (Exception e) {
            LOGGER.error(e.getMessage());
            throw new JsonException(e);
        }
    }

    public static JsonFactory getFactory() {
        return mapper.getFactory();
    }

    public static ObjectMapper getObjectMapper() {
        return mapper;
    }

    @SuppressWarnings("unchecked")
    public static <T> T readValueSnakeCase(String str, TypeReference<T> t) {
        try {
            return mapperSnake.readValue(str, t);
        } catch (Exception e) {
            LOGGER.error(e.getMessage());
            throw new JsonException(e);
        }
    }
}
