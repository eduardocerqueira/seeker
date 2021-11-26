//date: 2021-11-26T17:14:56Z
//url: https://api.github.com/gists/0d27cdbbdc6e15ec73235cc7eeadae89
//owner: https://api.github.com/users/joshualibrarian

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import com.fasterxml.jackson.databind.deser.std.StdDeserializer;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

@JsonDeserialize(using = MaybeMultiple.Deserializer.class)
public class MaybeMultiple {
    private static final Logger LOG = LoggerFactory.getLogger(MaybeMultiple.class);

    public static String maybeNull(MaybeMultiple value, Enum<?>... criteria) {
        try {
            return value.getValue(criteria);
        } catch (NullPointerException npe) {
            LOG.warn("missing configuration value", npe);
        }
        return null;
    }

    String singleValue = null;
    Map<String, Object> multipleValues = null;

    public MaybeMultiple(Object value) {
        if (value instanceof String) {
            singleValue = (String) value;
        } else if (value instanceof Map) {
            multipleValues = (Map<String, Object>) value;
        } else {
            throw new IllegalArgumentException("argument must be either String or Map<String, Object>");
        }
    }

    public String getValue(Enum<?>... criteria) {
        if (singleValue != null) {
            return singleValue;
        }
        return getValue(multipleValues, criteria);
    }

    private String getValue(Map<String, Object> values, Enum<?>... criteria) {
        for (Enum<?> c : criteria) {
            Object o = values.get(c.name());
            //TODO: add error handling here in case bad config
            if (o instanceof String) {
                return (String) o;
            } else if (o instanceof Map) {
                return getValue((Map<String, Object>) o, criteria);
            }
        }
        return null;
    }

    public static class Deserializer extends StdDeserializer<MaybeMultiple> {
        Deserializer() {
            this(null);
        }
        Deserializer(Class<MaybeMultiple> vc) {
            super(vc);
        }

        @Override
        public MaybeMultiple deserialize(JsonParser jsonParser, DeserializationContext context)
                throws IOException, JsonProcessingException {
            return new MaybeMultiple(jsonParser.readValueAs(Object.class));
        }
    }

    public static class ListDeserializer extends StdDeserializer<List<MaybeMultiple>> {
        ListDeserializer() {
            this(null);
        }
        ListDeserializer(Class<List<MaybeMultiple>> vc) {
            super(vc);
        }

        @Override
        public List<MaybeMultiple> deserialize(JsonParser p, DeserializationContext context)
                throws IOException, JsonProcessingException {
            Object o = p.readValueAs(Object.class);

            if (o instanceof String) {
                return List.of(new MaybeMultiple(o));
            }

            Map<String, Object> valueMap = (Map<String, Object>) o;
            return valueMap.entrySet().stream()
                    .map(Map::ofEntries)
                    .map(MaybeMultiple::new)
                    .collect(Collectors.toList());
        }
    }
}