//date: 2024-12-04T16:51:06Z
//url: https://api.github.com/gists/cf0f9afbcf359df65846a760da64d230
//owner: https://api.github.com/users/avishaybp81

package com.example.additionalproperties;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;


public class AdditionalPropertiesExtractor {

    public static Map<String, String> extractAdditionalProperties(Object obj) {
        Map<String, String> allProperties = new HashMap<>();
        processObject(obj, allProperties, "");
        return allProperties;
    }

    private static void processObject(Object obj, Map<String, String> allProperties, String prefix) {
        if (obj == null) {
            return;
        }

        if (obj instanceof Map) {
            for (Map.Entry<String, Object> entry : ((Map<String, Object>) obj).entrySet()) {
                allProperties.put(prefix + "." + entry.getKey(), entry.getValue().toString());
            }

        } else {
            for (Field field : obj.getClass().getDeclaredFields()) {
                if (field.getType().getPackage().equals(SecurityEvent.class.getPackage()) || field.getType().equals(Map.class)) {
                    try {
                        field.setAccessible(true);
                    } catch (Exception ignored) {
                        continue;
                    }
                    try {
                        Object fieldValue = field.get(obj);
                        System.out.println("--->:  " + fieldValue.getClass().getSimpleName() + " : " + fieldValue);
                        processObject(fieldValue, allProperties, obj.getClass().getSimpleName());
                    } catch (IllegalAccessException e) {
                        // Handle exception, e.g., log the error
                        e.printStackTrace();
                    }

                }

            }
        }
    }
}