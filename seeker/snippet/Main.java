//date: 2024-12-04T16:51:06Z
//url: https://api.github.com/gists/cf0f9afbcf359df65846a760da64d230
//owner: https://api.github.com/users/avishaybp81

package com.example.additionalproperties;

import com.example.additionalproperties.SecurityEvent;
import com.example.additionalproperties.AdditionalPropertiesExtractor;

import java.util.HashMap;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
        SecurityEvent event = new SecurityEvent();
        event.setEventId("event123");
        event.setEventType("security_event");

        // Set top-level additional properties
        Map<String, String> topLevelProps = new HashMap<>();
        topLevelProps.put("prop1", "value1");
        topLevelProps.put("prop2", "value2");
        event.setAdditionalProperties(topLevelProps);

        // Create nested event
        SecurityEvent.NestedSecurityEvent nestedEvent = new SecurityEvent.NestedSecurityEvent();
        nestedEvent.setNestedId("nested123");
        nestedEvent.setNestedType("nested_event");

        // Set nested additional properties
        Map<String, String> nestedProps = new HashMap<>();
        nestedProps.put("nestedProp1", "nestedValue1");
        nestedProps.put("nestedProp2", "nestedValue2");
        nestedEvent.setAdditionalProperties(nestedProps);

        // Create deeply nested event
        SecurityEvent.NestedSecurityEvent.DeeplyNestedSecurityEvent deeplyNestedEvent = new SecurityEvent.NestedSecurityEvent.DeeplyNestedSecurityEvent();
        deeplyNestedEvent.setDeeplyNestedId("deeplyNested123");
        deeplyNestedEvent.setDeeplyNestedType("deeplyNested_event");

        // Set deeply nested additional properties
        Map<String, String> deeplyNestedProps = new HashMap<>();
        deeplyNestedProps.put("deeplyNestedProp1", "deeplyNestedValue1");
        deeplyNestedProps.put("deeplyNestedProp2", "deeplyNestedValue2");
        deeplyNestedEvent.setAdditionalProperties(deeplyNestedProps);

        nestedEvent.setDeeplyNestedEvent(deeplyNestedEvent);
        event.setNestedEvent(nestedEvent);

        // Extract additional properties
        Map<String, String> extractedProps = AdditionalPropertiesExtractor.extractAdditionalProperties(event);

        // Print extracted properties

        System.out.println("Result below:");
        extractedProps.forEach((key, value) -> System.out.println("\t" + key + ": " + value));
    }
}