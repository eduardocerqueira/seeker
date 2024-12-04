//date: 2024-12-04T16:51:06Z
//url: https://api.github.com/gists/cf0f9afbcf359df65846a760da64d230
//owner: https://api.github.com/users/avishaybp81

package com.example.additionalproperties;

import java.util.Map;

public class SecurityEvent {
    private String eventId;
    private String eventType;
    private Map<String, String> additionalProperties;
    private NestedSecurityEvent nestedEvent;

    // Getters and Setters
    public String getEventId() {
        return eventId;
    }

    public void setEventId(String eventId) {
        this.eventId = eventId;
    }

    public String getEventType() {
        return eventType;
    }

    public void setEventType(String eventType) {
        this.eventType = eventType;
    }

    public Map<String, String> getAdditionalProperties() {
        return additionalProperties;
    }

    public void setAdditionalProperties(Map<String, String> additionalProperties) {
        this.additionalProperties = additionalProperties;
    }

    public NestedSecurityEvent getNestedEvent() {
        return nestedEvent;
    }

    public void setNestedEvent(NestedSecurityEvent nestedEvent) {
        this.nestedEvent = nestedEvent;
    }

    // NestedSecurityEvent Class
    public static class NestedSecurityEvent {
        private String nestedId;
        private String nestedType;
        private Map<String, String> additionalProperties;
        private DeeplyNestedSecurityEvent deeplyNestedEvent;

        // Getters and Setters
        public String getNestedId() {
            return nestedId;
        }

        public void setNestedId(String nestedId) {
            this.nestedId = nestedId;
        }

        public String getNestedType() {
            return nestedType;
        }

        public void setNestedType(String nestedType) {
            this.nestedType = nestedType;
        }

        public Map<String, String> getAdditionalProperties() {
            return additionalProperties;
        }

        public void setAdditionalProperties(Map<String, String> additionalProperties) {
            this.additionalProperties = additionalProperties;
        }

        public DeeplyNestedSecurityEvent getDeeplyNestedEvent() {
            return deeplyNestedEvent;
        }

        public void setDeeplyNestedEvent(DeeplyNestedSecurityEvent deeplyNestedEvent) {
            this.deeplyNestedEvent = deeplyNestedEvent;
        }

        // DeeplyNestedSecurityEvent Class
        public static class DeeplyNestedSecurityEvent {
            private String deeplyNestedId;
            private String deeplyNestedType;
            private Map<String, String> additionalProperties;

            // Getters and Setters
            public String getDeeplyNestedId() {
                return deeplyNestedId;
            }

            public void setDeeplyNestedId(String deeplyNestedId) {
                this.deeplyNestedId = deeplyNestedId;
            }

            public String getDeeplyNestedType() {
                return deeplyNestedType;
            }

            public void setDeeplyNestedType(String deeplyNestedType) {
                this.deeplyNestedType = deeplyNestedType;
            }

            public Map<String, String> getAdditionalProperties() {
                return additionalProperties;
            }

            public void setAdditionalProperties(Map<String, String> additionalProperties) {
                this.additionalProperties = additionalProperties;
            }
        }
    }
}

