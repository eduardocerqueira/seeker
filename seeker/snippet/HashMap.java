//date: 2025-01-07T16:39:11Z
//url: https://api.github.com/gists/856737c5286c2a35c84993b55d6f2bf5
//owner: https://api.github.com/users/YahorDanchanka

import java.util.Objects;

public class HashMap {
    KeyValuePair[] entries = new KeyValuePair[8];
    int size = 8;
    int cursor = 0;

    int hash(String key) {
        return 0;
    }

    void add(String key, String value) {
        int index = findIndex(key);
        entries[index] = new KeyValuePair(key, value);
        cursor++;

        if (cursor == size) {
            resize(size * 2);
        }
    }

    void resize(int newSize) {
        KeyValuePair[] newEntries = new KeyValuePair[newSize];

        for (int i = 0; i < size; i++) {
            int index = findIndex(entries[i].key);
            newEntries[index] = entries[i];
        }

        entries = newEntries;
        size = newSize;
    }

    String get(String key) {
        int index = findIndex(key);

        if (index == -1) {
            return null;
        }

        KeyValuePair entry = entries[index];

        if (entry == null || !Objects.equals(entry.key, key)) {
            return null;
        }

        return entry.value;
    }

    int findIndex(String key) {
        int hash = hash(key);
        int index = hash % size;

        for (int i = 0; i < size; i++) {
            int probingIndex = (index + i) % size;
            KeyValuePair entry = entries[probingIndex];

            if (entry == null || Objects.equals(entry.key, key)) {
                return probingIndex;
            }
        }

        return -1;
    }
}
