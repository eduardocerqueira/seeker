//date: 2023-03-07T17:08:10Z
//url: https://api.github.com/gists/983d3221721477f4808b6941e9dfc4ec
//owner: https://api.github.com/users/eltgm

package yandex.sprint4.final4.hashtable;

import static java.lang.Math.floorMod;
import static java.lang.Math.pow;

public class HashTable {
    private static final int MAX_HASH_TABLE_SIZE = 100000;
    private static final int CONST_1 = 33;
    private static final int CONST_2 = 9321;
    private int elementsCount = 0;

    private final Node[] table = new Node[MAX_HASH_TABLE_SIZE];

    public void put(long key, long value) {
        int bucket = calculateBucket(key);

        Node existNode = table[bucket];
        if (existNode != null && existNode.getKey() != key) {
            bucket = findActualIndexForPut(key, bucket);
        }

        table[bucket] = new Node(key, value);
        elementsCount++;
    }

    public long get(long key) {
        if (elementsCount == 0) {
            return Long.MIN_VALUE;
        }

        int bucket = calculateBucket(key);

        Node existNode = table[bucket];
        if (existNode == null) {
            return Long.MIN_VALUE;
        }

        if (existNode.getKey() != key) {
            int newBucket = findActualIndexForGet(key, bucket);
            if (newBucket == -1) {
                return Long.MIN_VALUE;
            }

            return table[newBucket].getValue();
        } else {
            return existNode.getValue();
        }
    }

    public long delete(long key) {
        if (elementsCount == 0) {
            return Long.MIN_VALUE;
        }

        int bucket = calculateBucket(key);
        Node existNode = table[bucket];
        if (existNode == null) {
            return Long.MIN_VALUE;
        } else {
            table[bucket] = new Node(Long.MIN_VALUE, Long.MIN_VALUE);
            elementsCount--;
            return existNode.getValue();
        }
    }

    private int calculateBucket(long key) {
        return floorMod(key, MAX_HASH_TABLE_SIZE);
    }

    private int findActualIndexForPut(long key, int firstBucket) {
        int newBucket = firstBucket;
        do {
            newBucket = floorMod((int) (key + CONST_1 * newBucket + CONST_2 * pow(newBucket, 2)), MAX_HASH_TABLE_SIZE);
        } while (table[newBucket] != null && table[newBucket].getKey() != Long.MIN_VALUE);

        return newBucket;
    }

    private int findActualIndexForGet(long key, int firstBucket) {
        int newBucket = firstBucket;

        while (true) {
            newBucket = floorMod((int) (key + CONST_1 * newBucket + CONST_2 * pow(newBucket, 2)), MAX_HASH_TABLE_SIZE);
            if (table[newBucket] == null) {
                return -1;
            }

            if (table[newBucket].getKey() == key) {
                return newBucket;
            }
        }
    }

    private static class Node {
        private final long key;
        private final long value;

        public Node(long key, long value) {
            this.key = key;
            this.value = value;
        }

        public long getKey() {
            return key;
        }

        public long getValue() {
            return value;
        }
    }
}
