//date: 2022-07-01T17:17:30Z
//url: https://api.github.com/gists/21270b137c80524f1f3d29651099f748
//owner: https://api.github.com/users/arafsheikh

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class TreeMultiMap<K extends Comparable<K>, V> {
    private final TreeMap<K, List<V>> treeMap;
    int size;

    public TreeMultiMap() {
        treeMap = new TreeMap<>(Comparable::compareTo);
        size = 0;
    }

    public void put(K key, V value) {
        if (treeMap.containsKey(key)) {
            treeMap.get(key).add(value);
        } else {
            treeMap.put(key, new ArrayList<>() {{
                add(value);
            }});
        }
        size++;
    }

    public V removeFirst() {
        K key = treeMap.firstKey();

        if (key == null) {
            return null;
        }

        size--;
        if (treeMap.get(key).size() == 1) {
            V value = treeMap.remove(key).get(0);
            treeMap.remove(key);
            return value;
        } else {
            return treeMap.get(key).remove(0);
        }
    }

    public List<Map.Entry<K, V>> getAllDescending() {
        List<Map.Entry<K, V>> entries = new ArrayList<>();

        for (Map.Entry<K, List<V>> entry : treeMap.descendingMap().entrySet()) {
            for (V value : entry.getValue()) {
                entries.add(new Map.Entry<>() {
                    @Override
                    public K getKey() {
                        return entry.getKey();
                    }

                    @Override
                    public V getValue() {
                        return value;
                    }

                    @Override
                    public V setValue(V value) {
                        throw new UnsupportedOperationException();
                    }
                });
            }
        }

        return entries;
    }

    public int size() {
        return size;
    }
}