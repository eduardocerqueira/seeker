//date: 2021-09-27T16:55:52Z
//url: https://api.github.com/gists/aa5c1d26753205b6bd0d3af64e86234b
//owner: https://api.github.com/users/esend7881

package com.ericsender;

import com.google.common.base.Equivalence;
import com.google.common.collect.Iterators;

import java.util.*;

/**
 * Combined LinkedHashMap with IdentityHashMap.
 * Uses the Google common's Equivalence method of implementing IdentityHashMaps.
 * <p>
 * Based on https://stackoverflow.com/a/46561815/1582712 (author: https://stackoverflow.com/users/1450343/moddyfire)
 */
public class IdentityLinkedHashMap<K, T> extends AbstractMap<K, T> {

    private final IdentityLinkedHashSet set = new IdentityLinkedHashSet();

    @Override
    public Set<Entry<K, T>> entrySet() {
        return set;
    }

    @Override
    public T put(K k, T t) {
        return set.innerMap.put(Equivalence.identity().wrap(k), t);
    }

    @Override
    public boolean containsKey(Object arg0) {
        return set.contains(arg0);
    }

    @Override
    public T remove(Object arg0) {
        return set.innerMap.remove(Equivalence.identity().wrap(arg0));
    }

    @Override
    public T get(Object arg0) {
        return set.innerMap.get(Equivalence.identity().wrap(arg0));
    }

    private class IdentityLinkedHashMapEntry implements Entry<K, T> {

        private final Entry<Equivalence.Wrapper<K>, T> entry;

        public IdentityLinkedHashMapEntry(Entry<Equivalence.Wrapper<K>, T> entry) {
            this.entry = entry;
        }

        @Override
        public K getKey() {
            return entry.getKey().get();
        }

        @Override
        public T getValue() {
            return entry.getValue();
        }

        @Override
        public T setValue(T value) {
            return entry.setValue(value);
        }
    }

    private class IdentityLinkedHashSet extends AbstractSet<Entry<K, T>> {

        private final Map<Equivalence.Wrapper<K>, T> innerMap = new LinkedHashMap<>();

        @Override
        public Iterator<Entry<K, T>> iterator() {
            return Iterators.transform(innerMap.entrySet().iterator(), IdentityLinkedHashMapEntry::new);
        }

        @Override
        public boolean add(Entry<K, T> entry) {
            innerMap.put(Equivalence.identity().wrap(entry.getKey()), entry.getValue());
            return true;
        }

        @Override
        public int size() {
            return innerMap.size();
        }

        @Override
        public boolean contains(Object arg0) {
            return innerMap.containsKey(Equivalence.identity().wrap(arg0));
        }
    }
}