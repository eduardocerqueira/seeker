//date: 2021-08-31T02:41:29Z
//url: https://api.github.com/gists/36faa3f501a0bb73709c62cc8d15fc67
//owner: https://api.github.com/users/patrot

// The following code is a series of test that were written in the order you'll read them. After
// each test was written it was run against the HashMap and the implementation was updated if it
// failed. The HashMap code is the final implementation that was built naively to pass each test.

// The Tests:

public class HashMapTest {
    private HashMap map;
  
    // Set up an empty map before each test  
    @Before
    public void setUp() {
        this.map = new HashMap();
    }

    // Check that a new HashMap returns 'true' for isEmpty
    @Test
    public void testIsEmptyForNewMap() {
        assertTrue(map.isEmpty());
    } 
    
    // Test adding an element makes isEmpty return 'false'
    @Test
    public void testAddMakesIsEmptyFalse() {
        map.add("Hello", 5);
        assertFalse(map.isEmpty());
    }
    
    // Check that size returns 0 for new HashMaps
    @Test
    public void testSizeForNewMap() {
        assertEquals(0, map.size());
    }
    
    // Test size increases as elements are added
    @Test
    public void testSizeIncrementsWhenAddingElements() {
        map.add("Hello", 5);
        assertEquals(1, map.size());
        
        map.add("Goodbye", 5);
        assertEquals(2, map.size());
    }
    
    // Make sure get returns the values added under keys
    @Test
    public void testGetReturnsCorrectValue() {
        map.add("Hello", 5);
        map.add("Goodbye", 6);
        assertEquals(5, map.get("Hello"));
        assertEquals(6, map.get("Goodbye"));
    }
    
    // Test that an exception is thrown if a key does not exist
    @Test(expected= NoSuchElementException.class)
    public void testThrowsExceptionIfKeyDoesNotExist() {
        map.get("Hello");
    }

    // Test thats an added element replaces another with the same key
    @Test
    public void testReplacesValueWithSameKey() {
        map.add("Hello", 5);
        map.add("Hello", 6);
        
        assertEquals(6, map.get("Hello"));
    }
    
    // Make sure that two (non-equal) keys with the same hash do not overwrite each other
    @Test
    public void testDoesNotOverwriteSeperateKeysWithSameHash() {
        map.add("Ea", 5);
        map.add("FB", 6);
        
        assertEquals(5, map.get("Ea"));
        assertEquals(6, map.get("FB"));
    }
    
    // Make sure size doesn't decrement below 0
    @Test
    public void testRemoveDoesNotEffectNewMap() {
        map.remove("Hello");

        assertEquals(0, map.size());
    }
    
    // Make sure that size decrements as elements are used
    @Test
    public void testRemoveDecrementsSize() {
        map.add("Hello", 5);
        map.add("Goodbye", 6);
        
        map.remove("Hello");
        
        assertEquals(1, map.size());
        
        map.remove("Goodbye");
        
        assertEquals(0, map.size());
    }

    // Test elements are actually removed when remove is called
    @Test(expected= NoSuchElementException.class)
    public void testRemoveDeletesElement() {
        map.add("Hello", 5);
        map.remove("Hello");
        
        map.get("Hello");
    }
    
    // Test that contains is 'false' for new maps
    @Test
    public void testContainsKeyForNewMap() {
        assertFalse(map.containsKey("Hello"));
    }
    
    // Test that contains returns 'false' when key doesn't exist
    @Test
    public void testContainsKeyForNonExistingKey() {
        map.add("Hello", 5);
        assertFalse(map.containsKey("Goodbye"));
    }

    // Make sure that contains returns 'true' when the key does exist
    @Test
    public void testContainsKeyForExistingKey() {
        map.add("Hello", 5);
        assertTrue(map.containsKey("Hello"));
    }
    
    // Check that contains is not fooled by equivalent hash codes
    @Test
    public void testContainsKeyForKeyWithEquivalentHash() {
        map.add("Ea", 5);
        
        assertFalse(map.containsKey("FB"));
    }
}

// The HashMap:

public class HashMap {
    private int size;
    private ArrayList<LinkedList<Element>> array;

    public HashMap() {
        this.size = 0;
        this.array = new ArrayList<LinkedList<Element>>(128);

        for (int i = 0; i < 128; i++) {
            this.array.add(new LinkedList<Element>());
        }
    }

    public boolean isEmpty() {
        return size() == 0;
    }

    public void add(String key, Object value) {
        this.remove(key);

        array.get(hash(key)).add(new Element(key, value));
        size++;
    }
    
    public boolean containsKey(String key) {
        for (Element ele : this.array.get(hash(key))) {
            if (ele.key.equals(key)) {
                return true;
            }
        }

        return false;
    }

    public int size() {
        return size;
    }

    public Object get(String key) {
        for (Element ele : this.array.get(hash(key))) {
            if (ele.key.equals(key)) {
                return ele.value;
            }
        }

        throw new NoSuchElementException();
    }
    
    private int hash(String key) {
        return key.hashCode() % 128;
    }

    public void remove(String key) {
        for (Element ele : this.array.get(hash(key))) {
            if (ele.key.equals(key)) {
                this.array.get(hash(key)).remove(ele);
                size--;
                break;
            }
        }
    }

    private class Element {
        public final String key;
        public final Object value;
        
        public Element(String key, Object value) {
            this.key = key;
            this.value = value;
        }
    }
}
