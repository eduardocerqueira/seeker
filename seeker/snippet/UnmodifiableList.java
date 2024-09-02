//date: 2024-09-02T16:44:47Z
//url: https://api.github.com/gists/80bdaec048a85c75aea97e18c2230ce9
//owner: https://api.github.com/users/cyurtoz

// Creating a mutable ArrayList
List<Integer> originalList = new ArrayList<>();
originalList.add(1);
originalList.add(2);

// Creating an unmodifiableList on top of the mutable ArrayList
List<Integer> unmodifiableList = Collections.unmodifiableList(originalList);
System.out.println(unmodifiableList); // prints [1, 2]

unmodifiableList.add(3); // throws java.lang.UnsupportedOperationException

// add 3 to underlying list 
originalList.add(3);
System.out.println(unmodifiableList); // prints [1, 2, 3] 
// data on unmodifiableList has changed