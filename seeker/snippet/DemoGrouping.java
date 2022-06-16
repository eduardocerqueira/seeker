//date: 2022-06-16T17:11:34Z
//url: https://api.github.com/gists/d6867f4f5320c3f47c3aa6010f06f14f
//owner: https://api.github.com/users/tirmizee

public static void main(String...args) {
    Map<String, List<Product>> map = new HashMap<>();
    for (Product mock: mocks) {

        if(map.get(mock.getName()) == null) {
            map.put(mock.getName(), new ArrayList<>());
        }

        map.get(mock.getName()).add(mock);
    }

    for (Map.Entry<String, List<Product>> entry: map.entrySet()) {
        System.out.println(entry.getKey() + ":" + entry.getValue());
    }

}

// Golang:[Product{name='Golang', price=1000, type='A', isActive=true}, Product{name='Golang', price=8000, type='B', isActive=true}]
// Java:[Product{name='Java', price=1000, type='A', isActive=true}, Product{name='Java', price=900, type='A', isActive=true}, Product{name='Java', price=900, type='B', isActive=true}]
// Angular:[Product{name='Angular', price=9000, type='A', isActive=true}]