//date: 2022-10-28T17:02:07Z
//url: https://api.github.com/gists/7c006b236bdcb72647cb318281ccc961
//owner: https://api.github.com/users/instancio

// Getters and setters omitted for brevity
class Person {
    private String name;
    private Address address;
    private LocalDateTime lastUpdated;
}
class Address {
    private String street;
    private String city;
    private String country;
    private List<Phone> phoneNumbers;
    private LocalDateTime lastUpdated;
}
class Phone {
    private String countryCode;
    private String number;
    private LocalDateTime lastUpdated;
}