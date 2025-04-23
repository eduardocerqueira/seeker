//date: 2025-04-23T17:05:00Z
//url: https://api.github.com/gists/219ba3ae92e83aeba6f375b09b877adf
//owner: https://api.github.com/users/kavicastelo

public class User {

    @NotBlank(message = "Name is required")
    private String name;

    @Email(message = "Invalid email format")
    private String email;

    @Min(value = 18, message = "Minimum age is 18")
    private int age;

    // getters and setters
}
