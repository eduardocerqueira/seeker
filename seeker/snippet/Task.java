//date: 2025-04-23T17:07:05Z
//url: https://api.github.com/gists/32a75d6f15e393ac329254c33ee958d5
//owner: https://api.github.com/users/kavicastelo

public class Task {
    private Long id;
    
    @NotBlank(message = "Title is required")
    private String title;

    private boolean completed = false;

    // constructor, getters, setters
}
