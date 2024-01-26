//date: 2024-01-26T17:04:15Z
//url: https://api.github.com/gists/e36d243d975fe5bc5c11ec4ba33237da
//owner: https://api.github.com/users/delta-dev-software

import javax.persistence.Entity;
import javax.persistence.Id;

@Entity
public class Book {
    @Id
    private Long id;
    private String title;
    private String author;

    // Getters and setters
}