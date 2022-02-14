//date: 2022-02-14T16:51:12Z
//url: https://api.github.com/gists/9f38ce2edabb838b9e918d8fc1246f12
//owner: https://api.github.com/users/recursivecodes

@MappedEntity
@AllArgsConstructor(access = AccessLevel.PACKAGE)
@NoArgsConstructor
@Data
@EqualsAndHashCode
public class Movie {
    @Id
    @GeneratedValue
    private String id;
    private String title;
    private String description;
    private Integer rating;
    private Integer runtimeMinutes;
    private LocalDateTime releasedOn;
}