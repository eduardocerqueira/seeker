//date: 2024-01-04T16:50:17Z
//url: https://api.github.com/gists/38bcb7d058aabb18d7c5ec31ccde6ed4
//owner: https://api.github.com/users/szbenceg

@GetMapping(value = "/user-details")
public Mono<UserDetails> sendMessage(@AuthenticationPrincipal Jwt principal) {

  String id = principal.getSubject();
  return userDao.findByUserId(id)
          .map(user -> {
            return UserDetails
                    .builder()
                     .id(user.getId().toString())
                     .username(user.getUsername())
                     .profilePictureName(configContext.getProfilePictureUrlPath(user.getProfilePictureName()))
                     .build();
          });

}