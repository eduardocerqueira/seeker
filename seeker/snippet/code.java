//date: 2024-06-18T17:05:58Z
//url: https://api.github.com/gists/5cfe6f1f75855490047931769699522b
//owner: https://api.github.com/users/Billthekidz

UserManager userManager = UserManager.get(this);
Optional<UserInfo> currentUser = userManager.getUsers().stream()
    .filter(userInfo -> userInfo.id != 0)
    .filter(userInfo -> userManager.isUserRunning(userInfo.id))
    .findFirst();
