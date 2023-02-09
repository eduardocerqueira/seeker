//date: 2023-02-09T17:07:44Z
//url: https://api.github.com/gists/cb9f36ee22f86fc070a97fc22c8f05cc
//owner: https://api.github.com/users/cutcell

if (arrayOfSheeps == null || arrayOfSheeps.length == 0) {
    return 0;
}

return Arrays.stream(arrayOfSheeps)
        .filter(Objects::nonNull)
        .filter(Boolean::booleanValue)
        .count();