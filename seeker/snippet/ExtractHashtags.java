//date: 2022-05-11T17:19:47Z
//url: https://api.github.com/gists/8e7297794fd4d82a68397837a691392f
//owner: https://api.github.com/users/B0BAI

  public static @NotNull List<String> extractHashtags(final @NotNull String text) {
    final var hashTags = new ArrayList<String>();

    Arrays.stream(text.split("\\s"))
        .filter(word -> word.startsWith(HASH))
        .distinct()
        .parallel()
        .forEachOrdered(
            word -> {
              if (word.matches("[#A-Za-z0-9]+") && word.length() > 1) {
                hashTags.add(word);
              } else {
                final var wordChar = word.toCharArray();
                final var wordBuilder = new StringBuilder().append(wordChar[0]);
                for (int i = 1; i < wordChar.length; ++i) {
                  if (Character.isAlphabetic(wordChar[i]) || Character.isDigit(wordChar[i])) {
                    wordBuilder.append(wordChar[i]);
                  } else {
                    hashTags.add(wordBuilder.toString());
                    break;
                  }
                }
              }
            });

    return hashTags;
  }