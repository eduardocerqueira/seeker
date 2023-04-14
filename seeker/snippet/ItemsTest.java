//date: 2023-04-14T17:07:20Z
//url: https://api.github.com/gists/86a7211c135097f58c563d2c9ec22fad
//owner: https://api.github.com/users/maartenl

  @Test
  public void testForEachLoopMax() {
    Item foundItem = null;
    for (Item item: items) {
      if (foundItem == null || foundItem.getPrice().isGreaterThan(item.getPrice())) {
        foundItem = item;
      }
    }
    assertThat(foundItem).isEqualTo(items.get(4));
  }