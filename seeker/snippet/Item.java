//date: 2023-04-14T17:05:25Z
//url: https://api.github.com/gists/5a791b91bbd44607fe09c496e5fdc8f3
//owner: https://api.github.com/users/maartenl

  public class Item {
    private final String name;
    private final MonetaryAmount price;

    private Item(String name, MonetaryAmount price) {
      this.name = name;
      this.price = price;
    }

    public MonetaryAmount getPrice() {
      return price;
    }

    public String getName() {
      return name;
    }
  }

  public static final CurrencyUnit euro = Monetary.getCurrency("EUR");

  public final List<Item> items = Arrays.asList(new Item("French fries", Money.of(2.5, euro)),
      new Item("Frikandel", Money.of(2.25, euro)),
      new Item("Hamburger Classic Double", Money.of(5.5, euro)),
      new Item("Hamburger Menu (French fries, hamburger, soda)", Money.of(10, euro)),
      new Item("Soda water", Money.of(1.25, euro))
  );
