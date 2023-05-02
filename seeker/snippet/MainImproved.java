//date: 2023-05-02T16:56:42Z
//url: https://api.github.com/gists/4eb64d91f9ae3ce749b2e019119b99fe
//owner: https://api.github.com/users/Monoradioactivo

public class Main {

  private static final String DELIMITER = ",|\n";
  private static final String DATA_FILE = "MOCK_DATA.csv";

  public static void main(String[] args) {
    List<Transaction> transactions = loadTransactions();
    TransactionStorage storage = TransactionStorage.getInstance();
    storage.setTransactions(transactions);

    TransactionRepository repository = new TransactionRepository(storage);
    GUI gui = new GUI(repository);
    gui.start();
  }

  private static List<Transaction> loadTransactions() {
    List<Transaction> transactions = new ArrayList<>();

    try (Scanner sc = new Scanner(new File(Main.class.getResource(DATA_FILE).getPath()))) {
      sc.useDelimiter(DELIMITER);
      parseTransactions(sc, transactions);
    } catch (FileNotFoundException e) {
      throw new TransactionLoadingException("Error loading transactions from the data file", e);
    }

    return transactions;
  }

  private static void parseTransactions(Scanner sc, List<Transaction> transactions) {
    String description;
    double amount;
    int id;
    LocalDate date;

    while (sc.hasNext()) {
      description = sc.next();
      amount = sc.nextDouble();
      id = sc.nextInt();
      date = LocalDate.parse(sc.next());
      transactions.add(new Transaction(description, amount, id, date));
    }
  }
}