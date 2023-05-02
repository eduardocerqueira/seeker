//date: 2023-05-02T16:56:42Z
//url: https://api.github.com/gists/4eb64d91f9ae3ce749b2e019119b99fe
//owner: https://api.github.com/users/Monoradioactivo

public class TransactionFillException extends Exception {
    public TransactionFillException(String message, Throwable cause) {
        super(message, cause);
    }
}

//Call it in your fillStorage() method
private static void fillStorage() {
    List<Transaction> transactions = new ArrayList<>();
    try {
      Scanner sc = new Scanner(new File(Main.class.getResource("MOCK_DATA.csv").getPath()));
      String description;
      double amount;
      int id;
      LocalDate date;
      sc.useDelimiter(",|\n");
      while (sc.hasNext()) {
        description = sc.next();
        amount = sc.nextDouble();
        id = sc.nextInt();
        date = LocalDate.parse(sc.next());
        transactions.add(new Transaction(description, amount, id, date));
      }
    } catch (FileNotFoundException e) {
      throw new TransactionLoadingException("Error loading transactions from the data file", e);
    }
    TransactionStorage.getInstance().setTransactions(transactions);
  }