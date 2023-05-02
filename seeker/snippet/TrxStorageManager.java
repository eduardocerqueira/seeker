//date: 2023-05-02T16:56:42Z
//url: https://api.github.com/gists/4eb64d91f9ae3ce749b2e019119b99fe
//owner: https://api.github.com/users/Monoradioactivo

public class TrxStorageManager {

  void fillTransactionStorage() {
    List<Transaction> transactions = new ArrayList<>();
    Path filePath = Paths.get(Main.class.getResource("MOCK_DATA.csv").getPath());

    try (Scanner sc = new Scanner(filePath)) {
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
      throw new RuntimeException(e);
    } catch (IOException e) {
      e.printStackTrace();
    } catch (NumberFormatException | IllegalStateException e) {
      System.err.println("Invalid data format in the CSV file.");
    }
    TransactionStorage.getInstance().setTransactions(transactions);
  }
}
