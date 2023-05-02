//date: 2023-05-02T16:56:42Z
//url: https://api.github.com/gists/4eb64d91f9ae3ce749b2e019119b99fe
//owner: https://api.github.com/users/Monoradioactivo

public class Main {

  public static void main(String[] args) {
    TransactionStorageManager transactionStorageManager = new TransactionStorageManager();
    transactionStorageManager.fillTransactionStorage();
    GUI gui = new GUI(
        new TransactionRepository(TransactionStorage.getInstance().getTransactions()));
    gui.start();
  }

}
