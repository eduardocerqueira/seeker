//date: 2022-08-31T17:17:36Z
//url: https://api.github.com/gists/fa1d339f22ca4ceb1a6e35c6dff1d969
//owner: https://api.github.com/users/CullenSUN

class Client {
    public static void main(String[] args) {
        Printer printer1 = new NewPrinter();
        printer1.print("Abcdefg", PrintingMode.COLOR);
        printer1.print("Hijklmn", PrintingMode.BLACK_AND_WHITE);

        Printer printer2 = new OldPrinterAdapter(new OldPrinter());
        printer2.print("Abcdefg", PrintingMode.COLOR);
        printer2.print("Hijklmn", PrintingMode.BLACK_AND_WHITE);
    }
}