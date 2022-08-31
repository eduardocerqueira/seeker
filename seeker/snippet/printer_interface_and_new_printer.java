//date: 2022-08-31T17:15:53Z
//url: https://api.github.com/gists/95a3450d2624d1fd5f251b9adea3f71e
//owner: https://api.github.com/users/CullenSUN

enum PrintingMode {
    BLACK_AND_WHITE,
    COLOR
}

interface Printer {
    public void print(String text, PrintingMode mode);
}

class NewPrinter implements Printer {
    public void print(String text, PrintingMode mode) {
        switch (mode) {
            case BLACK_AND_WHITE:
            System.out.println("New printer printing in black and white mode happily.");
            break;
            case COLOR:
            System.out.println("New printer printing in color mode happily.");
            break;
        }
        System.out.printf("printing: %s\n", text);
    }
}
