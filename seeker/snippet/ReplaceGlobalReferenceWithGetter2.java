//date: 2024-09-05T16:46:15Z
//url: https://api.github.com/gists/278e3df2a2f14aff40d3162dd3cef94c
//owner: https://api.github.com/users/trikitrok

// After applying Subclass & Override Method

class RegisterSale {
    private List<Item> items;
    // more code...

    public void addItem(Barcode code) {
        // using the Singleton!! x(
        Item newItem = getInventory().getItemForBarCode(code);
        items.add(newItem);
    }

    protected Inventory getInventory() {
        return Inventory.GetInstance();
    }

    // more code...
}

/////////////////////////////

// In some test

public class RegisterSaleTest {

    @Test
    public void Adds_An_Item() {
        Barcode code = new Barcode();
        // some more setup code
        //...
        // we subclass & override the getter and return a test double of Inventory
        Inventory inventory = mock(Inventory.class);

        when(inventory.getItemForBarCode(code)).thenReturn(AnItem().withBarcode(code).build());
        RegisterSaleForTesting registerSale = new RegisterSaleForTesting(inventory);

        // rest of the test...
    }
    
    public class RegisterSaleForTesting extends RegisterSale {
        private final Inventory inventory;

        public RegisterSaleForTesting(Inventory inventory) {
            this.inventory = inventory;
        }

        // overriden to separate from the singleton
        @Override
        protected Inventory getInventory() {
            return inventory;
        }
    }
}