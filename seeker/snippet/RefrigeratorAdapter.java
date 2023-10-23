//date: 2023-10-23T17:00:22Z
//url: https://api.github.com/gists/0840dc3dc4518567b83967e08f95a3ae
//owner: https://api.github.com/users/AaronRubinos

package AdapterPattern;

public class RefrigeratorAdapter implements PowerOutlet {
    private Refrigerator ref;

    public RefrigeratorAdapter(Refrigerator ref)   {
        this.ref = ref;
    }

    @Override
    public void plugIn() {
        ref.startCooling();
    }
}
