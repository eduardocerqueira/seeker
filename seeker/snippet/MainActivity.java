//date: 2023-06-02T17:05:21Z
//url: https://api.github.com/gists/3fca38abd0716d12fe344d2c66a74dd2
//owner: https://api.github.com/users/i2gan

public class MainActivity extends AppCompatActivity {

    Button button;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        button = findViewById(R.id.button);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                openActitivy2();
            }
        });
    }

    public void openActitivy2() {
        Intent intent = new Intent(this, Activity2.class);
        startActivity(intent);
    }
}