//date: 2021-10-27T16:55:09Z
//url: https://api.github.com/gists/fdc8f4190a8d0cb01c3fe132ece9c5d3
//owner: https://api.github.com/users/wojciechlibor

package com.example.twjkoszyk;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private ArrayList<Product> products = new ArrayList<>();
    private RecyclerView recyclerView;
    private ShoppingAdapter shoppingAdapter;
    private SharedPreferences sharedPreferences;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        sharedPreferences = getPreferences(MODE_PRIVATE);

        products = loadProducts();
        //products.add(new Product("mleko"));
        //products.add(new Product("jajka"));



        recyclerView = (RecyclerView) findViewById(R.id.recyclerView);
        shoppingAdapter = new ShoppingAdapter(this, products);
        recyclerView.setAdapter(shoppingAdapter);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        //przycisk dodaj

        Button btn_add = (Button) findViewById(R.id.btn_add);
        btn_add.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                EditText editText = (EditText) findViewById(R.id.editText);
                String productName = editText.getText().toString();
                productName = productName.trim();
                if (productName.length()>0){
                    Product prod = new Product(productName);
                    shoppingAdapter.addProduct(prod);
                    editText.setText("");  //czyszczenie pola
                }
            }
        });

        //przycisk usun

        Button btn_del = (Button) findViewById(R.id.btn_del);
        btn_del.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                shoppingAdapter.delProduct();

            }
        });


    }

    private ArrayList<Product> loadProducts() {
        String fromGson = sharedPreferences.getString("dane","");
        //deserializacja danych
        Gson gson = new Gson();
        Type typ = new TypeToken<List<Product>>(){}.getType();

        ArrayList<Product> products = gson.fromJson(fromGson, typ);

        if (products == null){
            products = new ArrayList<>();
        }

        return products;
    }

    @Override
    protected void onPause() {
        super.onPause();
        saveProducts();
    }

    private void saveProducts() {
        SharedPreferences.Editor editor = sharedPreferences.edit();
        Gson gson = new Gson();
        String productAsJson = gson.toJson(products);
        editor.putString("dane", productAsJson);
        editor.apply();
    }
}