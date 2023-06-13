//date: 2023-06-13T16:57:54Z
//url: https://api.github.com/gists/77bba48d4f9e91b6f18ee97e09ca2933
//owner: https://api.github.com/users/Iog1123

package com.example.myapplication;

import android.content.Context;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
   Button bSumar;
   Button bRestar;
   Button bMulti;
   Button bDivi;
   EditText num1;
   EditText num2;
   TextView total;
   
    public int sumar(int a, int b){
        int suma=0;
        suma=a+b;
        return suma;
    }
    public int restar(int a,int b){
        int resta=0;
        resta=a-b;
        return resta;
    }
    public int multi(int a,int b){
        int res=0;
        res=a*b;
        return res;
    }
    public String divid(int a,int b){
        String res="0";
        if(b!=0){
            res=""+a/b;
            return res;}else {
            return "Error";
        }

    }
   @Override
      protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //objetos para cada uno de los del xml
        total=(TextView) findViewById(R.id.etiResultado);
        num1=(EditText) findViewById(R.id.primerNum);
        num2=(EditText) findViewById(R.id.segundoNum);
        bSumar=(Button) findViewById(R.id.botonSumar);
        bRestar=(Button) findViewById(R.id.botonRestar);
        bMulti=(Button) findViewById(R.id.botonMulti);
        bDivi=(Button) findViewById(R.id.botonDivi);

        //variables para el mensaje toast a utilizar
        Context context =getApplicationContext();
        CharSequence text="Falta un número";
        int duracion = Toast.LENGTH_SHORT;
        Toast toast = Toast.makeText(context,text,duracion);

        //listeners para los botones
        bSumar.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //comprobacion
                if(num1.getText().toString().equals("") || num2.getText().toString().equals("")){
                    //mensaje al ser vacio algun numero
                    toast.show();
                }else{
                    //edicion del resultado segun los valores dados
                    total.setText(""+sumar(Integer.parseInt(num1.getText().toString()),Integer.parseInt(num2.getText().toString())));
                }

            }
        });

        bRestar.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //comprobacion
                if(num1.getText().toString().equals("") || num2.getText().toString().equals("")){
                    //mensaje al ser vacio algun numero
                    toast.show();
                }else{
                    //edicion del resultado segun los valores dados
                    total.setText(""+restar(Integer.parseInt(num1.getText().toString()),Integer.parseInt(num2.getText().toString())));
                }
            }
        });

        bMulti.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //comprobacion
                if(num1.getText().toString().equals("") || num2.getText().toString().equals("")){
                    //mensaje al ser vacio algun numero
                    toast.show();
                }else{
                    //edicion del resultado segun los valores dados
                    total.setText(""+multi(Integer.parseInt(num1.getText().toString()),Integer.parseInt(num2.getText().toString())));
                }
            }
        });

        bDivi.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //comprobacion
                if(num1.getText().toString().equals("") || num2.getText().toString().equals("")){
                    //mensaje al ser vacio algun numero
                    toast.show();
                }else{
                    //edicion del resultado segun los valores dados
                    total.setText(""+divid(Integer.parseInt(num1.getText().toString()),Integer.parseInt(num2.getText().toString())));
                }
            }
        });
    }
}