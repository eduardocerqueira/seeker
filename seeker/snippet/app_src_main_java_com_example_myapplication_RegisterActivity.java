//date: 2021-10-20T17:02:53Z
//url: https://api.github.com/gists/9c8bd28bd577eee6a0001baac47110b7
//owner: https://api.github.com/users/AleksandrMandrov

package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.widget.EditText;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class RegisterActivity extends AppCompatActivity {

    Button btnMoveToLogIn, btnRegistration;
    EditText login, email, password, checkPassword;
    TextView errorMesage;
    ImageView imageWelcom;

    DatabaseHelper databaseHelper;
    DatabaseHelper sqlHelper;
    SQLiteDatabase db;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);

        btnMoveToLogIn = (Button) findViewById(R.id.btnMoveToLogIn);
        btnRegistration = (Button) findViewById(R.id.btnRegistration);

        login = (EditText) findViewById(R.id.login);
        email = (EditText) findViewById(R.id.email);
        password = (EditText) findViewById(R.id.password);
        checkPassword = (EditText) findViewById(R.id.checkPassword);

        errorMesage = (TextView) findViewById(R.id.errorMesage);
        imageWelcom = (ImageView) findViewById(R.id.imageWelcom);

        databaseHelper = new DatabaseHelper(getApplicationContext());
        // создаем базу данных
        databaseHelper.create_db();

        sqlHelper = new DatabaseHelper(this);
        db = sqlHelper.open();

    }

    public void moveToLogIn(View view) {
        Intent intent = new Intent(this, MainActivity.class);
        startActivity(intent);
    }

    public void moveToLastSolutions(View view) {

        String loginValue = login.getText().toString();
        String emailValue = email.getText().toString();
        String passwordValue = password.getText().toString();
        String passwordCheak = checkPassword.getText().toString();

        String error = "";
        boolean errorFlag = true;

        if (loginValue.equals("")) {
            error += "\n- вы не ввели логин";
            errorFlag = false;
        }
        if (emailValue.equals("")) {
            error += "\n- вы не ввели почту";
            errorFlag = false;
        }
        if (passwordValue.equals("")) {
            error += "\n- вы не ввели пароль";
            errorFlag = false;
        }
        if (passwordCheak.equals("")) {
            error += "\n- вы не заполнили поле 'Подтверждение пароля'";
            errorFlag = false;
        }

        if (errorFlag) {

            if (loginValue.length() < 5) {
                error += "\n- ваш 'логин' должен содержать не менее 5 символов";
                errorFlag = false;
            }

            final String LOGIN_PATTERN = "^[a-zA-Z0-9]+$";
            Pattern pattern = Pattern.compile(LOGIN_PATTERN);
            Matcher matcher = pattern.matcher(loginValue);

            if (!matcher.matches()) {
                error += "\n- в поле 'Логин' можно использовать только латинские символы и цифры";
                errorFlag = false;
            }

            final String EMAIL_PATTERN =
                    "^[_A-Za-z0-9-\\+]+(\\.[_A-Za-z0-9-]+)*@"
                            + "[A-Za-z0-9-]+(\\.[A-Za-z0-9]+)*(\\.[A-Za-z]{2,})$";

            pattern = Pattern.compile(EMAIL_PATTERN);
            matcher = pattern.matcher(emailValue);
            if (!matcher.matches()) {
                error += "\n- вы ввели недопустимые символы для поля 'почта'";
                errorFlag = false;
            }

            if (errorFlag) {

                Cursor cursor = db.query(DatabaseHelper.TABLE, null, null, null, null, null, null);

                if (cursor.moveToFirst()) {
                    int idName = cursor.getColumnIndex(DatabaseHelper.COLUMN_NAME);
                    int idemail = cursor.getColumnIndex(DatabaseHelper.COLUMN_EMAIL);

                    do {
                        if (cursor.getString(idName).equals(loginValue)) {
                            error += "\n- пользователь с таким логином уже существует";
                            errorFlag = false;
                            break;
                        }
                    } while (cursor.moveToNext());
                    cursor.moveToFirst();
                    do {
                        if (cursor.getString(idemail).equals(emailValue)) {
                            error += "\n- пользователь с такой почтой уже существует";
                            errorFlag = false;
                            break;
                        }
                    } while (cursor.moveToNext());
                }

                cursor.close();

                if (errorFlag) {

                    if (passwordValue.length() < 8) {
                        error += "\n- ваш 'пароль' должен содержать не менее 8 символов";
                        errorFlag = false;
                    }

                    final String PASSWORDCHEAK_PATTERN = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d).+$";
                    pattern = Pattern.compile(PASSWORDCHEAK_PATTERN);
                    matcher = pattern.matcher(passwordValue);

                    if (!matcher.matches()) {
                        error += "\n- ваш 'пароль' не соответвует требованиям";
                        errorFlag = false;
                    }

                    if (!passwordValue.equals(passwordCheak)) {
                        error += "\n- ваши 'пароли' не совпадают";
                        errorFlag = false;
                    }

                    if (errorFlag) {

                        ContentValues cv = new ContentValues();
                        cv.put(DatabaseHelper.COLUMN_NAME, loginValue);
                        cv.put(DatabaseHelper.COLUMN_EMAIL, emailValue);
                        cv.put(DatabaseHelper.COLUMN_PASSWORD, passwordValue);
                        db.insert(DatabaseHelper.TABLE, null, cv);
                        db.close();

                        Intent intent = new Intent(this, LastSolutions.class);
                        startActivity(intent);
                    }
                }
            }
        }

        if (!errorFlag) {
            error = "При регистрации возникли ошибки:" + error;
            errorMesage.setText(error);
            errorMesage.setPaddingRelative(20, 20,20,20);
            imageWelcom.setPadding(0,50,0,6);
            imageWelcom.setImageResource(R.drawable.graymag);
        }


    }

}