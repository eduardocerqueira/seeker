//date: 2023-11-16T16:51:38Z
//url: https://api.github.com/gists/671981cef0f33a100476567a2a768954
//owner: https://api.github.com/users/cdrap27

package com.example.nintendo64reccomendations;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;

public class HelloApplication extends Application {
    @Override
    public void start(Stage stage) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(HelloApplication.class.getResource("hello-view.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 320, 240);
        stage.setTitle("Hello!");
        stage.setScene(scene);
        stage.show();
        //test
    }

    public static void main(String[] args) {
        launch();
    }
}