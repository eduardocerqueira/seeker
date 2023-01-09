//date: 2023-01-09T16:51:37Z
//url: https://api.github.com/gists/c3bb170dc2bc0cfbe41a7441ba0f53a9
//owner: https://api.github.com/users/PotatoGolden76

import javafx.application.Application;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.stage.Stage;

import java.io.IOException;
import java.util.Objects;

public class GUI extends Application {
    Parent root, run;
    Scene scene;
    @FXML
    private Label runLbl;

    public static void main(String[] args) {
        launch(args);
    }

    public void run() {
        scene.setRoot(run);
        System.out.println("run");
    }

    public void bye() {
        System.out.println("Bye");
    }

    @Override
    public void start(Stage stage) throws IOException {
        this.root = FXMLLoader.load(GUI.class.getResource("GUI.fxml"));
//        this.run = FXMLLoader.load(GUI.class.getResource("run.fxml"));

        scene = new Scene(root);

        stage.setTitle("FXML Welcome");
        stage.setScene(scene);
        stage.show();
    }
}
