//date: 2024-01-30T17:09:26Z
//url: https://api.github.com/gists/d5b8371bba291cf67613f49a43b7c8ae
//owner: https://api.github.com/users/OrangoMango

import javafx.application.Application;
import javafx.application.Platform;
import javafx.stage.Stage;
import javafx.scene.robot.Robot;
import javafx.animation.AnimationTimer;
import javafx.scene.input.MouseButton;

public class AutoClicker extends Application {
    @Override
    public void start(Stage stage) {
        Robot robot = new Robot();

        AnimationTimer timer = new AnimationTimer() {
            @Override
            public void handle(long time) {
                robot.mouseClick(MouseButton.PRIMARY);
            }
        };
        timer.start();

        new Thread(() - > {
            try {
                Thread.sleep(6000);
                timer.stop();
            } catch (InterruptedException ex) {
                ex.printStackTrace();
            }
        }).start();
    }

    public static void main(String[] args) {
        launch(args);
    }
}