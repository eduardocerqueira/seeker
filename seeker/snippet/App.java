//date: 2024-09-19T16:54:01Z
//url: https://api.github.com/gists/ea0c3425bcbdacac7ee1fdda8ee3b33c
//owner: https://api.github.com/users/gleidsonmt

import javafx.application.Application;
import javafx.concurrent.Task;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.layout.FlowPane;
import javafx.stage.Stage;

/**
 * @author Gleidson Neves da Silveira | gleidisonmt@gmail.com
 * Create on  18/09/2024
 */
public class App extends Application {
    @Override
    public void start(Stage stage)  {

        FlowPane root = new FlowPane();
        root.setHgap(20);
        root.setAlignment(Pos.CENTER);

        SuspenseCircle suspenseCircle = new SuspenseCircle();
        TechCircle techCircle = new TechCircle();
        Suspense3DCircle suspense3DCircle = new Suspense3DCircle();

        root.getChildren().addAll(suspenseCircle, techCircle, suspense3DCircle);

        stage.setTitle("Suspense Circle!");
        stage.setScene(new Scene(root, 800, 600));
        stage.getScene().getStylesheets().add(getClass().getResource("style.css").toExternalForm());


        Task task = new Task() {
            @Override
            protected Object call() throws Exception {
                Thread.sleep(2000);
                updateTitle("Initializing...");
                updateMessage(("Welcome!"));
                return null;
            }
        };
        Thread thread = new Thread(task);
        thread.start();
        suspenseCircle.titleProperty().bind(task.titleProperty());
        suspenseCircle.legendProperty().bind(task.messageProperty());

        stage.show();
    }
  
    public static void main(String[] args) {
      launch();
    }
}