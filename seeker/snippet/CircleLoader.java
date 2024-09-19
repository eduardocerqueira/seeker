//date: 2024-09-19T16:54:01Z
//url: https://api.github.com/gists/ea0c3425bcbdacac7ee1fdda8ee3b33c
//owner: https://api.github.com/users/gleidsonmt

package io.github.gleidsonmt.tutorial_circle;

import javafx.animation.Interpolator;
import javafx.animation.RotateTransition;
import javafx.animation.Timeline;
import javafx.beans.property.StringProperty;
import javafx.geometry.Point3D;
import javafx.geometry.Pos;
import javafx.scene.control.Label;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.shape.Circle;
import javafx.util.Duration;

/**
 * @author Gleidson Neves da Silveira | gleidisonmt@gmail.com
 * Create on  17/09/2024
 */
public abstract class CircleLoader extends VBox implements SuspenseLoader {

    protected final Label title;
    protected final Label legend;

    public CircleLoader() {
        this("Loading...", "Loading tasks...");
    }

    public CircleLoader(String _title, String _legend) {
        title = new Label(_title);
        legend = new Label(_legend);

        title.getStyleClass().add("title");
        legend.getStyleClass().add("legend");

        // setting a margin between sections
        this.setSpacing(10);
        this.setAlignment(Pos.CENTER);
        this.getStyleClass().add(0, "circle-loader");
        //
        StackPane circleContainer = createCircleContainer();
        circleContainer.getStyleClass().add("container-circle");
        circleContainer.getChildren().add(title);
        this.getChildren().setAll(circleContainer, legend);
    }

    protected abstract StackPane createCircleContainer(); //

    protected void rotate(Circle circle, int angle, Point3D point) {
        RotateTransition rotate = new RotateTransition(Duration.seconds(5), circle);

        rotate.setAxis(point);
        rotate.setAutoReverse(true);

        rotate.setByAngle(angle);
        rotate.setInterpolator(Interpolator.LINEAR);
        rotate.setCycleCount(Timeline.INDEFINITE);
        rotate.play();

    }

    protected void rotate(Circle circle, int angle, int duration) {
        RotateTransition transition = new RotateTransition(Duration.seconds(duration), circle);

        transition.setByAngle(angle);
//        transition.setInterpolator(Interpolator.TANGENT(Duration.millis(2000), 80));
//        transition.setInterpolator(Interpolator.LINEAR);
        transition.setInterpolator(Interpolator.SPLINE(1, 0.8, 0.6, 0.4));
        transition.setCycleCount(Timeline.INDEFINITE);
        transition.play();

    }

    @Override
    public StringProperty titleProperty() {
        return title.textProperty();
    }

    @Override
    public StringProperty legendProperty() {
        return legend.textProperty();
    }

    @Override
    public void setTitle(String _title) {
        title.setText(_title);
    }

    @Override
    public void setLegend(String _legend) {
        legend.setText(_legend);
    }
}
