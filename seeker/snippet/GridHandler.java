//date: 2023-04-12T16:54:31Z
//url: https://api.github.com/gists/de2a864eafe60db3f9bc7fc602ac0829
//owner: https://api.github.com/users/sindrihardar

import javafx.scene.layout.AnchorPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;

public class GridHandler extends GridBase {

    Color backgroundColor1 = Color.WHITE;
    Color backgroundColor2 = Color.color(0.82,0.82,0.82);

    public GridHandler(double planeWidth, double planeHeight, int gridSize, AnchorPane anchorPane) {
        super(planeWidth, planeHeight, gridSize, anchorPane);
    }

    public void updateGrid() {
        for(int i = 0; i < getTileAmount(); i++){
            int x = (i % getTilesAcross());
            int y = (i / getTilesAcross());

            Rectangle rectangle = new Rectangle(x * getGridSize(),y * getGridSize(),getGridSize(),getGridSize());

            if((x + y) % 2 == 0){
                rectangle.setFill(backgroundColor1);
            } else {
                rectangle.setFill(backgroundColor2);
            }
            getAnchorPane().getChildren().add(rectangle);
        }
    }
}
