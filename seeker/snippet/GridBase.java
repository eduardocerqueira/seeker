//date: 2023-04-12T16:54:31Z
//url: https://api.github.com/gists/de2a864eafe60db3f9bc7fc602ac0829
//owner: https://api.github.com/users/sindrihardar

import javafx.scene.layout.AnchorPane;

public abstract class GridBase {

    private double planeWidth;
    private double planeHeight;
    private int tilesAcross;
    private int tileAmount;
    private int gridSize;
    private int tilesDown;
    private AnchorPane anchorPane;

    public GridBase(double planeWidth, double planeHeight, int gridSize, AnchorPane anchorPane) {
        this.planeWidth = planeWidth;
        this.planeHeight = planeHeight;
        this.gridSize = gridSize;
        this.anchorPane = anchorPane;

        tilesAcross = (int) (planeWidth / gridSize);
        tileAmount = (int) ((planeWidth /gridSize) * (planeHeight /gridSize));
        tilesDown = tileAmount/tilesAcross;
    }

    public double getPlaneWidth() {
        return planeWidth;
    }

    public double getPlaneHeight() {
        return planeHeight;
    }

    public int getTilesAcross() {
        return tilesAcross;
    }

    public int getTileAmount() {
        return tileAmount;
    }

    public int getGridSize() {
        return gridSize;
    }

    public int getTilesDown() {
        return tilesDown;
    }

    public AnchorPane getAnchorPane() {
        return anchorPane;
    }
}
