//date: 2023-05-31T17:00:46Z
//url: https://api.github.com/gists/8c482ce25e8253deb117be4c2e233010
//owner: https://api.github.com/users/boombang

package com.example.chess;
import java.util.Hashtable;
import java.util.Random;
import javafx.application.Application;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.stage.Stage;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class Gribanov_Kon extends Application {
    private List<Cell> reachableCells;

    private static int BOARD_SIZE = 5;
    private static final int CELL_SIZE = 75;

    private int[][] grid;

    private static boolean[][] visited;

    private Button[][] cellButtons;

    private int selectedRowX = -1;
    private int selectedColY = -1;
    private int selectedMoves = 2;

    // Ход конём
    private final static int[][] horseMovement = {
            {-2, -1}, // Вниз-влево
            {-1, -2}, // Вниз-влево
            {1, -2}, // Вверх-влево
            {2, -1}, // Вверх-влево
            {2, 1}, // Вверх-вправо
            {1, 2}, // Вверх-вправо
            {-1, 2}, // Вниз-вправо
            {-2, 1} // Вниз-вправо
    };

    private String colors[] = {
            "DCDCDC",
            "D3D3D3",
            "C0C0C0",
            "A9A9A9",
            "808080",
            "696969",
            "778899",
            "B0C4DE",
            "708090",
    };

    GridPane gridWallsPane = new GridPane();

    static TextArea area = new TextArea();

    static String str = "";

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("КОНЬ");
        reachableCells = new ArrayList<>();
        HBox panes = new HBox(15);

        area.setPrefSize(100, 300);

        panes.setAlignment(Pos.CENTER);
        panes.getChildren().add(gridWallsPane);
        gridWallsPane.setAlignment(Pos.BASELINE_CENTER);

        VBox controlBox = new VBox(15);
        controlBox.setAlignment(Pos.CENTER);

        // Очистка панели
        gridWallsPane.getChildren().clear();

        // Инициализация поля
        BOARD_SIZE = 8;
        grid = new int[BOARD_SIZE][BOARD_SIZE];
        cellButtons = new Button[BOARD_SIZE][BOARD_SIZE];

        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                final int ii = i, jj = j;
                Button wallCellButton = new Button();
                wallCellButton.setPrefSize(CELL_SIZE, CELL_SIZE);
                wallCellButton.setOnAction(z -> {
                });
                cellButtons[i][j] = wallCellButton;
                gridWallsPane.add(wallCellButton, j, i);
            }
        }

        fieldColoring();

        // Установка коня на доске
        Button setWalls = new Button();
        setWalls.setPrefSize(170.0,40.0);
        setWalls.setText("Поставить коня");

        // Обработка события
        setWalls.setOnAction(e -> {
            gridWallsPane.getChildren().clear();

            cellButtons = new Button[BOARD_SIZE][BOARD_SIZE];

            for (int i = 0; i < BOARD_SIZE; i++) {
                for (int j = 0; j < BOARD_SIZE; j++) {
                    final int ii = i, jj = j;
                    Button wallCellButton = new Button();
                    wallCellButton.setPrefSize(CELL_SIZE, CELL_SIZE);
                    wallCellButton.setOnAction(z -> {
                        handleCellClick(ii, jj);
                    });

                    cellButtons[i][j] = wallCellButton;
                    gridWallsPane.add(wallCellButton, j, i);
                }
            }
            fieldColoring();
        });

        TextField stepField = new TextField();
        stepField.setPromptText("Кол-во шагов");

        Button findButton = new Button("Найти клетки");
        findButton.setPrefSize(170.0,40.0);
        findButton.setOnAction(e -> {
            selectedMoves = Integer.parseInt(stepField.getText());
            find();
            area.setText(str);
        });

        Button clearButton = new Button();
        clearButton.setPrefSize(170.0,38.0);
        clearButton.setText("Очистка");
        clearButton.setOnAction(e -> {
            // Очистка полей
            stepField.setText("");

            // Очистка панели
            gridWallsPane.getChildren().clear();
            // Очистка текстовой арены
            area.clear();

            // Инициализация поля
            BOARD_SIZE = 8;
            grid = new int[BOARD_SIZE][BOARD_SIZE];
            cellButtons = new Button[BOARD_SIZE][BOARD_SIZE];

            for (int i = 0; i < BOARD_SIZE; i++) {
                for (int j = 0; j < BOARD_SIZE; j++) {
                    final int ii = i, jj = j;
                    Button wallCellButton = new Button();
                    wallCellButton.setPrefSize(CELL_SIZE, CELL_SIZE);
                    wallCellButton.setOnAction(z -> {
                    });
                    cellButtons[i][j] = wallCellButton;
                    gridWallsPane.add(wallCellButton, j, i);
                }
            }
            selectedRowX = -1;
            selectedColY = -1;
            fieldColoring();
        });

        controlBox.getChildren().add(setWalls);
        controlBox.getChildren().add(stepField);
        controlBox.getChildren().add(findButton);
        controlBox.getChildren().add(clearButton);

        controlBox.getChildren().add(area);

        Scene scene = new Scene(new javafx.scene.layout.HBox(30, panes, controlBox), 790, 500);
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    Hashtable<Integer, List<CellsPair>> hashtable1 = new Hashtable<>();
    List<MarkedCell> markedCells = new ArrayList<>();

    // обработать щелчок по ячейке
    private void handleCellClick(int x, int y) {

        if (!markedCells.isEmpty() && selectedRowX != -1 && selectedColY != -1 && selectedMoves > 0) {
            MarkedCell destinationCell = null;
            for (MarkedCell markedCell: markedCells) {
                if (markedCell.cell.x == x && markedCell.cell.y == y) {
                    destinationCell = markedCell;
                    break;
                }
            }


            if (destinationCell != null) {
                List<Cell> path = new ArrayList<>();
                path.add(destinationCell.cell);

                Cell tempCell = destinationCell.cell;
                for (int i = destinationCell.step; i >= 1; i--) {
                    List<CellsPair> entries = hashtable1.get(i);
                    for (CellsPair pair: entries) {
                        if (pair.toCell.x == tempCell.x && pair.toCell.y == tempCell.y) {
                            path.add(pair.fromCell);
                            tempCell = pair.fromCell;
                            break;
                        }
                    }
                }

                path.add(new Cell(selectedRowX, selectedColY));

                Cell arr[] = new Cell[path.size()];

                int arrIndex = 0;
                for (Cell pathCell: path) {
                    arr[arrIndex++] = pathCell;
                }

                for (int i = 0; i < arr.length - 1; i++) {
                    Cell cell1 = arr[i];
                    Cell cell2 = arr[i + 1];

                    int cell2X = cell2.x;
                    while (cell2X != cell1.x) {
                        if (cell2X < cell1.x) {
                            cell2X++;
                        } else {
                            cell2X--;
                        }

                        cellButtons[cell2X][cell2.y].setStyle("-fx-background-color: #e6b6be;");

                    }

                    int cell2Y = cell2.y;
                    while (cell2Y != cell1.y) {
                        if (cell2Y < cell1.y) {
                            cell2Y++;
                        } else {
                            cell2Y--;
                        }

                        cellButtons[cell2X][cell2Y].setStyle("-fx-background-color: #e6b6be;");

                    }
                }

                return;
            }
        }

        if (selectedRowX == x && selectedColY == y) {
            if (x % 2 == 0){
                if (y % 2 == 0)
                {
                    // Отмените выделение ячейки при повторном нажатии
                    cellButtons[x][y].setText("");
                    cellButtons[x][y].setStyle("-fx-background-color: #A0522D;");
                }
                else {
                    // Отмените выделение ячейки при повторном нажатии
                    cellButtons[x][y].setText("");
                    cellButtons[x][y].setStyle("-fx-background-color: #FFEFD5;");
                }
            }else {
                if (y % 2 != 0)
                {
                    // Отмените выделение ячейки при повторном нажатии
                    cellButtons[x][y].setText("");
                    cellButtons[x][y].setStyle("-fx-background-color: #A0522D;");
                }
                else {
                    // Отмените выделение ячейки при повторном нажатии
                    cellButtons[x][y].setText("");
                    cellButtons[x][y].setStyle("-fx-background-color: #FFEFD5;");
                }
            }
            selectedRowX = -1;
            selectedColY = -1;
        } else {
            if (selectedRowX != -1 && selectedColY != -1) {
                cellButtons[selectedRowX][selectedColY].setStyle("");
            }
            cellButtons[x][y].setStyle("-fx-background-color: #FFD700;");
            cellButtons[x][y].setFont(new Font("Roboto", 20));
            cellButtons[x][y].setText("♞");

            selectedRowX = x;
            selectedColY = y;
        }
    }

    // Главный поиск
    private void find() {
        reachableCells.clear();
        if (selectedRowX != -1 && selectedColY != -1 && selectedMoves > 0) {
            // Инициализация
            visited = new boolean[BOARD_SIZE][BOARD_SIZE];
            for (int i = 0; i < BOARD_SIZE; i++) {
                for (int j = 0; j < BOARD_SIZE; j++) {
                    if( i == selectedRowX && j == selectedColY ) {
                        cellButtons[i][j].setStyle("-fx-background-color: #FFD700;");
                        cellButtons[i][j].setFont(new Font("Roboto", 20));
                        cellButtons[i][j].setText("♞");
                    } else if (i % 2 == 0){
                        if (j % 2 == 0)
                        {
                            cellButtons[i][j].setText("");
                            cellButtons[i][j].setStyle("-fx-background-color: #A0522D;");
                            cellButtons[i][j].setFont(new Font("Roboto", 20));
                        } else {
                            cellButtons[i][j].setText("");
                            cellButtons[i][j].setStyle("-fx-background-color: #FFEFD5;");
                            cellButtons[i][j].setFont(new Font("Roboto", 20));
                        }
                    } else {
                        if (j % 2 != 0) {
                            cellButtons[i][j].setText("");
                            cellButtons[i][j].setStyle("-fx-background-color: #A0522D;");
                            cellButtons[i][j].setFont(new Font("Roboto", 20));
                        } else {
                            cellButtons[i][j].setText("");
                            cellButtons[i][j].setStyle("-fx-background-color: #FFEFD5;");
                            cellButtons[i][j].setFont(new Font("Roboto", 20));
                        }
                    }

                }
            }

            // Закраска
            str = "";

            hashtable1.clear();
            markedCells.clear();

            for (int i = 1; i <= selectedMoves; i++) {
                System.out.println(i + "-й Ход");
                str += i + "-й Ход\n";

                List<Cell> reachableCells = searchReachableCells(grid, selectedRowX, selectedColY, i);

                for (Cell cell : reachableCells) {
                    String currentStyle = cellButtons[cell.x][cell.y].getStyle();
                    String currentColor = currentStyle.substring(currentStyle.lastIndexOf("#") + 1);

                    boolean loop = false;
                    for (int j = 0; j < i; j++){
                        loop = currentColor.equals(colors[j] + ";");
                        if (loop) break;
                    }

                    if(cell.x == selectedRowX && cell.y == selectedColY) {
                        cellButtons[cell.x][cell.y].setStyle("-fx-background-color: #FFD700;");
                        cellButtons[cell.x][cell.y].setFont(new Font("Roboto", 20));
                        cellButtons[cell.x][cell.y].setText("♞");
                    } else if (loop) {
                        continue;
                    } else {
                        cellButtons[cell.x][cell.y].setText(i + "-й ход" + " " + "(" + cell.x + " , " + cell.y + ")");
                        cellButtons[cell.x][cell.y].setStyle("-fx-background-color: #" + colors[i] + ";");
                        cellButtons[cell.x][cell.y].setFont(new Font(10));

                        markedCells.add(new MarkedCell(cell, i));
                    }
                }
            }
        }
    }

    // Раскраска сетки
    private void fieldColoring() {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (i % 2 == 0){
                    if (j % 2 == 0)
                    {
                        cellButtons[i][j].setStyle("-fx-background-color: #A0522D;");
                    }
                    else {
                        cellButtons[i][j].setStyle("-fx-background-color: #FFEFD5;");
                    }
                }else {
                    if (j % 2 != 0)
                        cellButtons[i][j].setStyle("-fx-background-color: #A0522D;");
                    else {
                        cellButtons[i][j].setStyle("-fx-background-color: #FFEFD5;");
                    }
                }
            }
        }
    }

    // Найти все клетки, до которых можно добраться за N ходов
    public List<Cell> searchReachableCells(int[][] field, int startX, int startY, int N) {
        List<Cell> reachableCells = new ArrayList<>();
        search(field, startX, startY, 0, 0, N, N, reachableCells);

        return reachableCells;
    }

    // Рекурсия
    private void search(int[][] field, int  x, int y, int oldX, int oldY, int movesLeft, int step, List<Cell> reachableCells) {
        if (movesLeft == 0) {
            Cell currentCell = new Cell(x, y);
            int count = 0;
            for (Cell d: reachableCells){
                if (d.x == x && y == d.y) {
                    count++;
                }
            }
            if(count == 0)
            {
                reachableCells.add(currentCell);
                str += "(" +(x - oldX) + " , " + (y - oldY) + ") -> " + "(" +   (x)  + " , " + (y) + ")\n";
                System.out.println("(" +(x - oldX) + " , " + (y - oldY) + ") -> " + "(" +   (x)  + " , " + (y) + ")");

                List<CellsPair> entry = hashtable1.get(step);
                if (entry == null) {
                    entry = new ArrayList<>();
                }

                entry.add(new CellsPair(new Cell(x - oldX, y - oldY), currentCell));

                hashtable1.put(step, entry);
            }

            return;
        }

        visited[x][y] = false;
        if (field[x][y] == 0) {
            for (int[] move : horseMovement) {
                int newRowX = x + move[0];
                int newColY = y + move[1];

                if (isValidMove(field, newRowX, newColY)) {
                    search(field, newRowX, newColY,move[0],move[1], movesLeft - 1, step, reachableCells);
                }
            }
        }
    }

    // Проверка
    private static boolean isValidMove(int[][] field, int x, int y) {
        return x >= 0 && y >= 0 && x < field.length && y < field[0].length && field[x][y] == 0;
    }

    private void addWalls() {
        // Генерируем случайные стены
        Random random = new Random();

        // Установка стен
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (i == selectedRowX && j == selectedColY) {
                    continue; // Пропускаем позицию коня
                }

                // Вероятность установки стены: 30%
                if (random.nextDouble() < 0.3) {
                    grid[i][j] = 1; // Устанавливаем стену
                    cellButtons[i][j].setStyle("-fx-background-color: #ee4949;");
                    cellButtons[i][j].setFont(new Font("Roboto", 15));
                    cellButtons[i][j].setText("✖");
                }
            }
        }
    }

    // Клетка
    static class Cell {
        int x;
        int y;
        Cell(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    static class MarkedCell {
        int step;
        Cell cell;
        MarkedCell(Cell cell, int step) {
            this.cell = cell;
            this.step = step;
        }
    }

    static class CellsPair {
        int step;
        Cell fromCell;
        Cell toCell;
        CellsPair(Cell fromCell, Cell toCell) {
            this.fromCell = fromCell;
            this.toCell = toCell;
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}