//date: 2022-05-10T17:18:34Z
//url: https://api.github.com/gists/ccb11eda8789ac029db921b99d5f5a5d
//owner: https://api.github.com/users/Dani-Sotano

package tictactoe;

import tictactoe.userinterface.UserInterface;

import java.util.List;
import java.util.Map;

public class TicTacToeGame {

    private final State state = new State();
    protected UserInterface userInterface;

    public TicTacToeGame(UserInterface userInterface) {
        this.userInterface = userInterface;
        this.start();
    }

    public void start(){
        this.userInterface.nextOutput("Heyho, let's play TicTacToe!");
        firstPlayerNameAndSymbolAreDefined();
        secondPlayerNameAndSymbolAreDefined();

        gameIsPlayedAsLongAsPlayersWantTo();
    }

    private void gameIsPlayedAsLongAsPlayersWantTo() {
        while (true) {
            newRoundIsPrepared();
            boardIsShown();

            while (!roundIsFinished()) {
                nextPlayerIsAskedToChooseField();
                boardIsShown();
            }

            resultsOfRoundAreShown();
            if (!playersWantToContinuePlaying()){
                break;
            }
        }
    }

    private void newRoundIsPrepared() {
        state.resetForNewGame();
        this.userInterface.nextOutput("Let's start!");
    }

    private void currentPlayerIsUpdated() {
        state.setCurrentPlayerId(
                state.getCurrentPlayerId() == State.PLAYER_ID.FIRST ? State.PLAYER_ID.SECOND : State.PLAYER_ID.FIRST
        );
    }

    private void firstPlayerNameAndSymbolAreDefined() {
        playerNameAndSymbolAreDefined(State.PLAYER_ID.FIRST, "first", "X");
    }

    private void secondPlayerNameAndSymbolAreDefined() {
        playerNameAndSymbolAreDefined(State.PLAYER_ID.SECOND, "second", "O");
    }

    void playerNameAndSymbolAreDefined(State.PLAYER_ID playerId, String identifier, String symbol) {
        this.userInterface.nextOutput("Please enter the name of the "+identifier+" player.");
        String playerName = playerIsAskedForName();
        state.setPlayer(playerId, playerName, symbol);
        this.userInterface.nextOutput("Ok, you have the Symbol " + symbol);
    }

    private String playerIsAskedForName() {
        String playerName = userInterface.getNextInput();

        while (playerName.isEmpty()) {
            this.userInterface.nextOutput("You did not enter any name. Please try again:");
            playerName = userInterface.getNextInput();
        }
        return playerName;
    }

    private void boardIsShown() {
        Map<String, String> fields = state.getFields();
        this.userInterface.nextOutput("   1   2   3 ");
        this.userInterface.nextOutput("A  " + fields.get("A1") + " | " + fields.get("A2") + " | " + fields.get("A3") + " ");
        this.userInterface.nextOutput("   -----------");
        this.userInterface.nextOutput("B  " + fields.get("B1") + " | " + fields.get("B2") + " | " + fields.get("B3") + " ");
        this.userInterface.nextOutput("   -----------");
        this.userInterface.nextOutput("C  " + fields.get("C1") + " | " + fields.get("C2") + " | " + fields.get("C3") + " ");
    }

    private boolean roundIsFinished() {
        return currentPlayerHasWon() || allFieldsAreOccupied();
    }

    private boolean currentPlayerHasWon() {
        return playerOccupiesAnyRow()
                || playerOccupiesAnyColumn()
                || playerOccupiesAnyDiagonal();
    }

    private boolean playerOccupiesFieldsInList(List<String> fields) {
        // every field in the list contains the player symbol
        return fields.stream().allMatch(field ->
                state.getFields().get(field).equals(state.getCurrentPlayer().symbol())
        );
    }

    private boolean playerOccupiesAnyDiagonal() {
        return playerOccupiesFieldsInList(List.of("A1", "B2", "C3"))
                || playerOccupiesFieldsInList(List.of("C1", "B2", "A3"));
    }


    private boolean playerOccupiesAnyColumn() {
        return playerOccupiesFieldsInList(List.of("A1", "B1", "C1"))
                || playerOccupiesFieldsInList(List.of("A2", "B2", "C2"))
                || playerOccupiesFieldsInList(List.of("A3", "B3", "C3"));
    }

    private boolean playerOccupiesAnyRow() {
        return playerOccupiesFieldsInList(List.of("A1", "A2", "A3"))
                || playerOccupiesFieldsInList(List.of("B1", "B2", "B3"))
                || playerOccupiesFieldsInList(List.of("C1", "C2", "C3"));
    }

    void nextPlayerIsAskedToChooseField() {
        currentPlayerIsUpdated();
        Player currentPlayer = state.getCurrentPlayer();

        this.userInterface.nextOutput(currentPlayer.name() + ", please select a field e.g. A1");

        String field = playerHasProvidedField();
        fieldIsUpdatedWithPlayerSymbol(field);
    }

    private void fieldIsUpdatedWithPlayerSymbol(String field) {
        state.getFields().put(field, state.getCurrentPlayer().symbol());
    }

    String playerHasProvidedField() {
        String input;
        input = userInterface.getNextInput();
        while (fieldInputIsRejected(input)) {
            input = userInterface.getNextInput();
        }
        return input;
    }

    private boolean fieldInputIsRejected(String input) {
        if (inputDoesNotCorrespondToAFieldName(input)) {
            this.userInterface.nextOutput(input + " is not a valid field name. Valid examples are A1, B3, C2, etc");
            return true;
        }

        String fieldName = input;
        if (fieldIsOccupied(fieldName)) {
            this.userInterface.nextOutput(state.getCurrentPlayer().name() + ", this field is already occupied. Please choose an empty field");
            return true;
        }

        // everything is fine
        return false;
    }

    private boolean fieldIsOccupied(String field) {
        return !state.getFields().get(field).equals(state.EMPTY_FIELD);
    }

    private boolean inputDoesNotCorrespondToAFieldName(String field) {
        return !state.getFields().containsKey(field);
    }

    private void resultsOfRoundAreShown() {
        if (currentPlayerHasWon()) {
            this.userInterface.nextOutput("Wow, " + state.getCurrentPlayer().name() + ", I think you are a professional, you won.");
        } else if (allFieldsAreOccupied()) {
            this.userInterface.nextOutput("Wow, you are both masters, you both won.");
        }
    }

    private boolean allFieldsAreOccupied() {
        return state.getFields().entrySet().stream()
                .noneMatch(field -> field.getValue().equals(state.EMPTY_FIELD));
    }

    private boolean playersWantToContinuePlaying() {
        this.userInterface.nextOutput("Do you want to play another round? [Y/N]");

        while (true) {
            String input = userInterface.getNextInput();
            if (input.equals("Y")) {
                return true;
            }
            if (input.equals("N")) {
                return false;
            }
            this.userInterface.nextOutput("Your input can not be processed. Please type Y if you want to try again and N if not.");
        }
    }


}