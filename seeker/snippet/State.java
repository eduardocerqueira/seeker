//date: 2022-05-10T17:18:34Z
//url: https://api.github.com/gists/ccb11eda8789ac029db921b99d5f5a5d
//owner: https://api.github.com/users/Dani-Sotano

package tictactoe;

import java.util.HashMap;
import java.util.Map;

public class State {
    private final Map<PLAYER_ID, Player> players = new HashMap<>();
    public enum PLAYER_ID {FIRST, SECOND}

    public final String EMPTY_FIELD = " ";
    private final Map<String, String> fields = new HashMap<>(Map.of(
            "A1", EMPTY_FIELD,
            "A2", EMPTY_FIELD,
            "A3", EMPTY_FIELD,
            "B1", EMPTY_FIELD,
            "B2", EMPTY_FIELD,
            "B3", EMPTY_FIELD,
            "C1", EMPTY_FIELD,
            "C2", EMPTY_FIELD,
            "C3", EMPTY_FIELD
    ));

    private PLAYER_ID currentPlayerId = PLAYER_ID.SECOND;

    public State() {
    }

    public void setPlayer(PLAYER_ID id, String name, String symbol){
        this.players.put(id, new Player(name, symbol));
    }

    public Map<String, String> getFields(){
        return this.fields;
    }

    public Player getCurrentPlayer() {
        return this.players.get(currentPlayerId);
    }

    public void setCurrentPlayerId(PLAYER_ID id) {
        this.currentPlayerId = id;
    }

    public void resetForNewGame() {
        this.fields.keySet().forEach(key -> this.fields.put(key, EMPTY_FIELD));
    }

    public PLAYER_ID getCurrentPlayerId() {
        return this.currentPlayerId;
    }
}