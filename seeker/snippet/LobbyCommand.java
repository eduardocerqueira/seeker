//date: 2024-01-02T16:59:44Z
//url: https://api.github.com/gists/5c02fa028073937c721de1c851bdd9be
//owner: https://api.github.com/users/TnTGamesTV

package de.throwstnt.developing.mmc.lobby.commands;

import java.util.Arrays;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import org.bukkit.entity.Player;
import de.throwstnt.developing.mmc.lib.base.commands.BrigadierCommand;

public class LobbyCommand extends BrigadierCommand {

    public LobbyCommand() {
        super("managelobby", "Lobby", "Lobby", "Manage the lobby server",
                Arrays.asList("ml"));
    }

    @Override
    public void init(LiteralArgumentBuilder<Player> root) {
        root.then(literal("shop").executes(e -> {
            //DO SOMETHING

            return 1;
        })).then(literal("inventory3").executes(e -> {
           //DO SOMETHING

            return 1;
        }));
    }

}