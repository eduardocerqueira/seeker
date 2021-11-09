//date: 2021-11-09T17:17:44Z
//url: https://api.github.com/gists/3dd153cc7d5373a295238981f043017e
//owner: https://api.github.com/users/FormerCanuck

package me.formercanuck.formersessentials.commnand;

import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.exceptions.CommandSyntaxException;
import net.minecraft.ChatFormatting;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.TextComponent;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.level.levelgen.Heightmap;

import java.util.Random;

public class RTPCommand {

    Random random;

    public RTPCommand(CommandDispatcher<CommandSourceStack> dispatcher) {
        dispatcher.register(Commands.literal("rtp").executes((command) -> rtp(command.getSource())));
    }

    private int rtp(CommandSourceStack source) throws CommandSyntaxException {
        random = new Random();

        ServerPlayer player = source.getPlayerOrException();

        BlockPos pos = new BlockPos(player.getX() + random.nextInt(10000), 20, player.getZ() + random.nextInt(10000));

        System.out.println(
                player.getLevel().getHeight(Heightmap.Types.WORLD_SURFACE, pos.getX(), pos.getZ()));

        player.moveTo(pos.getX(),
                player.getLevel().getHeight(Heightmap.Types.WORLD_SURFACE, pos.getX(),
                        pos.getZ()), pos.getZ());

        TextComponent textComponent = new TextComponent(ChatFormatting.GOLD + "You have been randomly teleported.");

        source.sendSuccess(textComponent, true);

        return 1;
    }
}
