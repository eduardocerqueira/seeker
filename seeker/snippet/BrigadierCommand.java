//date: 2024-01-02T16:55:35Z
//url: https://api.github.com/gists/8930ca10e2ad60059a12c6e5ad3355da
//owner: https://api.github.com/users/TnTGamesTV

package de.throwstnt.developing.mmc.lib.base.commands;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Lists;
import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.ParseResults;
import com.mojang.brigadier.arguments.ArgumentType;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import com.mojang.brigadier.builder.RequiredArgumentBuilder;
import com.mojang.brigadier.exceptions.CommandSyntaxException;
import com.mojang.brigadier.tree.CommandNode;

import org.bukkit.command.CommandMap;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.command.Command;

public abstract class BrigadierCommand extends Command {

    private static final List<String> CONST_IGNORE_ERRORS = Arrays.asList("Incorrect argument for command", "Unknown");

    public BrigadierCommand(String name, String title, String description, String fallbackPrefix,
            List<String> aliases) {
        this(name, "Server", title, description, fallbackPrefix, aliases);
    }

    public BrigadierCommand(String name, String prefix, String title, String description, String fallbackPrefix,
            List<String> aliases) {
        super(name, description, "/" + name, aliases);

        this.prefix = prefix;
        this.title = title;
        this.fallbackPrefix = fallbackPrefix;
        this.dispatcher = new CommandDispatcher<>();
    }

    private String prefix;
    private String title;
    private String fallbackPrefix;
    private CommandDispatcher<Player> dispatcher;

    public void registerSelf(CommandMap map) {
        map.register(getName(), fallbackPrefix, this);

        LiteralArgumentBuilder<Player> root = LiteralArgumentBuilder.literal(getName());

        init(root);

        this.getDispatcher().register(root);
    }

    public CommandDispatcher<Player> getDispatcher() {
        return this.dispatcher;
    }

    public abstract void init(LiteralArgumentBuilder<Player> root);

    @Override
    public boolean execute(CommandSender sender, String commandLabel, String[] args) {
        if (sender instanceof Player) {
            Player player = (Player) sender;

            List<String> stringArgs = Lists.newArrayList(args);
            String input = getName() + " " + stringArgs.stream()
                    .reduce((result, element) -> result == null ? element : result + " " + element).orElse("");

            ParseResults<Player> result = dispatcher.parse(input, player);

            try {
                dispatcher.execute(result);
            } catch (CommandSyntaxException exception) {
                CommandNode<Player> failureNode = result.getContext()
                        .findSuggestionContext(exception.getCursor()).parent;
                if (input.trim().endsWith("help") || input.trim().endsWith("?")) {
                    this._sendHelp(player, failureNode);
                } else {
                    this._sendSyntaxError(player, failureNode, exception);
                }
            }
        }
        return true;
    }

    private void _sendHelp(Player player, CommandNode<Player> failureNode) {
        String[] usages = this.dispatcher.getAllUsage(failureNode, player, false);

        String pathToFailure = this.dispatcher.getPath(failureNode).stream()
                .reduce((tmpResult, element) -> tmpResult == null ? element : tmpResult + " " + element).orElse("");

        String prefix = "§1[§9" + this.prefix + "§1] §7";

        player.sendMessage(prefix + "Help for §9" + title + "§7:");
        Lists.newArrayList(usages).forEach(usage -> {
            String space = pathToFailure.length() > 0 ? " " : "";

            String coloredCommand = "§8/" + pathToFailure + space + "§7" + usage;

            player.sendMessage(prefix + coloredCommand);
        });
    }

    private void _sendSyntaxError(Player player, CommandNode<Player> failureNode, CommandSyntaxException exception) {
        Map<CommandNode<Player>, String> failureNodeUsage = this.dispatcher.getSmartUsage(failureNode, player);

        String pathToFailure = this.dispatcher.getPath(failureNode).stream()
                .reduce((tmpResult, element) -> tmpResult == null ? element : tmpResult + " " + element).orElse("");

        String prefix = "§1[§9" + this.prefix + "§1] §7";

        player.sendMessage(prefix + "There was an §cissue §7with the command");
        failureNodeUsage.entrySet().forEach(entry -> {
            String space = pathToFailure.length() > 0 ? " " : "";

            String coloredCommand = "§8/" + pathToFailure + space + "§7" + entry.getValue();

            player.sendMessage(prefix + coloredCommand);
        });

        String exceptionMessage = exception.getRawMessage().getString();

        if (CONST_IGNORE_ERRORS.stream().noneMatch(exceptionMessage::contains)) {
            player.sendMessage(prefix + "§cError§7: " + exception.getRawMessage().getString());
        }
    }

    public LiteralArgumentBuilder<Player> literal(String name) {
        return LiteralArgumentBuilder.literal(name);
    }

    public RequiredArgumentBuilder<Player, ?> required(String name, ArgumentType<?> type) {
        return RequiredArgumentBuilder.argument(name, type);
    }

    public IntegerArgumentType integer() {
        return IntegerArgumentType.integer();
    }
}