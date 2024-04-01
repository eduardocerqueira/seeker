//date: 2024-04-01T16:53:13Z
//url: https://api.github.com/gists/7b26de0dba3051b353874743980642e6
//owner: https://api.github.com/users/ThomasWega

package me.wega.rpgambling;

import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerChatEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.plugin.Plugin;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.Predicate;

/**
 * A utility class to use player chat input using callbacks
 *
 * @param <T> Type to parse player input to
 * @author <a href="https://github.com/ThomasWega">Tomáš Weglarz</a>
 */
public class ChatCallback<T> implements Listener {
    private static final Map<UUID, ChatCallback<?>> runningChatConsumers = new HashMap<>();
    private final IParser<T> parser;
    private Consumer<T> onSuccess;
    private Consumer<Player> onCancel;
    private Consumer<String> onFail;
    private Predicate<T> onCondition;
    private final Player player;

    /**
     * Constructs a new ChatCallback instance.
     *
     * @param plugin The plugin instance.
     * @param player The player associated with this callback.
     * @param parser The parser used to parse the chat input into a desired type.
     */
    public ChatCallback(Plugin plugin, Player player, IParser<T> parser) {
        this.player = player;
        this.parser = parser;
        ChatCallback<?> previousCallback = runningChatConsumers.put(player.getUniqueId(), this);
        if (previousCallback != null)
            previousCallback.cancel();
        plugin.getServer().getPluginManager().registerEvents(this, plugin);
    }

    /**
     * Sets the callback to be executed upon successful parsing of the chat input.
     *
     * @param callback The onSuccess callback.
     * @return This ChatCallback instance.
     */
    public ChatCallback<T> onSuccess(Consumer<T> callback) {
        onSuccess = callback;
        return this;
    }

    /**
     * Sets the callback to be executed when the player cancels the chat input.
     *
     * @param callback The onCancel callback.
     * @return This ChatCallback instance.
     */
    public ChatCallback<T> onCancel(Consumer<Player> callback) {
        onCancel = callback;
        return this;
    }

    /**
     * Sets the callback to be executed when the chat input fails to parse.
     *
     * @param callback The onFail callback.
     * @return This ChatCallback instance.
     */
    public ChatCallback<T> onFail(Consumer<String> callback) {
        onFail = callback;
        return this;
    }

    /**
     * Sets the condition predicate to be checked before executing the onSuccess callback.
     *
     * @param condition The onCondition predicate.
     * @return This ChatCallback instance.
     */
    public ChatCallback<T> onCondition(Predicate<T> condition) {
        onCondition = condition;
        return this;
    }

    /**
     * Cancels the chat callback and removes it from the running callbacks map.
     */
    public void cancel() {
        onSuccess = null;
        onCancel = null;
        runningChatConsumers.remove(player.getUniqueId());
    }

    @EventHandler(ignoreCancelled = true)
    private void onPlayerQuit(PlayerQuitEvent event) {
        if (!player.equals(event.getPlayer())) return;
        cancel();
    }


    @EventHandler
    private void onPlayerChat(PlayerChatEvent event) {
        if (onSuccess == null && onCancel == null) return;

        event.setCancelled(true);
        String msg = event.getMessage();

        if (msg.equalsIgnoreCase("cancel") && onCancel != null) {
            onCancel.accept(player);
            cancel();
            return;
        }

        T parsedValue = parser.parse(msg);
        if (parsedValue == null) {
            onFail.accept(msg);
            return;
        }

        if (!onCondition.test(parsedValue)) return;

        if (onSuccess != null) {
            onSuccess.accept(parsedValue);
            cancel();
        }
    }

    /**
     * The interface for parsing chat input into a desired type.
     *
     * @param <T> The type of the parsed value.
     */
    public interface IParser<T> {
        T parse(String input);
    }

    /**
     * Provides built-in parsers for String, Integer, and Double types.
     */
    public static class Parser {
        /**
         * A parser that returns the input string as is.
         */
        public static final IParser<String> STRING = input -> input;

        /**
         * A parser that parses the input string into an Integer.
         */
        public static final IParser<Integer> INTEGER = input -> {
            try {
                return Integer.valueOf(input);
            } catch (NumberFormatException e) {
                return null;
            }
        };

        /**
         * A parser that parses the input string into a Double.
         */
        public static final IParser<Double> DOUBLE = input -> {
            try {
                return Double.valueOf(input);
            } catch (NumberFormatException e) {
                return null;
            }
        };
    }
}
