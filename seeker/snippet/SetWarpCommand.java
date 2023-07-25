//date: 2023-07-25T16:58:50Z
//url: https://api.github.com/gists/5d0809bcf06f22ebb833f2fb2b4d201a
//owner: https://api.github.com/users/meltoid872

package com.example.warpplugin;

import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.Location;

import com.example.warpplugin.Main;
import com.example.warpplugin.PlayerMoveListener;

public class SetWarpCommand implements CommandExecutor {

    private final Main plugin;

    public SetWarpCommand(Main plugin) {
        this.plugin = plugin;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (!(sender instanceof Player)) {
            sender.sendMessage("Only players can set a warp.");
            return true;
        }

        if (args.length != 1) {
            sender.sendMessage("Usage: /setwarp <warp_name>");
            return true;
        }

        Player player = (Player) sender;
        String warpName = args[0];
        Location warpLocation = player.getLocation();

        plugin.getConfig().set("warps." + warpName + ".world", warpLocation.getWorld().getName());
        plugin.getConfig().set("warps." + warpName + ".x", warpLocation.getX());
        plugin.getConfig().set("warps." + warpName + ".y", warpLocation.getY());
        plugin.getConfig().set("warps." + warpName + ".z", warpLocation.getZ());
        plugin.getConfig().set("warps." + warpName + ".yaw", warpLocation.getYaw());
        plugin.getConfig().set("warps." + warpName + ".pitch", warpLocation.getPitch());

        plugin.saveConfig();
        player.sendMessage("Warp '" + warpName + "' has been set!");


        PlayerMoveListener moveListener = new PlayerMoveListener(plugin);
        moveListener.addCooldown(player);

        return true;
    }
}