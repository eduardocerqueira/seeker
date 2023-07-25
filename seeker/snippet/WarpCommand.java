//date: 2023-07-25T16:58:50Z
//url: https://api.github.com/gists/5d0809bcf06f22ebb833f2fb2b4d201a
//owner: https://api.github.com/users/meltoid872

package com.example.warpplugin;

import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.Location;
import org.bukkit.command.Command;

import com.example.warpplugin.Main;

 class WarpCommand implements CommandExecutor {

    private final Main plugin;

    public WarpCommand(Main plugin) {
        this.plugin = plugin;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (!(sender instanceof Player)) {
            sender.sendMessage("Only players can use this command.");
            return true;
        }

        if (args.length != 1) {
            sender.sendMessage("Usage: /warp <warp_name>");
            return true;
        }

        Player player = (Player) sender;
        String warpName = args[0];
        Location warpLocation = plugin.getConfig().getLocation("warps." + warpName);

        if (warpLocation != null) {
            player.teleport(warpLocation);
            player.sendMessage("You have been teleported to warp '" + warpName + "'!");
        } else {
            player.sendMessage("Warp '" + warpName + "' doesn't exist.");
        }

        return true;
    }
}