//date: 2023-07-25T16:58:50Z
//url: https://api.github.com/gists/5d0809bcf06f22ebb833f2fb2b4d201a
//owner: https://api.github.com/users/meltoid872

package com.example.warpplugin.Listener;

import com.example.warpplugin.Main;
import org.bukkit.Particle;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerMoveEvent;
import org.bukkit.scheduler.BukkitRunnable;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

class PlayerMoveListener implements Listener {
    private final Main plugin;
    private final Map<UUID, Long> teleportCooldown;

    public PlayerMoveListener(Main plugin) {
        this.plugin = plugin;
        this.teleportCooldown = new HashMap<>();
    }

    @EventHandler
    public void onPlayerMove(PlayerMoveEvent event) {
        Player player = event.getPlayer();
        UUID playerId = player.getUniqueId();

        if (teleportCooldown.containsKey(playerId)) {
            long lastTeleportTime = teleportCooldown.get(playerId);
            long currentTime = System.currentTimeMillis();

            long cooldownTime = 3000;

            if (currentTime - lastTeleportTime < cooldownTime) {
                player.getWorld().spawnParticle(Particle.SMOKE_NORMAL, player.getLocation(), 30, 0.5, 0.5, 0.5);
            } else {
                teleportCooldown.remove(playerId);
            }
        }
    }

    public void addCooldown(Player player) {
        UUID playerId = player.getUniqueId();

        int cooldownTicks = 60;

        teleportCooldown.put(playerId, System.currentTimeMillis());
        new BukkitRunnable() {
            @Override
            public void run() {
                teleportCooldown.remove(playerId);
            }
        }.runTaskLater(plugin, cooldownTicks);
    }}