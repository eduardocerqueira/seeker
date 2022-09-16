//date: 2022-09-16T21:52:28Z
//url: https://api.github.com/gists/2cbc88b82bf50b1a8ba98568210b95b7
//owner: https://api.github.com/users/codebyxemu

package me.xemu.haymc.listener;

import me.xemu.haymc.HayCore;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerJoinEvent;

public class JoinListener implements Listener {

	@EventHandler
	public void onPlayerJoin(PlayerJoinEvent event) {
		Player player = event.getPlayer();

		HayCore.getInstance().getProfileManager().handleProfileCreation(player.getUniqueId(), player.getName());
	}

}
