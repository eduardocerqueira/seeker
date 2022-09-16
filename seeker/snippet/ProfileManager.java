//date: 2022-09-16T21:51:19Z
//url: https://api.github.com/gists/ec0d09ddd1c109442844cefd90533f44
//owner: https://api.github.com/users/codebyxemu

package me.xemu.haymc.structure.data;

import me.xemu.haymc.HayCore;
import org.bukkit.entity.Player;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

public class ProfileManager extends Manager {

	private Map<UUID, Profile> profiles = new HashMap<>();

	public ProfileManager(HayCore plugin) {
		super(plugin);
	}

	public void handleProfileCreation(UUID uuid, String name) {
		if (!this.profiles.containsKey(uuid)) {
			profiles.put(uuid, new Profile(uuid, name));
		}
	}

	public Profile getProfile(Object object) {
		if (object instanceof Player) {
			Player target = (Player) object;
			if (!this.profiles.containsKey(target.getUniqueId())) {
				return null;
			}
			return profiles.get(target.getUniqueId());
		}
		if (object instanceof UUID) {
			UUID uuid = (UUID) object;
			if (!this.profiles.containsKey(uuid)) {
				return null;
			}
			return profiles.get(uuid);
		}
		if (object instanceof String) {
			return this.profiles.values().stream().filter(profile -> profile.getPlayerName().equalsIgnoreCase(object.toString())).findFirst().orElse(null);
		}
		return null;
	}

	public Map<UUID, Profile> getProfiles() {
		return this.profiles;
	}

	public void setProfiles(Map<UUID, Profile> profiles) {
		this.profiles = profiles;
	}


}
