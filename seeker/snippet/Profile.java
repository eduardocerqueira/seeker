//date: 2022-09-16T21:50:58Z
//url: https://api.github.com/gists/003d09be1a221dbf15ae6c35520fa3dc
//owner: https://api.github.com/users/codebyxemu

package me.xemu.haymc.structure.data;

import lombok.Getter;
import lombok.Setter;
import me.xemu.haymc.structure.data.db.PlayerData;

import java.util.UUID;

@Getter
public class Profile {

	private PlayerData data;
	private UUID UUID;
	private String playerName;

	public Profile(UUID uuid, String name) {
		this.UUID = uuid;
		this.playerName = name;
		this.data = new PlayerData(uuid, name);
	}

}
