//date: 2022-09-16T21:50:25Z
//url: https://api.github.com/gists/089807265f21c58815d8c286664849c5
//owner: https://api.github.com/users/codebyxemu

package me.xemu.haymc.structure.data.db;

import com.mongodb.client.model.Filters;
import com.mongodb.client.model.UpdateOptions;
import lombok.Getter;
import lombok.Setter;
import me.xemu.haymc.HayCore;
import me.xemu.haymc.rank.Rank;
import me.xemu.haymc.structure.data.Manager;
import me.xemu.haymc.structure.data.ProfileManager;
import org.bson.Document;

import java.util.UUID;

@Getter
@Setter
public class PlayerData {

	private UUID uuid;
	private String playerName;

	private String rank;
	private int networkLevel;
	private int networkExperience;

	public PlayerData(UUID uuid, String name) {
		this.uuid = uuid;
		this.playerName = name;

		if (HayCore.getInstance().getServerCollection().find(Filters.eq("uuid", getUuid().toString())).first() == null) {
			this.rank = Rank.PLAYER.toString();
			this.networkLevel = 1;
			this.networkExperience = 1000;
			HayCore.getInstance().getProfileManager().handleProfileCreation(uuid, name);
			save();
		}
	}

	public void load() {
		Document document = HayCore.getInstance().getServerCollection().find(Filters.eq("uuid", getUuid().toString())).first();

		if (document != null) {
			this.rank = document.getString("rank");
			this.networkLevel = document.getInteger("network-level");
			this.networkExperience = document.getInteger("network-experience");
		}
	}

	public void save() {
		Document document = new Document();
		document.put("uuid", getUuid().toString());
		document.put("name", getPlayerName().toLowerCase());
		document.put("realName", getPlayerName());
		document.put("rank", getRank());
		document.put("network-level", getNetworkLevel());
		document.put("network-experience", getNetworkExperience());
		HayCore.getInstance().getServerCollection().replaceOne(Filters.eq("uuid", getUuid().toString()), document, new UpdateOptions().upsert(true));
	}

}
