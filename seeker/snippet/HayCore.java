//date: 2022-09-16T21:49:36Z
//url: https://api.github.com/gists/ec152918b3e0ffae7ca1105cd799b44a
//owner: https://api.github.com/users/codebyxemu

package me.xemu.haymc;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import lombok.Getter;
import me.xemu.haymc.implement.database.HayDatabase;
import me.xemu.haymc.structure.AdvancedLicense;
import me.xemu.haymc.structure.data.ProfileManager;
import me.xemu.haymc.util.HayConfig;
import me.xemu.haymc.util.messages.MessageBuilder;
import org.bson.Document;
import org.bukkit.plugin.java.JavaPlugin;

import java.util.logging.Level;
import java.util.logging.Logger;

public class HayCore extends JavaPlugin {

	private static HayCore instance;

	public static boolean DEBUG;

	private static HayConfig messagesConfig;

	@Getter
	private MongoClient mongoClient;
	@Getter private MongoDatabase mongoDatabase;
	@Getter private MongoCollection<Document> serverCollection;
	@Getter private ProfileManager profileManager;


	@Override
	public void onEnable() {
		if(!new AdvancedLicense(getConfig().getString("key"), "http://license.haymc.eu/verify.php", this).setSecurityKey("YecoF0I6M05thxLeokoHuW8iUhTdIUInjkfF").register()) return;

		instance = this;

		DEBUG = getConfig().getBoolean("debug");
		messagesConfig = new HayConfig(this, "messages.yml");

		new HayRegistration();

		mongoDatabase = HayDatabase.getDatabase();
		serverCollection = HayDatabase.getCollection();

		// handle mongodb
		System.setProperty("DEBUG.GO", "true");
		System.setProperty("DB.TRACE", "true");
		Logger mongoLogger = Logger.getLogger("org.mongodb.driver");
		mongoLogger.setLevel(Level.WARNING);
		this.profileManager = new ProfileManager(this);

		HayDatabase.connect();

		new MessageBuilder("*----------------------------------------*").console(false);
		new MessageBuilder("Welcome to HayMC-Core (Version <ver>)")
				.setPlaceholder("<ver>", getDescription().getVersion())
				.console(false);
		new MessageBuilder("Only to be used with HayMC.").console(false);
		new MessageBuilder("Written by Xemu & HayMC Development Team").console(false);
		new MessageBuilder("*----------------------------------------*").console(false);

	}

	@Override
	public void onDisable() {
		instance = null;

		new MessageBuilder("*----------------------------------------*").console(false);
		new MessageBuilder("HayMC-Core was successfully disabled. All data saved!").console(false);
		new MessageBuilder("*----------------------------------------*").console(false);

	}

	/* STATIC GETTERS */
	public static HayCore getInstance() {
		return instance;
	}
	public static HayConfig getMessagesConfig() {
		return messagesConfig;
	}

	/* HANDLE MONGODB */

}
