//date: 2022-09-26T17:10:42Z
//url: https://api.github.com/gists/f34f358b7ad6b763160675c96ae1d795
//owner: https://api.github.com/users/akbaryahya

package emu.grasscutter.tools;

import static emu.grasscutter.config.Configuration.RESOURCE;

import emu.grasscutter.GameConstants;
import emu.grasscutter.Grasscutter;
import emu.grasscutter.command.CommandHandler;
import emu.grasscutter.command.CommandMap;
import emu.grasscutter.data.GameData;
import emu.grasscutter.data.ResourceLoader;
import emu.grasscutter.data.excels.AvatarData;
import emu.grasscutter.data.excels.ItemData;
import emu.grasscutter.data.excels.QuestData;
import emu.grasscutter.utils.Language;
import emu.grasscutter.utils.Language.TextStrings;
import emu.grasscutter.utils.Utils;
import it.unimi.dsi.fastutil.ints.Int2IntRBTreeMap;
import it.unimi.dsi.fastutil.ints.Int2IntSortedMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public final class Tools {

	public static void createGmHandbooks() throws Exception {
		final List<Language> languages = Language.TextStrings.getLanguages();
		final Int2ObjectMap<TextStrings> textMaps = Language.getTextMapStrings();

		ResourceLoader.loadAll();

		final Int2IntSortedMap avatarNames = new Int2IntRBTreeMap(
			GameData
				.getAvatarDataMap()
				.int2ObjectEntrySet()
				.stream()
				.collect(Collectors.toMap(e -> (int) e.getIntKey(), e -> (int) e.getValue().getNameTextMapHash()))
		);
		final Int2IntSortedMap itemNames = new Int2IntRBTreeMap(
			GameData
				.getItemDataMap()
				.int2ObjectEntrySet()
				.stream()
				.collect(Collectors.toMap(e -> (int) e.getIntKey(), e -> (int) e.getValue().getNameTextMapHash()))
		);
		final Int2IntSortedMap monsterNames = new Int2IntRBTreeMap(
			GameData
				.getMonsterDataMap()
				.int2ObjectEntrySet()
				.stream()
				.collect(Collectors.toMap(e -> (int) e.getIntKey(), e -> (int) e.getValue().getNameTextMapHash()))
		);
		final Int2IntSortedMap mainQuestTitles = new Int2IntRBTreeMap(
			GameData
				.getMainQuestDataMap()
				.int2ObjectEntrySet()
				.stream()
				.collect(Collectors.toMap(e -> (int) e.getIntKey(), e -> (int) e.getValue().getTitleTextMapHash()))
		);

		// Preamble
		final List<StringBuilder> handbookBuilders = new ArrayList<>(TextStrings.NUM_LANGUAGES);
		final String now = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss").format(LocalDateTime.now());
		for (int i = 0; i < TextStrings.NUM_LANGUAGES; i++) handbookBuilders.add(
			new StringBuilder()
				.append("// Grasscutter " + Utils.GetLast(GameConstants.VERSION) + " GM Handbook\n")
				.append("// Created " + now + "\n\n")
				.append("// Commands\n")
		);
		// Commands
		final List<CommandHandler> cmdList = CommandMap.getInstance().getHandlersAsList();
		final String padCmdLabel =
			"%" +
			cmdList.stream().map(CommandHandler::getLabel).map(String::length).max(Integer::compare).get().toString() +
			"s : ";
		for (CommandHandler cmd : cmdList) {
			final String label = padCmdLabel.formatted(cmd.getLabel());
			final String descKey = cmd.getDescriptionKey();
			for (int i = 0; i < TextStrings.NUM_LANGUAGES; i++) {
				String desc = languages.get(i).get(descKey).replace("\n", "\n\t\t\t\t").replace("\t", "    ");
				handbookBuilders.get(i).append(label + desc + "\n");
			}
		}

		// Gadgets
		handbookBuilders.forEach(b -> b.append("// Gadgets\n"));
		final var GadgetDataMap = GameData.getGadgetDataMap();
		GadgetDataMap
			.keySet()
			.intStream()
			.sorted()
			.forEach(id -> {
				final String data = GadgetDataMap.get(id).getJsonName();
				handbookBuilders.forEach(b -> b.append(id + " : " + (data.equals("") ? "Unknown" : data) + "\n"));
			});

		//Avatars, Items, Monsters
		final String[] handbookSections = { "Avatars", "Items", "Monsters" };
		final Int2IntSortedMap[] handbookNames = { avatarNames, itemNames, monsterNames };
		for (int section = 0; section < handbookSections.length; section++) {
			final var h = handbookNames[section];
			final String s = "// " + handbookSections[section] + "\n";

			handbookBuilders.forEach(b -> b.append(s));

			h.forEach((id, hash) -> {
				final TextStrings t = textMaps.get((int) hash);

				String data = "";
				for (int i = 0; i < TextStrings.NUM_LANGUAGES; i++) {
					if (t != null) {
						data = t.strings[i];
					}
					handbookBuilders.get(i).append(id + " : " + (data.equals("") ? "Unknown" : data) + "\n");
				}
			});
		}

		// Scenes - no translations
		handbookBuilders.forEach(b -> b.append("// Scenes\n"));
		final var sceneDataMap = GameData.getSceneDataMap();
		sceneDataMap
			.keySet()
			.intStream()
			.sorted()
			.forEach(id -> {
				final String data = sceneDataMap.get(id).getScriptData();
				handbookBuilders.forEach(b -> b.append(id + " : " + (data.equals("") ? "Unknown" : data) + "\n"));
			});

		// Quests
		handbookBuilders.forEach(b -> b.append("// Quests\n"));
		final var questDataMap = GameData.getQuestDataMap();
		questDataMap
			.keySet()
			.intStream()
			.sorted()
			.forEach(id -> {
				final QuestData get = questDataMap.get(id);
				if (get == null) {
					return;
				}

				final TextStrings title = textMaps.get((int) mainQuestTitles.get(get.getMainId()));
				final TextStrings desc = textMaps.get((int) get.getDescTextMapHash());

				String data = "";
				String data2 = "";
				for (int i = 0; i < TextStrings.NUM_LANGUAGES; i++) {
					if (desc != null) {
						data = desc.strings[i];
					}
					if (desc != null) {
						data2 = title.strings[i];
					}
					handbookBuilders
						.get(i)
						.append(
							id +
							" : " +
							(data.equals("") ? "Unknown" : data) +
							" - " +
							(data2.equals("") ? "Unknown" : data) +
							"\n"
						);
				}
			});

		// Write txt files
		for (int i = 0; i < TextStrings.NUM_LANGUAGES; i++) {
			File GMHandbookOutputpath = new File("./GM Handbook");
			GMHandbookOutputpath.mkdir();
			final String fileName = "./GM Handbook/GM Handbook - %s.txt".formatted(TextStrings.ARR_LANGUAGES[i]);
			try (
				PrintWriter writer = new PrintWriter(
					new OutputStreamWriter(new FileOutputStream(fileName), StandardCharsets.UTF_8),
					false
				)
			) {
				writer.write(handbookBuilders.get(i).toString());
			}
		}
		Grasscutter.getLogger().info("GM Handbooks generated!");
	}

	public static void createGachaMapping(String location) throws Exception {
		createGachaMappings(location);
	}

	public static List<String> createGachaMappingJsons() {
		final int NUM_LANGUAGES = Language.TextStrings.NUM_LANGUAGES;
		final Language.TextStrings CHARACTER = Language.getTextMapKey(4233146695L); // "Character" in EN
		final Language.TextStrings WEAPON = Language.getTextMapKey(4231343903L); // "Weapon" in EN
		final Language.TextStrings STANDARD_WISH = Language.getTextMapKey(332935371L); // "Standard Wish" in EN
		final Language.TextStrings CHARACTER_EVENT_WISH = Language.getTextMapKey(2272170627L); // "Character Event Wish" in EN
		final Language.TextStrings CHARACTER_EVENT_WISH_2 = Language.getTextMapKey(3352513147L); // "Character Event Wish-2" in EN
		final Language.TextStrings WEAPON_EVENT_WISH = Language.getTextMapKey(2864268523L); // "Weapon Event Wish" in EN
		final List<StringBuilder> sbs = new ArrayList<>(NUM_LANGUAGES);
		for (int langIdx = 0; langIdx < NUM_LANGUAGES; langIdx++) sbs.add(new StringBuilder("{\n")); // Web requests should never need Windows line endings

		// Avatars
		GameData
			.getAvatarDataMap()
			.keySet()
			.intStream()
			.sorted()
			.forEach(id -> {
				AvatarData data = GameData.getAvatarDataMap().get(id);
				int avatarID = data.getId();
				if (avatarID >= 11000000) { // skip test avatar
					return;
				}
				String color =
					switch (data.getQualityType()) {
						case "QUALITY_PURPLE" -> "purple";
						case "QUALITY_ORANGE" -> "yellow";
						case "QUALITY_BLUE" -> "blue";
						default -> "";
					};
				Language.TextStrings avatarName = Language.getTextMapKey(data.getNameTextMapHash());
				for (int langIdx = 0; langIdx < NUM_LANGUAGES; langIdx++) {
					sbs
						.get(langIdx)
						.append("\t\"")
						.append(avatarID % 1000 + 1000)
						.append("\": [\"")
						.append(avatarName.get(langIdx))
						.append(" (")
						.append(CHARACTER.get(langIdx))
						.append(")\", \"")
						.append(color)
						.append("\"],\n");
				}
			});

		// Weapons
		GameData
			.getItemDataMap()
			.keySet()
			.intStream()
			.sorted()
			.forEach(id -> {
				ItemData data = GameData.getItemDataMap().get(id);
				if (data.getId() <= 11101 || data.getId() >= 20000) {
					return; //skip non weapon items
				}
				String color =
					switch (data.getRankLevel()) {
						case 3 -> "blue";
						case 4 -> "purple";
						case 5 -> "yellow";
						default -> null;
					};
				if (color == null) return; // skip unnecessary entries
				Language.TextStrings weaponName = Language.getTextMapKey(data.getNameTextMapHash());
				for (int langIdx = 0; langIdx < NUM_LANGUAGES; langIdx++) {
					sbs
						.get(langIdx)
						.append("\t\"")
						.append(data.getId())
						.append("\": [\"")
						.append(weaponName.get(langIdx).replaceAll("\"", "\\\\\""))
						.append(" (")
						.append(WEAPON.get(langIdx))
						.append(")\", \"")
						.append(color)
						.append("\"],\n");
				}
			});

		for (int langIdx = 0; langIdx < NUM_LANGUAGES; langIdx++) {
			sbs
				.get(langIdx)
				.append("\t\"200\": \"")
				.append(STANDARD_WISH.get(langIdx))
				.append("\",\n\t\"301\": \"")
				.append(CHARACTER_EVENT_WISH.get(langIdx))
				.append("\",\n\t\"400\": \"")
				.append(CHARACTER_EVENT_WISH_2.get(langIdx))
				.append("\",\n\t\"302\": \"")
				.append(WEAPON_EVENT_WISH.get(langIdx))
				.append("\"\n}");
		}
		return sbs.stream().map(StringBuilder::toString).toList();
	}

	public static void createGachaMappings(String location) throws Exception {
		ResourceLoader.loadResources();
		List<String> jsons = createGachaMappingJsons();
		StringBuilder sb = new StringBuilder("mappings = {\n");
		for (int i = 0; i < Language.TextStrings.NUM_LANGUAGES; i++) {
			sb.append("\t\"%s\": ".formatted(Language.TextStrings.ARR_GC_LANGUAGES[i].toLowerCase())); // TODO: change the templates to not use lowercased locale codes
			sb.append(jsons.get(i).replace("\n", "\n\t") + ",\n");
		}
		sb.setLength(sb.length() - 2); // Delete trailing ",\n"
		sb.append("\n}");

		try (
			PrintWriter writer = new PrintWriter(
				new OutputStreamWriter(new FileOutputStream(location), StandardCharsets.UTF_8),
				false
			)
		) {
			// if the user made choices for language, I assume it's okay to assign his/her selected language to "en-us"
			// since it's the fallback language and there will be no difference in the gacha record page.
			// The end-user can still modify the `gacha/mappings.js` directly to enable multilingual for the gacha record system.
			writer.println(sb);
			Grasscutter.getLogger().info("Mappings generated to " + location + " !");
		}
	}

	public static List<String> getAvailableLanguage() {
		File textMapFolder = new File(RESOURCE("TextMap"));
		List<String> availableLangList = new ArrayList<>();
		for (String textMapFileName : Objects.requireNonNull(
			textMapFolder.list((dir, name) -> name.startsWith("TextMap") && name.endsWith(".json"))
		)) {
			availableLangList.add(textMapFileName.replace("TextMap", "").replace(".json", "").toLowerCase());
		}
		return availableLangList;
	}
}
