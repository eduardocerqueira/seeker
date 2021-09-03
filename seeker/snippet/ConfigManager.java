//date: 2021-09-03T17:15:01Z
//url: https://api.github.com/gists/74e25f7cec27b99c59da5d6c4fff6b70
//owner: https://api.github.com/users/JusticeValley

package com.eebee.fishbuilder.Config;

import com.eebee.fishbuilder.FishBuilder;
import ninja.leaping.configurate.commented.CommentedConfigurationNode;
import ninja.leaping.configurate.hocon.HoconConfigurationLoader;
import ninja.leaping.configurate.loader.ConfigurationLoader;
import org.spongepowered.api.scheduler.Task;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

public class ConfigManager {

    /** Name of the file to grab configuration settings from. */
    private static final String[] FILE_NAMES = {"ability.conf", "evs.conf", "growth.conf", "ivs.conf", "messages.conf", "nature.conf", "pokeball.conf"};
    /** Paths needed to locate the configuration file. */
    private static Path dir;
    private static Path[] config = new Path[FILE_NAMES.length];
    /** Loader for the configuration file. */
    private static ArrayList<ConfigurationLoader<CommentedConfigurationNode>> configLoad = new ArrayList<ConfigurationLoader<CommentedConfigurationNode>>(FILE_NAMES.length);

    /** Storage for all the configuration settings. */
    private static CommentedConfigurationNode[] configNode = new CommentedConfigurationNode[FILE_NAMES.length];

    /**
     * Locates the configuration file and loads it.
     * @param folder Folder where the configuration file is located.
     */
    public static void setup(Path folder){
        dir = folder;
        for (int i = 0; i < FILE_NAMES.length; i++) {
            config[i] = dir.resolve(FILE_NAMES[i]);
        }
        load();
    }

    /**
     * Loads the configuration settings into storage.
     */
    public static void load(){
        //Create directory if it doesn't exist.
        try{
            if (!Files.exists(dir)) {
                Files.createDirectory(dir);
            }

            for (int i = 0; i < FILE_NAMES.length; i++) {
                //Create or locate file and load configuration file into storage.
                FishBuilder.getContainer().getAsset(FILE_NAMES[i]).get().copyToFile(config[i], false, true);

                ConfigurationLoader<CommentedConfigurationNode> tempConfigLoad = HoconConfigurationLoader.builder().setPath(config[i]).build();

                configLoad.add(i, tempConfigLoad);
                configNode[i] = tempConfigLoad.load();
            }

        } catch (IOException e){
            FishBuilder.getLogger().error("FishBuilder configuration could not conf");
            e.printStackTrace();
        }
    }

    /**
     * Saves the configuration settings to configuration file.
     */
    public static void save(){
        Task.builder().execute(() -> {
            for (int i = 0; i < FILE_NAMES.length; i++) {
                try{
                    configLoad.get(i).save(configNode[i]);
                }
                catch(IOException e){
                    FishBuilder.getLogger().error("FishBuilder could not save conf");
                    e.printStackTrace();
                }
            }
        }).async().submit(FishBuilder.instance);
    }

    public static CommentedConfigurationNode getConfigNode(int index, Object... node) {
        return configNode[index].getNode(node);
    }

}