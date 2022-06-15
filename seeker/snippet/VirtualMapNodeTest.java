//date: 2022-06-15T17:15:47Z
//url: https://api.github.com/gists/15376129dce8393c2be7ce41d44326a4
//owner: https://api.github.com/users/Konicai

package dev.projectg.crossplatforms;

import io.leangen.geantyref.TypeToken;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.spongepowered.configurate.ConfigurateException;
import org.spongepowered.configurate.ConfigurationNode;
import org.spongepowered.configurate.serialize.SerializationException;
import org.spongepowered.configurate.yaml.YamlConfigurationLoader;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class VirtualMapNodeTest {

    @TempDir
    private static File TEMP_DIR;
    private static YamlConfigurationLoader LOADER;
    private ConfigurationNode map;

    @BeforeAll
    public static void setup() {
        LOADER = YamlConfigurationLoader.builder().file(new File(TEMP_DIR, "config.yml")).build();
    }

    @BeforeEach
    public void setBlankMap() throws ConfigurateException {
        map = LOADER.load();
    }

    // All below this fails

    @Test
    public void testIntegerKey() throws SerializationException {
        map.node(1, "name").set("John");
    }

    @Test
    public void testIntegerKey2() throws SerializationException {
        map.node(2).set("Summer");
    }

    // All below this passes

    @Test
    public void testShortKey() throws SerializationException {
        map.node((short) 5, "name").set("John");
    }
    
    @Test
    public void testLongKey() throws SerializationException {
        map.node(6L, "name").set("John");
    }

    @Test
    public void testRealIntegerKey() throws ConfigurateException {
        map.set(new HashMap<Integer, ConfigurationNode>());
        testIntegerKey();
    }

    @Test
    public void testRealIntegerKey2() throws ConfigurateException{
        map.set(new TypeToken<Map<Integer, ConfigurationNode>>() {}, Collections.emptyMap());
        testIntegerKey();
    }

    @Test
    public void testRealIntegerKey3() throws ConfigurateException, MalformedURLException {
        map.set(new URL("https://github.com"));
        testIntegerKey();
    }

    @Test
    public void testStringKey() throws ConfigurateException {
        map.node("1", "name").set("John");
    }
}
