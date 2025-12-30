//date: 2025-12-30T17:08:59Z
//url: https://api.github.com/gists/3063965073a848ce04b380aab853f89e
//owner: https://api.github.com/users/mworzala

package net.hollowcube.multipart.demo;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import net.hollowcube.multipart.resourcepack.ResourcePackWriter;
import net.kyori.adventure.key.Key;
import net.minestom.server.MinecraftServer;
import net.minestom.server.utils.MajorMinorVersion;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public class ZipResourcePackWriter implements ResourcePackWriter, AutoCloseable {
    private final ByteArrayOutputStream output = new ByteArrayOutputStream();
    private final ZipOutputStream zipOutput = new ZipOutputStream(output);
    private boolean closed = false;

    public ZipResourcePackWriter(String packName) throws IOException {
        final JsonObject packMeta = new JsonObject();
        final JsonObject pack = new JsonObject();
        pack.add("min_format", packVersion(MinecraftServer.RESOURCE_PACK_VERSION));
        pack.add("max_format", packVersion(MinecraftServer.RESOURCE_PACK_VERSION));
        pack.addProperty("description", packName);
        packMeta.add("pack", pack);
        writeEntry("pack.mcmeta", packMeta.toString().getBytes(StandardCharsets.UTF_8));
    }

    public byte[] collect() throws IOException {
        if (!closed) zipOutput.close();
        closed = true;
        return output.toByteArray();
    }

    @Override
    public Key writeTexture(Key key, byte[] data) throws IOException {
        final var filePath = "assets/%s/textures/%s".formatted(key.namespace(), key.value());
        writeEntry(filePath, data);
        return key;
    }

    @Override
    public Key writeModel(Key key, JsonObject model) throws IOException {
        final var filePath = "assets/%s/models/%s.json".formatted(key.namespace(), key.value());
        writeEntry(filePath, model.toString().getBytes(StandardCharsets.UTF_8));
        return key;
    }

    @Override
    public Key writeItem(Key key, JsonObject item) throws IOException {
        final var filePath = "assets/%s/items/%s.json".formatted(key.namespace(), key.value());
        writeEntry(filePath, item.toString().getBytes(StandardCharsets.UTF_8));
        return key;
    }

    @Override
    public void close() throws Exception {
        if (!closed) zipOutput.close();
        closed = true;
    }

    private void writeEntry(String path, byte[] data) throws IOException {
        zipOutput.putNextEntry(new ZipEntry(path));
        zipOutput.write(data);
        zipOutput.closeEntry();
    }

    private JsonElement packVersion(MajorMinorVersion version) {
        final var array = new JsonArray();
        array.add(version.major());
        array.add(version.minor());
        return array;
    }
}