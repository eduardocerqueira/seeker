//date: 2025-12-30T17:08:59Z
//url: https://api.github.com/gists/3063965073a848ce04b380aab853f89e
//owner: https://api.github.com/users/mworzala

package net.hollowcube.multipart.resourcepack;

import com.google.gson.JsonObject;
import net.kyori.adventure.key.Key;

import java.io.IOException;
import java.nio.file.Path;

/// Interface for writing resources into a resource pack-like structure.
///
/// May be implemented to do key remapping or any other required processing.
///
/// @see #direct(Path) direct(Path) for a reference implementation that writes to a resource pack directly.
public interface ResourcePackWriter {

    /// Creates a writer that creates files in a vanilla resource pack structure.
    ///
    /// This is the default writer implementation and does not require a [ResourcePackMapper].
    ///
    /// @param resourcePackRoot The root directory of the resource pack, where the `assets` directory is located.
    /// @return A new [ResourcePackWriter] that writes directly to the specified path.
    static ResourcePackWriter direct(Path resourcePackRoot) {
        return new DirectResourcePackWriter(resourcePackRoot);
    }

    /// Creates a new texture in the resource pack with the following content.
    ///
    /// It is valid to return a different key which will be used to reference this texture in other data.
    ///
    /// @param key  The key for the texture, eg `my_server:item/my_model/my_texture.png`
    /// @param data The data for the texture
    /// @return The key for the texture, which may be different from the input key if
    /// the resource pack has a different naming scheme.
    Key writeTexture(Key key, byte[] data) throws IOException;

    /// Create a new model in the resource pack with the following content.
    ///
    /// It is valid to return a different key which will be used to reference this model in other data.
    ///
    /// @param key   The key for the model, eg `my_server:item/my_model/my_model.json`
    /// @param model The model data in JSON format.
    /// @return The key for the model, which may be different from the input key if
    /// the resource pack has a different naming scheme.
    Key writeModel(Key key, JsonObject model) throws IOException;

    /// Create a new item model in the resource pack with the following content.
    ///
    /// Remapping the key for this model is OK, however you must keep track of the mapping
    /// of key into this function to the resulting key of the model on the client. Multipart
    /// will request the remapped key via a [ResourcePackMapper] at runtime.
    ///
    /// @param key  The key for the item model, eg `my_server:item/my_model/my_item.json`
    /// @param item The item model data in JSON format.
    /// @return The key for the model, which may be different from the input key if
    /// the resource pack has a different naming scheme.
    /// @see ResourcePackMapper
    Key writeItem(Key key, JsonObject item) throws IOException;

}