//date: 2025-12-30T17:08:59Z
//url: https://api.github.com/gists/3063965073a848ce04b380aab853f89e
//owner: https://api.github.com/users/mworzala

package net.hollowcube.multipart.resourcepack;

import com.google.gson.JsonObject;
import net.kyori.adventure.key.Key;

/// Remaps resource pack keys from server data to the client. This should be used if you remap
/// resource pack entries inside a [ResourcePackWriter] implementation.
///
/// @see ResourcePackWriter#writeItem(Key, JsonObject)
public interface ResourcePackMapper {

    /**
     * A no-op mapper that does not change the key.
     */
    static ResourcePackMapper noop() {
        return (key) -> key;
    }

    Key remapItemKey(Key key);

}