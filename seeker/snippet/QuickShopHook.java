//date: 2022-05-03T17:21:07Z
//url: https://api.github.com/gists/86db5b70cd16da4122c60984987d3616
//owner: https://api.github.com/users/SrBedrock

package com.armamc.armatransfer.hook;

import com.armamc.armatransfer.ArmaTransfer;
import org.bukkit.Bukkit;
import org.bukkit.plugin.PluginManager;
import org.maxgamer.quickshop.api.QuickShopAPI;
import org.maxgamer.quickshop.api.ShopAPI;
import org.maxgamer.quickshop.shop.Shop;

import java.util.Collections;
import java.util.List;
import java.util.UUID;

public final class QuickShopHook extends Hook {

    private boolean enabled = false;

    public QuickShopHook(ArmaTransfer plugin) {
        super(plugin);
    }

    @Override
    public void register() {
        PluginManager pm = Bukkit.getPluginManager();

        if (pm.isPluginEnabled("QuickShop")) {
            this.enabled = true;
            this.plugin.getLogger().info("Hooked into QuickShop!");
        } else {
            this.plugin.getLogger().info("Couldn't hook to QuickShop");
        }
    }

    @Override
    public boolean isEnabled() {
        return this.enabled;
    }

    public List<Shop> getShops(UUID uuid) {
        ShopAPI shopAPI = QuickShopAPI.getShopAPI();

        return shopAPI == null ? Collections.emptyList() : shopAPI.getShops(uuid);
    }
}