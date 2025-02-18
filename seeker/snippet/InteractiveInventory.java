//date: 2025-02-18T16:50:04Z
//url: https://api.github.com/gists/4f17629bb2276dad300e20b1f18f7df5
//owner: https://api.github.com/users/MaSp005

// CHANGE WHEN IMPLEMENTING:
package _;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

import javax.annotation.Nullable;

import org.bukkit.Bukkit;
import org.bukkit.Material;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.inventory.InventoryClickEvent;
import org.bukkit.event.inventory.InventoryCloseEvent;
import org.bukkit.event.inventory.InventoryType;
import org.bukkit.inventory.Inventory;
import org.bukkit.inventory.ItemStack;
import org.bukkit.plugin.java.JavaPlugin;

public class InteractiveInventory {
    public static InteractiveInventoryHandler handler;
    private static JavaPlugin plugin;
    private static int ROW_SIZE = 9;
    protected Inventory inventory;
    private InteractiveItem[] items;
    private InventoryType type;

    public InteractiveInventory(int rows) {
        items = new InteractiveItem[rows * ROW_SIZE];
    }

    /**
     * Opens the generated Inventory for a player.
     * 
     * @param holder The player it should be opened for.
     * @param name   The title of the inventory.
     */
    public void open(Player holder, String name) {
        if (type == null)
            inventory = Bukkit.createInventory(holder, items.length, name);
        else
            inventory = Bukkit.createInventory(holder, type, name);
        for (int i = 0; i < items.length; i++) {
            if (items[i] != null)
                inventory.setItem(i, items[i].toItemStack());
        }
        holder.openInventory(inventory);
        handler.inventories.put(holder, this);
    }

    private void handleClick(InventoryClickEvent event) {
        int slot = event.getRawSlot();
        InteractiveItem item = null;
        if (slot >= 0 && slot < items.length)
            item = items[event.getRawSlot()];
        if (item != null)
            item.handleClick(event);
        else
            event.setCancelled(true);
    }

    /**
     * Sets the item at a slot and starts its modification.
     * 
     * @param slot     The raw slot number for the item to be placed at. 0 = top
     *                 left.
     * @param material The material of the item.
     * @return The generated InteractiveItem to be modified or listened to.
     */
    public InteractiveItem setItem(int slot, Material material) {
        InteractiveItem item = new InteractiveItem(new ItemStack(material), this);
        items[slot] = item;
        return item;
    }

    /**
     * Sets the item at a slot and starts its modification.
     * 
     * @param slot     The raw slot number for the item to be placed at. 0 = top
     *                 left.
     * @param material The item to use.
     * @return The generated InteractiveItem to be modified or listened to.
     */
    public InteractiveItem setItem(int slot, ItemStack material) {
        InteractiveItem item = new InteractiveItem(material, this);
        items[slot] = item;
        return item;
    }

    /**
     * Instantly sets an uninteractible item at a slot, skipping its modification.
     * 
     * @param slot     The raw slot number for the item to be placed at. 0 = top
     *                 left.
     * @param material The item to use.
     * @return The inventory itself.
     */
    public InteractiveInventory setItemInstant(int slot, ItemStack material) {
        items[slot] = new InteractiveItem(material, this);
        return this;
    }

    /**
     * Instantly sets an item at a slot with a specified universal interaction
     * listener, skipping its modification.
     * 
     * @param slot     The raw slot number for the item to be placed at. 0 = top
     *                 left.
     * @param material The item to use.
     * @param listener The event listener of the item.
     * @return The inventory itself.
     */
    public InteractiveInventory setItemInstant(int slot, ItemStack material, Consumer<InventoryClickEvent> listener) {
        items[slot] = new InteractiveItem(material, this);
        items[slot].setUniversalListener(listener);
        return this;
    }

    public static class InteractiveItem {
        private InteractiveInventory inventory;
        private ItemStack itemStack;
        private Consumer<InventoryClickEvent> listener;

        protected InteractiveItem(ItemStack itemStack, InteractiveInventory inventory) {
            this.itemStack = itemStack;
            this.inventory = inventory;
            setUniversalListener(null);
        }

        /**
         * Finishes the item's configuration.
         * 
         * @return The inventory it is associated with.
         */
        public InteractiveInventory finish() {
            return inventory;
        }

        /**
         * Sets this item's universal event listener.
         * 
         * @param listener a universal event listener.
         * @return Itself.
         */
        public InteractiveItem setUniversalListener(@Nullable Consumer<InventoryClickEvent> listener) {
            if (listener == null)
                listener = event -> event.setCancelled(true);
            this.listener = listener;
            return this;
        }

        private void handleClick(InventoryClickEvent event) {
            listener.accept(event);
        }

        protected ItemStack toItemStack() {
            return itemStack;
        }
    }

    public static class InteractiveInventoryHandler implements Listener {
        protected Map<Player, InteractiveInventory> inventories = new HashMap<>();

        public InteractiveInventoryHandler(JavaPlugin pl) {
            plugin = pl;
            Bukkit.getPluginManager().registerEvents(this, plugin);
        }

        @EventHandler
        public void onInventoryClick(InventoryClickEvent event) {
            InteractiveInventory inventory = inventories.get((Player) event.getWhoClicked());
            if (inventory != null)
                inventory.handleClick(event);
        }

        @EventHandler
        public void onInventoryClose(InventoryCloseEvent event) {
            inventories.remove((Player) event.getPlayer());
            // TODO: inventory-specific close handler?
        }
    }
}