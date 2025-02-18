//date: 2025-02-18T16:50:04Z
//url: https://api.github.com/gists/4f17629bb2276dad300e20b1f18f7df5
//owner: https://api.github.com/users/MaSp005

// Create Interactive Inventory with 6 rows
new InteractiveInventory(3)
  // Sets Item without interaction at 11th slot
  .setItemInstant(11, Material.NAME_TAG).
  // Sets Item at 15th Slot with interaction that prints a message to chat.
  .setItem(9 + 6, Material.NAME_TAG).setUniversalListener(event -> {
    event.setCancelled(true);
    player.sendMessage("I have been activated from an Interactive Inventory!");
  }).finish()
  // Opens the Inventory with a given title
  .open(player, "Inventory Name");