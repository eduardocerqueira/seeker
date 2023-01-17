//date: 2023-01-17T16:54:23Z
//url: https://api.github.com/gists/8014ae066538d3c9e8246386be761b53
//owner: https://api.github.com/users/edevs-damien

    @EventHandler
    public void onPlayerClick(PlayerInteractEvent event) {
        Player player = event.getPlayer();
if(player.getCooldown(Material.STICK) == 0) {
    if (((event.getAction() == Action.RIGHT_CLICK_AIR) || (event.getAction() == Action.RIGHT_CLICK_BLOCK)) && (player.getItemInHand().isSimilar(new ItemStack(Material.STICK)))) {
        player.setVelocity(player.getLocation().getDirection().multiply(0.5).setY(1.2D));
        player.setCooldown(Material.STICK, 20);
        player.getWorld().spawnParticle(Particle.FLAME, player.getLocation(), 100);
        player.getWorld().playSound(player.getLocation(), Sound.ENTITY_FIREWORK_ROCKET_BLAST, 20,20);
        player.getWorld().playSound(player.getLocation(), Sound.ENTITY_FIREWORK_ROCKET_LAUNCH, 20,20);


    }
}




        }


    @EventHandler
    public void onPlayerFall(EntityDamageEvent event) {

        Entity ent = event.getEntity();
        if(ent instanceof Player player)
        {
            if(event.getCause() == EntityDamageEvent.DamageCause.FALL) {
                if(player.getItemInHand().isSimilar(new ItemStack(Material.STICK))) {
                    event.setCancelled(true);
                }
            }
        }

    }