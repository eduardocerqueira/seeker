//date: 2023-12-27T17:05:28Z
//url: https://api.github.com/gists/a202c1f2e222ad37d141c1ab999ab54a
//owner: https://api.github.com/users/DDX5

package org.multicoder.mcpaintball.common.items.weapons;

import net.minecraft.nbt.CompoundTag;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.InteractionResultHolder;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.entity.projectile.AbstractArrow;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.Level;
import org.multicoder.mcpaintball.MCPaintball;
import org.multicoder.mcpaintball.common.utility.Teams;

public class PistolItem extends Item
{
    public PistolItem()
    {
        super(new Properties().setNoRepair().stacksTo(1));
    }

    @Override
    public InteractionResultHolder<ItemStack> use(Level level, Player player, InteractionHand hand)
    {
        if(!level.isClientSide())
        {
            CompoundTag playerData = player.getPersistentData();
            if(playerData.contains("mcpaintball"))
            {
                CompoundTag paintball = playerData.getCompound("mcpaintball");
                if(paintball.contains("team"))
                {
                    Teams team = Teams.values()[paintball.getInt("team")];
                    AbstractArrow paintball_arrow = team.getPaintball(level);
                    paintball_arrow.shootFromRotation(player,player.getXRot(),player.getYRot(),0f,3f,0f);
                    level.addFreshEntity(paintball_arrow);
                    player.getCooldowns().addCooldown(this,40);
                }
            }
        }
        return super.use(level, player, hand);
    }
}
