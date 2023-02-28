//date: 2023-02-28T17:01:27Z
//url: https://api.github.com/gists/b175b442c7ef0722422735093fd64f18
//owner: https://api.github.com/users/DDX5

package org.multicoder.mcpaintball.entity;

import net.minecraft.core.BlockPos;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.projectile.ThrowableItemProjectile;
import net.minecraft.world.item.Item;
import net.minecraft.world.level.Level;
import net.minecraft.world.phys.HitResult;
import org.multicoder.mcpaintball.init.entityinit;
import org.multicoder.mcpaintball.init.iteminit;

public class RedGrenade extends ThrowableItemProjectile
{

    public RedGrenade( double pX, double pY, double pZ, Level pLevel)
    {
        super((EntityType<? extends ThrowableItemProjectile>) entityinit.RED_GRENADE.get(), pX, pY, pZ, pLevel);
    }

    public RedGrenade(EntityType<? extends ThrowableItemProjectile> pEntityType, Level pLevel)
    {
        super(pEntityType, pLevel);
    }

    public RedGrenade(LivingEntity pShooter, Level pLevel)
    {
        super((EntityType<? extends ThrowableItemProjectile>) entityinit.RED_GRENADE.get(), pShooter, pLevel);
    }

    @Override
    protected void onHit(HitResult pResult)
    {
        BlockPos pos = new BlockPos(pResult.getLocation());
        getLevel().explode(this,pos.getX(),pos.getY(),pos.getZ(),3, Level.ExplosionInteraction.TNT);
        super.onHit(pResult);
    }

    @Override
    protected Item getDefaultItem()
    {
        return iteminit.RED_GRENADE.get();
    }
}