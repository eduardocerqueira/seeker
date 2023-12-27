//date: 2023-12-27T17:05:28Z
//url: https://api.github.com/gists/a202c1f2e222ad37d141c1ab999ab54a
//owner: https://api.github.com/users/DDX5

public class BluePaintballEntity extends AbstractArrow
{
    public BluePaintballEntity(EntityType<?> p_36721_, Level p_36722_) {
        super((EntityType<? extends AbstractArrow>) p_36721_, p_36722_);
    }

    public BluePaintballEntity(EntityType<? extends AbstractArrow> p_36711_, double p_36712_, double p_36713_, double p_36714_, Level p_36715_)
    {
        super(p_36711_, p_36712_, p_36713_, p_36714_, p_36715_);
    }

    public BluePaintballEntity(EntityType<? extends AbstractArrow> p_36717_, LivingEntity p_36718_, Level p_36719_)
    {
        super(p_36717_, p_36718_, p_36719_);
    }

    @Override
    protected void onHitEntity(EntityHitResult hitResult)
    {
        super.onHitEntity(hitResult);
    }

    @Override
    protected ItemStack getPickupItem()
    {
        return null;
    }
}