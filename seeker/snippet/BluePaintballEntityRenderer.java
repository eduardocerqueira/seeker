//date: 2023-12-27T17:05:28Z
//url: https://api.github.com/gists/a202c1f2e222ad37d141c1ab999ab54a
//owner: https://api.github.com/users/DDX5

public class BluePaintballEntityRenderer extends ArrowRenderer<BluePaintballEntity>
{
    public static final ResourceLocation BLUE = new ResourceLocation("mcpaintball:textures/entity/projectiles/paintball/blue_paintball.png");

    public BluePaintballEntityRenderer(EntityRendererProvider.Context p_173917_)
    {
        super(p_173917_);
    }

    @Override
    public ResourceLocation getTextureLocation(BluePaintballEntity p_114482_) {
        return BLUE;
    }
}