//date: 2023-02-28T17:01:27Z
//url: https://api.github.com/gists/b175b442c7ef0722422735093fd64f18
//owner: https://api.github.com/users/DDX5

package org.multicoder.mcpaintball.entity.renderer;

import net.minecraft.client.renderer.entity.EntityRendererProvider;
import net.minecraft.client.renderer.entity.ThrownItemRenderer;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.world.entity.Entity;
import org.multicoder.mcpaintball.entity.RedGrenade;

public class RedGrenadeRenderer extends ThrownItemRenderer<RedGrenade>
{
    public static final ResourceLocation TEXTURE = new ResourceLocation("mcpaintball:textures/item/utility/grenades/red.png");
    public RedGrenadeRenderer(EntityRendererProvider.Context pContext)
    {
        super(pContext);
    }

    @Override
    public ResourceLocation getTextureLocation(Entity pEntity)
    {
        return TEXTURE;
    }
}
