//date: 2021-09-06T16:55:40Z
//url: https://api.github.com/gists/4c805b03ccda41577b4d4a99d95205b1
//owner: https://api.github.com/users/BWBJustin

package com.bwbjustin.paintblocks.client.screens;

import java.util.*;

import com.bwbjustin.paintblocks.PaintBlocks;
import com.bwbjustin.paintblocks.common.menus.SelectPaintBlockMenu;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vertex.PoseStack;

import net.minecraft.ChatFormatting;
import net.minecraft.client.gui.components.EditBox;
import net.minecraft.client.gui.screens.inventory.AbstractContainerScreen;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.network.chat.*;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.inventory.*;
import net.minecraft.world.item.*;

public class SelectPaintBlockScreen extends AbstractContainerScreen<SelectPaintBlockMenu>
{
	private EditBox search;
	
	public SelectPaintBlockScreen(SelectPaintBlockMenu menu, Inventory inventory, Component component)
	{
		super(menu, inventory, component);
		imageWidth = 230;
		imageHeight = 250;
	}
	
	@Override
	protected void containerTick()
	{
		super.containerTick();
		search.tick();
	}
	
	@Override
	protected void init()
	{
		super.init();
		minecraft.keyboardHandler.setSendRepeatsToGui(true);
		search = new EditBox(font, leftPos + 103, topPos + 6, 114, 9, new TranslatableComponent("container.paintblocks.select_paint_block"));
		search.setMaxLength(25);
		search.setBordered(false);
		search.setVisible(true);
		search.setCanLoseFocus(false);
		search.setFocus(true);
		search.setValue("");
		search.setTextColor(0xFFFFFF);
		addWidget(search);
		setFocused(search);
	}
	
	@Override
	public void removed()
	{
		super.removed();
		minecraft.keyboardHandler.setSendRepeatsToGui(false);
	}
	
	private void refreshResults()
	{
		menu.items.clear();
		menu.items.addAll(SelectPaintBlockMenu.INITIAL_ITEMS);
		
		if (!search.getValue().isEmpty())
		{
			Iterator<ItemStack> iterator = menu.items.iterator();
			
			while (iterator.hasNext())
			{
				if (!ChatFormatting.stripFormatting(
					iterator.next()
					.getTooltipLines(minecraft.player, minecraft.options.advancedItemTooltips ? TooltipFlag.Default.ADVANCED : TooltipFlag.Default.NORMAL)
					.get(0)
					.getString()
				).toLowerCase().contains(search.getValue().toLowerCase()))
					iterator.remove();
			}
		}
					
		menu.addSlots();
	}
	
	@Override
	public boolean charTyped(char codePoint, int modifiers)
	{
		String previous = search.getValue();
		
		if (search.charTyped(codePoint, modifiers))
		{
			if (!Objects.equals(previous, search.getValue()))
				refreshResults();
			
			return true;
		}
		
		return false;
	}
	
	@Override
	public boolean keyPressed(int keyCode, int scanCode, int modifiers)
	{
		if (keyCode == 256)
			minecraft.player.closeContainer();
		
		if (keyCode == 257 && search.getValue().toLowerCase().equals("reset"))
		{
			PaintBlocks.CURRENT_STATES.put(minecraft.player.getDisplayName().getString(), null);
			minecraft.player.closeContainer();
		}
		
		String previous = search.getValue();
		
		if (search.keyPressed(keyCode, scanCode, modifiers))
		{
			if (!Objects.equals(previous, search.getValue()))
				refreshResults();
			
			return true;
		}
		
		return search.isFocused() && search.isVisible() ? true : super.keyPressed(keyCode, scanCode, modifiers);
	}
	
	@Override
	public void render(PoseStack stack, int mX, int mY, float ticks)
	{
		renderBackground(stack);
		super.render(stack, mX, mY, ticks);
		renderTooltip(stack, mX, mY);
	}

	@Override
	protected void renderBg(PoseStack stack, float ticks, int mX, int mY)
	{
		RenderSystem.setShader(GameRenderer::getPositionTexShader);
		RenderSystem.setShaderColor(1.0F, 1.0F, 1.0F, 1.0F);
		RenderSystem.setShaderTexture(0, new ResourceLocation(PaintBlocks.MOD_ID, "textures/gui/container/select_paint_block.png"));
		
		blit(stack, (width - imageWidth) / 2, (height - imageHeight) / 2, 0, 0, imageWidth, imageHeight);
		search.render(stack, mX, mY, ticks);
	}
	
	@Override
	protected void renderLabels(PoseStack stack, int mX, int mY)
	{
		font.draw(stack, title, titleLabelX, titleLabelY, 4210752);
		font.draw(stack,
			new TranslatableComponent("container.paintblocks.current",
				PaintBlocks.CURRENT_STATES.get(minecraft.player.getDisplayName().getString()) != null ?
				new TranslatableComponent("block.paintblocks."+PaintBlocks.CURRENT_STATES.get(minecraft.player.getDisplayName().getString()).getBlock().getRegistryName().getPath()).getString() :
				"None"
			),
		titleLabelX, titleLabelY + 230, 4210752);
	}
	
	@Override
	protected void slotClicked(Slot slot, int mX, int mY, ClickType type)
	{
		if (slot.hasItem())
		{
			PaintBlocks.CURRENT_STATES.put(minecraft.player.getDisplayName().getString(),((BlockItem)slot.getItem().getItem()).getBlock().defaultBlockState());
			minecraft.player.closeContainer();
		}
	}
}
