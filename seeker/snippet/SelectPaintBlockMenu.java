//date: 2021-09-06T17:15:30Z
//url: https://api.github.com/gists/997d203f8a9d2a7271094d513e63d21c
//owner: https://api.github.com/users/BWBJustin

package com.bwbjustin.paintblocks.common.menus;

import java.util.ArrayList;

import com.bwbjustin.paintblocks.core.init.*;

import net.minecraft.network.FriendlyByteBuf;
import net.minecraft.world.SimpleContainer;
import net.minecraft.world.entity.player.*;
import net.minecraft.world.inventory.*;
import net.minecraft.world.item.ItemStack;

public class SelectPaintBlockMenu extends AbstractContainerMenu
{
	public static final ArrayList<ItemStack> INITIAL_ITEMS = new ArrayList<>(
		ItemInit.ITEMS.getEntries().stream().filter(x -> !x.getId().getPath().equals("rainbow_block")).map(x -> x.get().getDefaultInstance()).toList()
	);
	
	public ArrayList<ItemStack> items = new ArrayList<>(INITIAL_ITEMS);
	private SimpleContainer container = new SimpleContainer(144);
	
	public SelectPaintBlockMenu(int id, Inventory inventory, FriendlyByteBuf data)
	{
		super(MenuTypeInit.SELECT_PAINT_BLOCK_MENU_TYPE.get(), id);
		
		addSlots();
	}
	
	public void addSlots()
	{
		for (int i = 0; i < 12; ++i)
		{
			for (int j = 0; j < 12; ++j)
			{
				int slot = i * 12 + j;
				addSlot(new Slot(container, slot, 8 + j * 18, 18 + i * 18));
				container.setItem(slot, slot < items.size() ? items.get(slot) : ItemStack.EMPTY);
			}
		}
	}
	
	@Override
	public boolean stillValid(Player player)
	{
		return container.stillValid(player);
	}
}
