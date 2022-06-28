//date: 2022-06-28T17:00:40Z
//url: https://api.github.com/gists/f693bd02d856f5177fb7661a4dee568e
//owner: https://api.github.com/users/Cy4Bot

package com.CheeseMan.swordsplus.common.recipe;


import com.CheeseMan.swordsplus.SwordsPlus;
import com.CheeseMan.swordsplus.common.te.ExcimerLaserTileEntity;
import com.CheeseMan.swordsplus.core.init.ItemInit;
import com.CheeseMan.swordsplus.core.init.RecipeInit;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import net.minecraft.item.ItemStack;
import net.minecraft.item.crafting.IRecipe;
import net.minecraft.item.crafting.IRecipeSerializer;
import net.minecraft.item.crafting.IRecipeType;
import net.minecraft.item.crafting.Ingredient;
import net.minecraft.item.crafting.ShapedRecipe;
import net.minecraft.network.PacketBuffer;
import net.minecraft.util.JSONUtils;
import net.minecraft.util.ResourceLocation;
import net.minecraft.world.World;
import net.minecraftforge.registries.ForgeRegistryEntry;


public class ExcimerLaserRecipe implements IRecipe<ExcimerLaserTileEntity>{
	
	public static final Serializer SERIALIZER = new Serializer();

	private final Ingredient input;
	private final Ingredient input1;
	private final ItemStack output;
	private final ResourceLocation id;
	
	public ExcimerLaserRecipe(Ingredient input, Ingredient input1, ItemStack output, ResourceLocation id) {
		this.input = input;
		this.input1 = input1;
		this.output = output;
		this.id = id;
	}
	@Override
	public boolean matches(ExcimerLaserTileEntity inv, World world) {
		if (inv.getContainerSize() == 4) {
			return testInputs(inv, input) && testInputs(inv, input1);
		}
		return false;
	}
	@Override
	public ItemStack assemble(ExcimerLaserTileEntity p_77572_1_) {
		return output.copy();
	}

	@Override
	public ItemStack getResultItem() {
		return this.output;
	}

	@Override
	public ResourceLocation getId() {
		return this.id;
	}

	@Override
	public IRecipeSerializer<?> getSerializer() {
		return SERIALIZER;
	}

	@Override
	public IRecipeType<?> getType() {
		return RecipeInit.EXCIMER_LASER_RECIPE;
	}
	@Override
	public ItemStack getToastSymbol() {
		return new ItemStack(ItemInit.BATTERY.get());
	}

	@Override
	public boolean canCraftInDimensions(int width, int height) {
		return true;
	}
	
	private boolean testInputs(ExcimerLaserTileEntity tileEntity, Ingredient input) {
		for (int i = 0; i < 2; i++) {
			if (input.test(tileEntity.getItem(i)))
				return true;
		}
		return false;
	}


	private static class Serializer extends ForgeRegistryEntry<IRecipeSerializer<?>>
			implements IRecipeSerializer<ExcimerLaserRecipe> {
		Serializer() {
			this.setRegistryName(SwordsPlus.MOD_ID, "excimer_laser_recipe");
		}

		@Override
		public ExcimerLaserRecipe fromJson(ResourceLocation recipeId, JsonObject json) {
			final JsonElement inputEL = JSONUtils.isArrayNode(json, "input") ? JSONUtils.getAsJsonArray(json, "input")
					: JSONUtils.getAsJsonObject(json, "input");
			final JsonElement inputEL1 = JSONUtils.isArrayNode(json, "input1")
					? JSONUtils.getAsJsonArray(json, "input1")
					: JSONUtils.getAsJsonObject(json, "input1");
			final Ingredient input = Ingredient.fromJson(inputEL);
			final Ingredient input1 = Ingredient.fromJson(inputEL1);

			final ItemStack output = ShapedRecipe.itemFromJson(JSONUtils.getAsJsonObject(json, "output"));

			return new ExcimerLaserRecipe(input, input1, output, recipeId);
		}

		@Override
		public ExcimerLaserRecipe fromNetwork(ResourceLocation recipeId, PacketBuffer buffer) {
			final Ingredient input = Ingredient.fromNetwork(buffer);
			final Ingredient input1 = Ingredient.fromNetwork(buffer);
			final ItemStack output = buffer.readItem();

			return new ExcimerLaserRecipe(input, input1, output, recipeId);
		}

		@Override
		public void toNetwork(PacketBuffer buffer, ExcimerLaserRecipe recipe) {
			recipe.input.toNetwork(buffer);
			recipe.input1.toNetwork(buffer);
			buffer.writeItem(recipe.output);
		}

	}
}