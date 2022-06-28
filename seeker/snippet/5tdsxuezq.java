//date: 2022-06-28T17:02:56Z
//url: https://api.github.com/gists/55624bc6d5c5075c0c4755753a591a9c
//owner: https://api.github.com/users/Cy4Bot

package com.CheeseMan.swordsplus.core.init;


import java.util.Map;

import com.CheeseMan.swordsplus.common.recipe.ExcimerLaserRecipe;
import com.CheeseMan.swordsplus.common.recipe.ExcimerLaserRecipeType;

import net.minecraft.item.crafting.IRecipe;
import net.minecraft.item.crafting.IRecipeSerializer;
import net.minecraft.item.crafting.IRecipeType;
import net.minecraft.item.crafting.RecipeManager;
import net.minecraft.util.ResourceLocation;
import net.minecraft.util.registry.Registry;
import net.minecraftforge.event.RegistryEvent.Register;
import net.minecraftforge.fml.common.ObfuscationReflectionHelper;

public class RecipeInit {
	public static final IRecipeType<ExcimerLaserRecipe> EXCIMER_LASER_RECIPE = new ExcimerLaserRecipeType();
	
	public static void registerRecipes(Register<IRecipeSerializer<?>> event) {
		registerRecipe(event, EXCIMER_LASER_RECIPE, ExcimerLaserRecipe.SERIALIZER);
	}
	
	private static void registerRecipe(Register<IRecipeSerializer<?>> event, IRecipeType<?> type,
			IRecipeSerializer<?> serializer) {
		Registry.register(Registry.RECIPE_TYPE, new ResourceLocation(type.toString()), type);
		event.getRegistry().register(serializer);
	}
	
	public static Map<ResourceLocation, IRecipe<?>> getRecipes(IRecipeType<?> type, RecipeManager manager) {
		final Map<IRecipeType<?>, Map<ResourceLocation, IRecipe<?>>> recipes = ObfuscationReflectionHelper
				.getPrivateValue(RecipeManager.class, manager, "field_199522_d");
		
		return recipes.get(type);
	}
}
