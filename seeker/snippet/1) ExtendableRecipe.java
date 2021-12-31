//date: 2021-12-31T16:29:35Z
//url: https://api.github.com/gists/d0a6f9e89ae59dfa765a27dddec92888
//owner: https://api.github.com/users/Lanse505

package com.teamacronymcoders.essence.api.recipe.infusion;

import com.hrznstudio.titanium.recipe.serializer.SerializableRecipe;
import net.minecraft.core.NonNullList;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.world.Container;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.crafting.Ingredient;
import net.minecraft.world.level.Level;

import java.util.Arrays;

public abstract class ExtendableInfusionRecipe extends SerializableRecipe {

    public Ingredient infusable;
    public Ingredient[] inputIngredients;
    public int duration;

    public ExtendableInfusionRecipe(ResourceLocation id) {
        super(id);
    }

    public ExtendableInfusionRecipe(ResourceLocation id, Ingredient infusable, Ingredient[] inputIngredients, int duration) {
        super(id);
        this.infusable = infusable;
        this.inputIngredients = inputIngredients;
        this.duration = duration;
    }

    @Override
    public boolean matches(Container container, Level level) {
        return false;
    }

    @Override
    public ItemStack assemble(Container container) {
        return ItemStack.EMPTY;
    }

    @Override
    public boolean canCraftInDimensions(int width, int height) {
        return false;
    }

    @Override
    public ItemStack getResultItem() {
        return ItemStack.EMPTY;
    }

    public boolean isValid(NonNullList<ItemStack> stacks) {
        Arrays.stream(inputIngredients).flatMap(ingredient -> Arrays.stream(ingredient.getItems())).map(stack -> stacks.stream().map(stack::sameItem)).anyMatch(bool -> bool.findAny().isPresent() && bool.anyMatch(Boolean::booleanValue));
        return Arrays.stream(inputIngredients).map(list -> Arrays.stream(list.getItems()).map(tagStack -> stacks.stream().map(tagStack::sameItem))).anyMatch(booleanStream -> booleanStream.findAny().isPresent() && booleanStream.findAny().get().anyMatch(Boolean::booleanValue));
    }

    public ItemStack resolveRecipe(ItemStack stack) {
        return stack;
    }
}