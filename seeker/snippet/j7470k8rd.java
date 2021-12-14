//date: 2021-12-14T17:09:24Z
//url: https://api.github.com/gists/db19476051c1c68efa327ee4cc88238a
//owner: https://api.github.com/users/Cy4Bot

package com.example.Inits;

import com.example.examplemod.CryIndustry;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import net.minecraft.entity.Entity;
import net.minecraft.entity.merchant.villager.VillagerProfession;
import net.minecraft.entity.merchant.villager.VillagerTrades;
import net.minecraft.item.*;
import net.minecraft.util.SoundEvents;
import net.minecraft.village.PointOfInterestType;
import net.minecraftforge.fml.RegistryObject;
import net.minecraftforge.fml.common.ObfuscationReflectionHelper;
import net.minecraftforge.registries.DeferredRegister;
import net.minecraftforge.registries.ForgeRegistries;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;


public class VillagerInit {
    public static final DeferredRegister<PointOfInterestType> POI_TYPES = DeferredRegister.create(ForgeRegistries.POI_TYPES, CryIndustry.MOD_ID);
    public static final DeferredRegister<VillagerProfession> PROFESSIONS = DeferredRegister.create(ForgeRegistries.PROFESSIONS, CryIndustry.MOD_ID);

    public static final RegistryObject<PointOfInterestType> EXAMPLE_POI = POI_TYPES.register("example",
            () -> new PointOfInterestType("example", PointOfInterestType.getBlockStates(BlockInit.REACTOR_FRAME.get()), 1, 1));
    public static final RegistryObject<VillagerProfession> EXAMPLE_PROFESSION = PROFESSIONS.register("example",
            () -> new VillagerProfession("example", EXAMPLE_POI.get(), ImmutableSet.of(ItemInit.SMART_UPGRADE.get()), ImmutableSet.of(), SoundEvents.VILLAGER_WORK_LEATHERWORKER));

    public static void registerPointOfInterests() {
        try {
            ObfuscationReflectionHelper.findMethod(PointOfInterestType.class, "registerBlockStates", PointOfInterestType.class).invoke(null, EXAMPLE_POI.get());
        } catch (IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }

    public static class MultiItemTradeEmeraldsForItem implements VillagerTrades.ITrade {
        private final int emeraldCount;
        private final int sellingItemCount;
        private final int maxUses;
        private final int xpValue;
        private final float priceMultiplier;
        private Item selling;
        private final List<Item> items;

        public MultiItemTradeEmeraldsForItem(List<Item> items, int maxUsesIn, int sellingItemCount, int xpValueIn, int emeraldCount) {
            this.maxUses = maxUsesIn;
            this.sellingItemCount = sellingItemCount;
            this.xpValue = xpValueIn;
            this.emeraldCount = emeraldCount;
            this.priceMultiplier = 0.05F;
            this.items = items;
        }

        @Nullable
        @Override
        public MerchantOffer getOffer(@Nonnull Entity trader, @Nonnull Random rand) {
            selling = items.get((int) (Math.random() * items.size()));
            return new MerchantOffer(new ItemStack(Items.EMERALD, this.emeraldCount), new ItemStack(selling, this.sellingItemCount), this.maxUses, this.xpValue, this.priceMultiplier);
        }
    }


    public static class MultiItemTradeEmeraldsForItemWeighted implements VillagerTrades.ITrade {
        private final int emeraldCount;
        private final int sellingItemCount;
        private final int maxUses;
        private final int xpValue;
        private final float priceMultiplier;
        private Item selling;
        private final Map<Item, Integer> items;

        public MultiItemTradeEmeraldsForItemWeighted(Map<Item, Integer> items, int maxUsesIn, int buyItemCount, int xpValueIn, int emeraldCount) {
            this.maxUses = maxUsesIn;
            this.sellingItemCount = buyItemCount;
            this.xpValue = xpValueIn;
            this.emeraldCount = emeraldCount;
            this.priceMultiplier = 0.05F;
            this.items = items;
        }

        @Nullable
        @Override
        public MerchantOffer getOffer(@Nonnull Entity trader, @Nonnull Random rand) {
            selling = getItemByWeigth(items);
            return new MerchantOffer(new ItemStack(Items.EMERALD, this.emeraldCount), new ItemStack(selling, this.sellingItemCount), this.maxUses, this.xpValue, this.priceMultiplier);
        }

        private Item getItemByWeigth(Map<Item, Integer> itemIntegerHashMap) {
            AtomicInteger totalVal = new AtomicInteger();
            itemIntegerHashMap.values().parallelStream().reduce((id, val) -> {
                id += val;
                return id;
            }).ifPresent(totalVal::set);
            int choose = (int) (Math.random() * totalVal.get());
            int currentIndex = 0;
            for (Map.Entry<Item, Integer> entry : items.entrySet()) {
                currentIndex += entry.getValue();
                if (choose < currentIndex) {
                    return entry.getKey();
                }
            }
            throw new IllegalArgumentException("Supplied Empty HashMap to Trade");
        }
    }

    private static Int2ObjectMap<VillagerTrades.ITrade[]> gatAsIntMap(ImmutableMap<Integer, VillagerTrades.ITrade[]> p_221238_0_) {
        return new Int2ObjectOpenHashMap(p_221238_0_);
    }

    public static void populateTrades() {
        MultiItemTradeEmeraldsForItem trade = new MultiItemTradeEmeraldsForItem(ImmutableList.of(Items.ACACIA_BOAT, Items.BIRCH_BOAT, Items.OAK_BOAT, Items.DARK_OAK_BOAT), 5, 1, 10, 40);
        MultiItemTradeEmeraldsForItemWeighted tradeWeighted = new MultiItemTradeEmeraldsForItemWeighted(ImmutableMap.of(Items.ACACIA_BOAT,20, Items.BIRCH_BOAT,10, Items.OAK_BOAT,10, Items.DARK_OAK_BOAT,60),5, 1, 500, 1);
        VillagerTrades.ITrade[] level_1 = new VillagerTrades.ITrade[]{tradeWeighted};
        VillagerTrades.ITrade[] level_2 = new VillagerTrades.ITrade[]{tradeWeighted};
        VillagerTrades.ITrade[] level_3 = new VillagerTrades.ITrade[]{tradeWeighted};
        VillagerTrades.ITrade[] level_4 = new VillagerTrades.ITrade[]{tradeWeighted};
        VillagerTrades.ITrade[] level_5 = new VillagerTrades.ITrade[]{tradeWeighted};
        VillagerTrades.TRADES.put(VillagerInit.EXAMPLE_PROFESSION.get(),
                gatAsIntMap(ImmutableMap.of(1, level_1, 2, level_2, 3, level_3, 4, level_4, 5, level_5)));
    }
}


