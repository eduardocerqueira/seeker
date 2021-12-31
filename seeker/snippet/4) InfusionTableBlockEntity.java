//date: 2021-12-31T16:29:35Z
//url: https://api.github.com/gists/d0a6f9e89ae59dfa765a27dddec92888
//owner: https://api.github.com/users/Lanse505

package com.teamacronymcoders.essence.common.block.infusion.tile;

import com.hrznstudio.titanium.annotation.Save;
import com.hrznstudio.titanium.block.tile.ActiveTile;
import com.hrznstudio.titanium.component.inventory.InventoryComponent;
import com.teamacronymcoders.essence.Essence;
import com.teamacronymcoders.essence.api.recipe.infusion.ExtendableInfusionRecipe;
import com.teamacronymcoders.essence.common.block.infusion.InfusionPedestalBlock;
import com.teamacronymcoders.essence.common.item.tome.TomeOfKnowledgeItem;
import com.teamacronymcoders.essence.common.util.helper.EssenceWorldHelper;
import com.teamacronymcoders.essence.compat.registrate.EssenceBlockRegistrate;
import net.minecraft.core.BlockPos;
import net.minecraft.core.NonNullList;
import net.minecraft.util.Mth;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.state.BlockState;

import javax.annotation.Nonnull;

public class InfusionTableBlockEntity extends ActiveTile<InfusionTableBlockEntity> {

    private static final BlockPos[] pedestal_positions = new BlockPos[]{
            new BlockPos(-4, 0, 0),
            new BlockPos(+4, 0, 0),
            new BlockPos(0, 0, +4),
            new BlockPos(0, 0, -4),
            new BlockPos(+3, 0, -3),
            new BlockPos(+3, 0, +3),
            new BlockPos(-3, 0, +3),
            new BlockPos(-3, 0, -3)
    };

    private ExtendableInfusionRecipe recipe;

    @Save
    private Boolean shouldBeWorking = false;
    @Save
    private Boolean isWorking = false;
    @Save
    private Integer workDuration = 0;
    @Save
    private Integer totalWorkDuration = 0;
    @Save
    private Boolean hasFiredSound = false;
    @Save
    private Integer ticksExisted = 0;

    @Save
    private final InventoryComponent<InfusionTableBlockEntity> infusable;
    @Save
    private final InventoryComponent<InfusionTableBlockEntity> tome;

    // Book Rendering Variables
    public int ticks;
    public float flip;
    public float oldFlip;
    public float flipT;
    public float flipA;
    public float open;
    public float oldOpen;
    public float rot;
    public float oldRot;
    public float tRot;

    @Save
    public Long pageSoundLastPlayed = 0L;

    public InfusionTableBlockEntity(BlockPos pos, BlockState state) {
        super(EssenceBlockRegistrate.INFUSION_TABLE.get(), pos, state);
        addInventory(infusable = new InventoryComponent<InfusionTableBlockEntity>("input", 80, 20, 1)
                .setComponentHarness(this)
                .setOutputFilter(this::canExtractInfusable)
                .setSlotLimit(1)
        );
        addInventory(tome = new InventoryComponent<InfusionTableBlockEntity>("tome", 9, 10, 1)
                .setComponentHarness(this)
                .setOnSlotChanged((stack, integer) -> markComponentForUpdate(false))
                .setInputFilter((stack, integer) -> false)
                .setOutputFilter((stack, integer) -> false)
                .setSlotLimit(1)
        );
    }

    @Override
    public void clientTick(Level level, BlockPos pos, BlockState state, InfusionTableBlockEntity blockEntity) {
        super.clientTick(level, pos, state, blockEntity);
        ticksExisted++;
        blockEntity.oldOpen = blockEntity.open;
        blockEntity.oldRot = blockEntity.rot;
        Player player = level.getNearestPlayer((double)pos.getX() + 0.5D, (double)pos.getY() + 0.5D, (double)pos.getZ() + 0.5D, 3.0D, false);
        if (player != null) {
            double d0 = player.getX() - ((double)pos.getX() + 0.5D);
            double d1 = player.getZ() - ((double)pos.getZ() + 0.5D);
            blockEntity.tRot = (float) Mth.atan2(d1, d0);
            blockEntity.open += 0.1F;
            if (blockEntity.open < 0.5F || Essence.RANDOM.nextInt(40) == 0) {
                float f1 = blockEntity.flipT;

                do {
                    blockEntity.flipT += (float)(Essence.RANDOM.nextInt(4) - Essence.RANDOM.nextInt(4));
                } while(f1 == blockEntity.flipT);
            }
        } else {
            blockEntity.tRot += 0.02F;
            blockEntity.open -= 0.1F;
        }

        while(blockEntity.rot >= (float)Math.PI) {
            blockEntity.rot -= ((float)Math.PI * 2F);
        }

        while(blockEntity.rot < -(float)Math.PI) {
            blockEntity.rot += ((float)Math.PI * 2F);
        }

        while(blockEntity.tRot >= (float)Math.PI) {
            blockEntity.tRot -= ((float)Math.PI * 2F);
        }

        while(blockEntity.tRot < -(float)Math.PI) {
            blockEntity.tRot += ((float)Math.PI * 2F);
        }

        float f2;
        for(f2 = blockEntity.tRot - blockEntity.rot; f2 >= (float)Math.PI; f2 -= ((float)Math.PI * 2F)) {
        }

        while(f2 < -(float)Math.PI) {
            f2 += ((float)Math.PI * 2F);
        }

        blockEntity.rot += f2 * 0.4F;
        blockEntity.open = Mth.clamp(blockEntity.open, 0.0F, 1.0F);
        ++blockEntity.ticks;
        blockEntity.oldFlip = blockEntity.flip;
        float f = (blockEntity.flipT - blockEntity.flip) * 0.4F;
        float f3 = 0.2F;
        f = Mth.clamp(f, -0.2F, 0.2F);
        blockEntity.flipA += (f - blockEntity.flipA) * 0.9F;
        blockEntity.flip += blockEntity.flipA;
    }

    @Override
    public void serverTick(Level level, BlockPos pos, BlockState state, InfusionTableBlockEntity blockEntity) {
        super.serverTick(level, pos, state, blockEntity);
        NonNullList<ItemStack> stacks = getPedestalStacks();
        if (shouldBeWorking && recipe == null) getInfusionRecipe(stacks);
        if (shouldBeWorking || isWorking) {
            if (!recipe.isValid(getPedestalStacks())) {
                totalWorkDuration = recipe.duration;
            }
            isWorking = true;
            if (workDuration >= totalWorkDuration) {
                ItemStack infusable = this.infusable.getStackInSlot(0);
                recipe.resolveRecipe(infusable);
                shouldBeWorking = false;
                isWorking = false;
                hasFiredSound = false;
            }
            if (isWorking) Essence.LOGGER.info("I'M WORKING");
            if (isWorking && !hasFiredSound) {
                EssenceWorldHelper.playInfusionSound(this, true);
                hasFiredSound = true;
            }
        }
    }

    @Nonnull
    @Override
    public InfusionTableBlockEntity getSelf() {
        return this;
    }

    private boolean canExtractInfusable(ItemStack stack, int slot) {
        return !isWorking;
    }

    private NonNullList<ItemStack> getPedestalStacks() {
        BlockPos tablePosition = getBlockPos();
        NonNullList<ItemStack> stacks = NonNullList.create();
        if (getLevel() != null) {
            for (BlockPos pos : pedestal_positions) {
                BlockPos pedestalPosition = tablePosition.offset(pos.getX(), pos.getY(), pos.getZ());
                if (getLevel().getBlockState(pedestalPosition).getBlock() instanceof InfusionPedestalBlock && getLevel().getBlockEntity(pedestalPosition) instanceof InfusionPedestalBlockEntity) {
                    InfusionPedestalBlockEntity pedestal = (InfusionPedestalBlockEntity) getLevel().getBlockEntity(pos);
                    if (pedestal != null) {
                        stacks.add(pedestal.getStack());
                    }
                }
            }
        }
        return stacks;
    }

    private void getInfusionRecipe(NonNullList<ItemStack> stacks) {
        if (getLevel() != null) {
            recipe = getLevel().getRecipeManager().getRecipes()
                    .stream()
                    .filter(iRecipe -> iRecipe instanceof ExtendableInfusionRecipe)
                    .map(iRecipe -> (ExtendableInfusionRecipe) iRecipe)
                    .filter(recipes -> recipes.isValid(stacks))
                    .findFirst().orElse(null);
        }
    }

    public Boolean getWorking() {
        return isWorking;
    }

    public InventoryComponent<InfusionTableBlockEntity> getTome() {
        return tome;
    }

    public InventoryComponent<InfusionTableBlockEntity> getInfusable() {
        return infusable;
    }

    public void setShouldBeWorking(Boolean shouldBeWorking) {
        this.shouldBeWorking = shouldBeWorking;
    }

    public boolean hasTome() {
        return !tome.getStackInSlot(0).isEmpty() && tome.getStackInSlot(0).getItem() instanceof TomeOfKnowledgeItem;
    }

    public Integer getTicksExisted() {
        return ticksExisted;
    }

    public Long getPageSoundLastPlayed() {
        return pageSoundLastPlayed;
    }

}