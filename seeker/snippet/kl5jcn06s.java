//date: 2021-12-14T17:01:41Z
//url: https://api.github.com/gists/19c348416b3feaa2987c88e36f4c9240
//owner: https://api.github.com/users/Cy4Bot

package com.example.Util.MultiBlocks.StructureChecks;

import com.example.Util.MultiBlocks.MultiBlockData;
import net.minecraft.block.Block;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountedCompleter;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

public class StructureCheckCompute extends CountedCompleter<java.lang.Boolean> {
    private enum ECalculationStep {
        TOP,
        BOTTOM,
        FRAME,
        SIDE,
        INSIDE;
    }

    private final MultiBlockData data;
    private final BlockPos corner;
    private final Block block;
    private final World world;
    private final AtomicBoolean valid;
    private final CountedCompleter<java.lang.Boolean> parent;
    private final ECalculationStep step;
    private final BlockPos size;

    public StructureCheckCompute(CountedCompleter<java.lang.Boolean> parent, MultiBlockData data, Block block, World w, BlockPos corner, BlockPos size, ECalculationStep step, AtomicBoolean valid) {
        super(parent);
        this.block = block;
        this.world = w;
        this.corner = corner;
        this.parent = parent;
        this.step = step;
        this.valid = valid;
        this.data = data;
        this.size = size;

    }

    @Override
    public void compute() {
        if (parent == null) {
            StructureCheckCompute bottomTask = new StructureCheckCompute(this, data, block, world, corner, size, ECalculationStep.BOTTOM, valid);
            StructureCheckCompute topTask = new StructureCheckCompute(this, data, block, world, corner, size, ECalculationStep.TOP, valid);
            StructureCheckCompute frameTask = new StructureCheckCompute(this, data, block, world, corner, size, ECalculationStep.FRAME, valid);
            StructureCheckCompute sideTask = new StructureCheckCompute(this, data, block, world, corner, size, ECalculationStep.SIDE, valid);
            StructureCheckCompute insideTask = new StructureCheckCompute(this, data, block, world, corner, size, ECalculationStep.INSIDE, valid);
            addToPendingCount(5);
            bottomTask.fork();
            topTask.fork();
            frameTask.fork();
            sideTask.fork();
            insideTask.fork();
        } else {
            if (!valid.get()) {
                tryComplete();
            }
            BlockPos diagonalBlock = corner.offset(size.getX() - 1, 0, size.getZ() - 1);
            switch (step) {
                case FRAME: {
                    List<String> blockNamesFrame = Arrays.stream(data.config.BLOCKS.get("frame")).map(k -> data.config.getBlocknameByKey(k)).collect(Collectors.toList());
                    if (data.isRegular && data.isFixedSize) {
                        valid.set(StructureCheckFixedSize.validateFixedSizeFrame(corner, world, data, size, blockNamesFrame, diagonalBlock));
                        tryComplete();
                    }
                    break;
                }
                case TOP: {
                    List<String> blockNamesTop = Arrays.stream(data.config.BLOCKS.get("top")).map(k -> data.config.getBlocknameByKey(k)).collect(Collectors.toList());
                    if (data.isFixedSize && data.isRegular) {
                        valid.set(StructureCheckFixedSize.validateTopFixedSize(corner, world, data, size, blockNamesTop, diagonalBlock));
                        tryComplete();
                    }
                    break;
                }
                case BOTTOM:{
                    List<String> blockNamesBottom = Arrays.stream(data.config.BLOCKS.get("bottom")).map(k -> data.config.getBlocknameByKey(k)).collect(Collectors.toList());
                    if (data.isFixedSize && data.isRegular) {
                        valid.set(StructureCheckFixedSize.validateBottomFixedSize(corner, world, data, size, blockNamesBottom, diagonalBlock));
                        tryComplete();
                    }
                    break;
                }
                case SIDE:{
                    List<String> blockNamesFaces = Arrays.stream(data.config.BLOCKS.get("face")).map(k -> data.config.getBlocknameByKey(k)).collect(Collectors.toList());
                    if (data.isFixedSize && data.isRegular) {
                        valid.set(StructureCheckFixedSize.validateBottomFixedSize(corner, world, data, size, blockNamesFaces, diagonalBlock));
                        tryComplete();
                    }
                    break;
                }
                case INSIDE:{
                    tryComplete();
                }

            }
            System.out.println("I am checking something i guess at " + step + " is valid? = " + valid.get());
        }
        tryComplete();
    }

    @Override
    public void onCompletion(CountedCompleter<?> caller) {
        if (caller == this) {
            System.out.printf("completed thread : %s Value of valid =%s%n", Thread
                    .currentThread().getName(), valid);
        }
        super.onCompletion(caller);
    }

    @Override
    public Boolean getRawResult() {
        return valid.get();
    }
}
