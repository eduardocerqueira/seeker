//date: 2021-11-23T17:12:09Z
//url: https://api.github.com/gists/5238b28943e152d434154680004b46ef
//owner: https://api.github.com/users/Cy4Bot

package com.CheeseMan.firstmod.common.block;

import java.util.Random;

import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.state.BooleanProperty;
import net.minecraft.state.properties.BlockStateProperties;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.IWorldReader;
import net.minecraft.world.server.ServerWorld;

public class Grapes extends Block{
	protected static final BooleanProperty DOWN = BlockStateProperties.DOWN;

	public Grapes(Properties builder) {
		super(builder); 
		this.registerDefaultState(this.stateDefinition.any().setValue(DOWN, Boolean.valueOf(false)));
		 
		
	}
	@Override
	public void randomTick(BlockState state, ServerWorld world, BlockPos pos, Random rand) {
		
	}
	@Override
	public boolean canSurvive(BlockState state, IWorldReader worldIn, BlockPos pos) {
		return super.canSurvive(state, worldIn, pos);
	}
	
}
