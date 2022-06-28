//date: 2022-06-28T17:02:17Z
//url: https://api.github.com/gists/f4ad8b1f2ab56d20ceac0c021f5d13bd
//owner: https://api.github.com/users/Cy4Bot

@Override
	public void tick() {
		if(level.isClientSide()) {
            return;
		}
		if (getRecipe() == null) {
			System.out.println("Recipe is not found");
		}
		BlockState state = level.getBlockState(getBlockPos());
        if (state.getValue(BlockStateProperties.POWERED) != counter > 0) {
            level.setBlock(getBlockPos(), state.setValue(BlockStateProperties.POWERED, counter > 0),
                    Constants.BlockFlags.NOTIFY_NEIGHBORS + Constants.BlockFlags.BLOCK_UPDATE);

        }
        if(fuelCounter > 0)
            fuelCounter--;
        if(getItem(0).isEmpty() || getItem(1).isEmpty() || getItem(2).isEmpty()){
            reset();
            return;
           }
        // this checks whether there is a matching recipe, and then checks whether fuel
        // is available and the checks whether the outputSlot is occupied
        ExcimerLaserRecipe recipe = getRecipe();
        if(recipe == null || !canStartWorking(recipe)){
          reset();
           return;
        }
        if(counter <= 0){
            //ill create this method in a second
            startWorking(recipe);
        }
        if(counter > 0){
          counter--;
          if(counter == 0){
            //ill create the method in a second
            finishWork(recipe);
          }
        }
}