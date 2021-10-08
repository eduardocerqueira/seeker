//date: 2021-10-08T17:00:40Z
//url: https://api.github.com/gists/1e0f1632157d9d92379925ae044a300d
//owner: https://api.github.com/users/Cy4Bot

public void handleAutoEject() {
		Direction direction = Direction.DOWN;
		if (HopperTileEntity.getContainerAt(level, worldPosition.relative(direction, 1)) != null) {
			if (!this.getItem(0).isEmpty()) {
				IInventory autoEjectContainer = HopperTileEntity.getContainerAt(level,
						worldPosition.relative(direction, 1));
				ItemStack output = this.getItem(0);
				for (int i = 0; i <= autoEjectContainer.getContainerSize() - 1; i++) {
					if (TileEntityHelper.canPlaceItemInStack(autoEjectContainer.getItem(i), output)) {
						InventoryHelper.tryMoveInItem((IInventory) this, autoEjectContainer, output, i,
								direction.getOpposite());
						TileEntityHelper.updateTE(this);
						TileEntityHelper.updateTE(this.level.getBlockEntity(worldPosition.relative(direction, 1)));
						break;
					}
				}
			}
		}
	}
/*
Here's what this method does:
1. Check if the hopper is full. If it is, then it will try to eject the item into the world.
2. If the hopper is not full, then it will try to eject the item into the container above it.
3. If the hopper is not full and there is no container above it, then it will try to eject the item into the container below it.
4. If the hopper is not full and there is no container above it or below it, then it will try to eject the item into the world.
*/
	@Override
	public void handleAutoEject(Direction direction) {
		if (this.getItem(0).isEmpty()) {
			return;
		}
		if (this.isFull()) {
			this.handleAutoEject();
		} else {
			if (HopperTileEntity.getContainerAt(level, worldPosition.relative(direction, 1)) != null) {
				IInventory autoEjectContainer = HopperTileEntity.getContainerAt(level,
						worldPosition.relative(direction, 1));
				ItemStack output = this.getItem(0);
				for (int i = 0; i <= autoEjectContainer.getContainerSize() - 1; i++) {
					if (TileEntityHelper.canPlaceItemInStack(autoEjectContainer.getItem(i), output)) {
						InventoryHelper.tryMoveInItem((IInventory) this, autoEjectContainer, output, i,
								direction.getOpposite());
						TileEntityHelper.updateTE(this);
						TileEntityHelper.updateTE(this.level.getBlockEntity(worldPosition.relative(direction, 1)));
						break;
					}
				}
			} else {
				this.handleAutoEject();
			}
		}
	}