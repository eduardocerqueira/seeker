//date: 2022-03-03T17:14:12Z
//url: https://api.github.com/gists/450a547aaefa9ffe2a96b7e2b923c666
//owner: https://api.github.com/users/Yuhtin

    public Collection<Chunk> getNearbyChunksGrid(Location location) {
        val offset = new int[]{-1, 0, 1};

        val world = location.getWorld();
        val baseX = location.getChunk().getX();
        val baseZ = location.getChunk().getZ();

        val chunksAroundPlayer = new HashSet<Chunk>();
        for (val x : offset) {
            for (val z : offset) {
                val chunk = world.getChunkAt(baseX + x, baseZ + z);
                chunksAroundPlayer.add(chunk);
            }
        }

        return chunksAroundPlayer;
    }