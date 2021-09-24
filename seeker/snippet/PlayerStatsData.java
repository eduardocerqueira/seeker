//date: 2021-09-24T16:52:40Z
//url: https://api.github.com/gists/ed9228c7e1ca3e8f1947d4624eb4353d
//owner: https://api.github.com/users/joshuaepstein

package joshuaepstein.lastlife.world.data;

import joshuaepstein.lastlife.Main;
import joshuaepstein.lastlife.world.PlayerLivesStats;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.entity.player.ServerPlayerEntity;
import net.minecraft.nbt.CompoundNBT;
import net.minecraft.nbt.ListNBT;
import net.minecraft.nbt.StringNBT;
import net.minecraft.world.server.ServerWorld;
import net.minecraft.world.storage.WorldSavedData;
import net.minecraftforge.common.util.Constants;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

public class PlayerStatsData extends WorldSavedData {
    protected static final String DATA_NAME = Main.MOD_ID + "_PlayerLives";

    private Map<UUID, PlayerLivesStats> playerMap = new HashMap<>();

    public PlayerStatsData() {
        super(DATA_NAME);
    }

    public PlayerStatsData(String name){
        super(name);
    }

    public PlayerLivesStats getStats(PlayerEntity player){
        return getStats(player.getUUID());
    }

    public PlayerLivesStats getStats(UUID uuid){
        return this.playerMap.computeIfAbsent(uuid, PlayerLivesStats::new);
    }
    public PlayerStatsData addLife(ServerPlayerEntity player){
        this.getStats(player).addLife(player.getServer());

        setDirty();
        return this;
    }
    public PlayerStatsData setLives(ServerPlayerEntity player, int lives){
        this.getStats(player).setPlayerLives(player.getServer(), lives);

        setDirty();
        return this;
    }

    public PlayerStatsData removeLife(ServerPlayerEntity player){
        this.getStats(player).removeLife(player.getServer());

        setDirty();
        return this;
    }
    public PlayerStatsData reset(ServerPlayerEntity player) {
        this.getStats(player).reset(player.getServer());

        setDirty();
        return this;
    }

    /* ------------------------------- */

    @Override
    public void load(CompoundNBT nbt) {
        ListNBT playerList = nbt.getList("PlayerEntries", Constants.NBT.TAG_STRING);
        ListNBT statEntries = nbt.getList("StatEntries", Constants.NBT.TAG_COMPOUND);

        if (playerList.size() != statEntries.size()) {
            throw new IllegalStateException("Map doesn't have the same amount of keys as values");
        }

        for (int i = 0; i < playerList.size(); i++) {
            UUID playerUUID = UUID.fromString(playerList.getString(i));
            this.getStats(playerUUID).deserializeNBT(statEntries.getCompound(i));
        }
    }

    @Override
    public CompoundNBT save(CompoundNBT nbt) {
        ListNBT playerList = new ListNBT();
        ListNBT statsList = new ListNBT();

        this.playerMap.forEach((uuid, stats) -> {
            playerList.add(StringNBT.valueOf(uuid.toString()));
            statsList.add(stats.serializeNBT());
        });

        nbt.put("PlayerEntries", playerList);
        nbt.put("StatEntries", statsList);

        return nbt;
    }

    public static PlayerStatsData get(ServerWorld world) {
        return world.getServer().overworld()
                .getDataStorage().computeIfAbsent(PlayerStatsData::new, DATA_NAME);
    }
}
