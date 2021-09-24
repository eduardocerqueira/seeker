//date: 2021-09-24T16:52:40Z
//url: https://api.github.com/gists/ed9228c7e1ca3e8f1947d4624eb4353d
//owner: https://api.github.com/users/joshuaepstein

package joshuaepstein.lastlife.world;

import joshuaepstein.lastlife.Main;
import joshuaepstein.lastlife.init.ModNetwork;
import joshuaepstein.lastlife.network.message.LivesMessage;
import joshuaepstein.lastlife.util.NetcodeUtils;
import net.minecraft.nbt.CompoundNBT;
import net.minecraft.server.MinecraftServer;
import net.minecraftforge.common.util.INBTSerializable;
import net.minecraftforge.fml.network.NetworkDirection;

import java.util.UUID;

public class PlayerLivesStats implements INBTSerializable<CompoundNBT> {

    private final UUID uuid;
    private int playerLives;

    public PlayerLivesStats(UUID uuid){
        this.uuid = uuid;
    }

    public int getPlayerLives(){
        return playerLives;
    }



    public PlayerLivesStats setPlayerLives(MinecraftServer server, int level){
        this.playerLives = level;

        sync(server);

        return this;
    }
    public PlayerLivesStats addLife(MinecraftServer server){
        this.playerLives+=1;

        sync(server);

        return this;
    }
    public PlayerLivesStats removeLife(MinecraftServer server){
        this.playerLives-=1;

        sync(server);

        return this;
    }

    public void sync(MinecraftServer server) {
        NetcodeUtils.runIfPresent(server, this.uuid, player -> {
            ModNetwork.CHANNEL.sendTo(
                    new LivesMessage(this.playerLives),
                    player.connection.getConnection(),
                    NetworkDirection.PLAY_TO_CLIENT
            );
        });
    }
    @Override
    public CompoundNBT serializeNBT() {
        CompoundNBT nbt = new CompoundNBT();
        nbt.putInt("playerLives", playerLives);
        return nbt;
    }
    public PlayerLivesStats reset(MinecraftServer server) {
        this.playerLives = Main.getRandomNumber(2, 7);

        sync(server);

        return this;
    }
    public PlayerLivesStats addLives(int amount){
        this.playerLives += amount;
        return this;
    }
    public PlayerLivesStats removeLife(){
        this.playerLives -= 1;
        return this;
    }
    @Override
    public void deserializeNBT(CompoundNBT nbt) {
        this.playerLives = nbt.getInt("playerLives");
    }
}
