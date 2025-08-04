//date: 2025-08-04T17:17:07Z
//url: https://api.github.com/gists/12db6fc4c3966705fdbf827407bc6279
//owner: https://api.github.com/users/mudkipdev

package gg.skylite.lobby;

import net.kyori.adventure.text.Component;
import net.minestom.server.MinecraftServer;
import net.minestom.server.coordinate.Pos;
import net.minestom.server.coordinate.Vec;
import net.minestom.server.entity.*;
import net.minestom.server.entity.metadata.display.AbstractDisplayMeta;
import net.minestom.server.entity.metadata.display.TextDisplayMeta;
import net.minestom.server.event.entity.EntityAttackEvent;
import net.minestom.server.event.player.PlayerEntityInteractEvent;
import net.minestom.server.instance.Instance;
import net.minestom.server.network.packet.server.play.*;
import net.minestom.server.scoreboard.Team;
import net.minestom.server.timer.TaskSchedule;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;

public final class Npc extends EntityCreature {
    private static final double LOOK_DISTANCE = 20.0D;
    private static Team npcTeam;

    private final String username;
    private final @Nullable PlayerSkin skin;
    private final Entity nameTag;
    private final Consumer<Player> action;

    public Npc(Component name, @Nullable PlayerSkin skin, Consumer<Player> action) {
        super(EntityType.PLAYER, UUID.randomUUID());
        this.username = UUID.randomUUID().toString().substring(0, 6);
        this.skin = skin;
        this.action = action;

        this.setNoGravity(true);
        this.hasPhysics = false;
        this.setSynchronizationTicks(Integer.MAX_VALUE);

        this.nameTag = new Entity(EntityType.TEXT_DISPLAY);
        this.nameTag.editEntityMeta(TextDisplayMeta.class, this.editNameTagMeta(name));

        if (npcTeam == null) {
            npcTeam = MinecraftServer.getTeamManager().createBuilder("npcs")
                    .nameTagVisibility(TeamsPacket.NameTagVisibility.NEVER)
                    .build();
        }

        // setTeam does not work, as it will use uuid instead of username
        npcTeam.addMember(this.username);
    }

    @Override
    public CompletableFuture<Void> setInstance(@NotNull Instance instance, @NotNull Pos spawnPosition) {
        var future = super.setInstance(instance, spawnPosition);
        instance.scheduler().submitTask(this::lookTask);
        instance.eventNode().addListener(EntityAttackEvent.class, this::handleAttack);
        instance.eventNode().addListener(PlayerEntityInteractEvent.class, this::handleInteraction);
        return CompletableFuture.allOf(future, this.nameTag.setInstance(instance, spawnPosition)
                .whenComplete((value, error) -> this.addPassenger(this.nameTag)));
    }

    @Override
    public void updateNewViewer(@NotNull Player player) {
        var properties = new ArrayList<PlayerInfoUpdatePacket.Property>();

        if (this.skin != null && this.skin.textures() != null && this.skin.signature() != null) {
            properties.add(new PlayerInfoUpdatePacket.Property("textures", this.skin.textures(), this.skin.signature()));
        }

        var entry = new PlayerInfoUpdatePacket.Entry(
                this.getUuid(), this.username, properties,
                false, 0, GameMode.SURVIVAL,
                null, null, 0);

        player.sendPacket(new PlayerInfoUpdatePacket(PlayerInfoUpdatePacket.Action.ADD_PLAYER, entry));
        super.updateNewViewer(player);
        this.nameTag.addViewer(player);

        player.sendPacket(new EntityMetaDataPacket(
                this.getEntityId(),
                Map.of(17, Metadata.Byte((byte) 127))));
    }

    @Override
    public void updateOldViewer(@NotNull Player player) {
        super.updateOldViewer(player);
        player.sendPacket(new PlayerInfoRemovePacket(this.getUuid()));
        this.nameTag.removeViewer(player);
    }

    @Override
    protected void remove(boolean permanent) {
        super.remove(permanent);
        this.nameTag.remove();
    }

    public void setName(Component name) {
        this.nameTag.editEntityMeta(TextDisplayMeta.class, this.editNameTagMeta(name));
    }

    private Consumer<TextDisplayMeta> editNameTagMeta(Component name) {
        return meta -> {
            meta.setTranslation(new Vec(0.0D, 0.3D, 0.0D));
            meta.setBillboardRenderConstraints(AbstractDisplayMeta.BillboardConstraints.CENTER);
            meta.setBackgroundColor(0x00000000);
            meta.setShadow(true);
            meta.setText(name);
        };
    }

    private void handleAttack(EntityAttackEvent event) {
        if (event.getTarget() == this && event.getEntity() instanceof Player player) {
            this.action.accept(player);
        }
    }

    private void handleInteraction(PlayerEntityInteractEvent event) {
        if (event.getTarget() == this && event.getHand() == PlayerHand.MAIN) {
            this.action.accept(event.getPlayer());
        }
    }

    private TaskSchedule lookTask() {
        for (var player : this.getInstance().getPlayers()) {
            var position = player.getDistance(this) > LOOK_DISTANCE
                    ? this.position : this.position.withLookAt(player.getPosition());

            player.sendPackets(
                    new EntityHeadLookPacket(this.getEntityId(), position.yaw()),
                    new EntityRotationPacket(this.getEntityId(), position.yaw(), position.pitch(), this.onGround));
        }

        return TaskSchedule.nextTick();
    }
}