//date: 2022-11-18T16:55:59Z
//url: https://api.github.com/gists/08acc6e7d9632d4da09881a274c2b6cd
//owner: https://api.github.com/users/zskamljic

package com.zskamljic.tophium.rendering;

import net.minecraft.MethodsReturnNonnullByDefault;
import net.minecraft.client.renderer.block.model.BakedQuad;
import net.minecraft.client.renderer.block.model.ItemOverrides;
import net.minecraft.client.renderer.texture.TextureAtlasSprite;
import net.minecraft.client.resources.model.BakedModel;
import net.minecraft.core.Direction;
import net.minecraft.util.RandomSource;
import net.minecraft.world.level.block.state.BlockState;
import org.jetbrains.annotations.Nullable;

import javax.annotation.ParametersAreNonnullByDefault;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

@ParametersAreNonnullByDefault
@MethodsReturnNonnullByDefault
public class VoxelBakedModel implements BakedModel {

    private static final Map<Direction, float[]> VERTICES = Map.of(
        Direction.UP, new float[]{
            0.4f, 0.8f, 0.4f,
            0.4f, 0.8f, 0.8f,
            0.8f, 0.8f, 0.8f,
            0.8f, 0.8f, 0.4f
        },
        Direction.DOWN, new float[]{
            0.4f, 0.4f, 0.8f,
            0.4f, 0.4f, 0.4f,
            0.8f, 0.4f, 0.4f,
            0.8f, 0.4f, 0.8f
        },
        Direction.NORTH, new float[]{
            0.8f, 0.8f, 0.4f,
            0.8f, 0.4f, 0.4f,
            0.4f, 0.4f, 0.4f,
            0.4f, 0.8f, 0.4f
        },
        Direction.SOUTH, new float[]{
            0.4f, 0.8f, 0.8f,
            0.4f, 0.4f, 0.8f,
            0.8f, 0.4f, 0.8f,
            0.8f, 0.8f, 0.8f
        },
        Direction.EAST, new float[]{
            0.8f, 0.8f, 0.8f,
            0.8f, 0.4f, 0.8f,
            0.8f, 0.4f, 0.4f,
            0.8f, 0.8f, 0.4f
        },
        Direction.WEST, new float[]{
            0.4f, 0.8f, 0.4f,
            0.4f, 0.4f, 0.4f,
            0.4f, 0.4f, 0.8f,
            0.4f, 0.8f, 0.8f
        }
    );

    private final BakedModel model;

    public VoxelBakedModel(BakedModel model) {
        this.model = model;
    }

    @Override
    public List<BakedQuad> getQuads(@Nullable BlockState state, @Nullable Direction direction, RandomSource random) {
        var originalQuads = model.getQuads(state, direction, random);
        if (direction == null) return originalQuads;

        var actualQuads = new ArrayList<BakedQuad>();
        var byteBuffer = ByteBuffer.allocate(48);
        for (var quad : originalQuads) {
            var floatBuffer = byteBuffer.asFloatBuffer();
            floatBuffer.clear();
            floatBuffer.put(VERTICES.get(direction));
            var vertices = Arrays.copyOf(quad.getVertices(), quad.getVertices().length);
            vertices[0] = byteBuffer.getInt(0);
            vertices[1] = byteBuffer.getInt(4);
            vertices[2] = byteBuffer.getInt(8);

            vertices[8] = byteBuffer.getInt(12);
            vertices[9] = byteBuffer.getInt(16);
            vertices[10] = byteBuffer.getInt(20);

            vertices[16] = byteBuffer.getInt(24);
            vertices[17] = byteBuffer.getInt(28);
            vertices[18] = byteBuffer.getInt(32);

            vertices[24] = byteBuffer.getInt(36);
            vertices[25] = byteBuffer.getInt(40);
            vertices[26] = byteBuffer.getInt(44);

            var newQuad = new BakedQuad(vertices, quad.getTintIndex(), quad.getDirection(), quad.getSprite(), quad.isShade());
            actualQuads.add(newQuad);
        }
        return actualQuads;
    }

    @Override
    public boolean useAmbientOcclusion() {
        return model.useAmbientOcclusion();
    }

    @Override
    public boolean isGui3d() {
        return model.isGui3d();
    }

    @Override
    public boolean usesBlockLight() {
        return model.usesBlockLight();
    }

    @Override
    public boolean isCustomRenderer() {
        return model.isCustomRenderer();
    }

    @Override
    public TextureAtlasSprite getParticleIcon() {
        return model.getParticleIcon();
    }

    @Override
    public ItemOverrides getOverrides() {
        return model.getOverrides();
    }
}
