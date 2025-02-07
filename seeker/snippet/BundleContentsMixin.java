//date: 2025-02-07T17:11:47Z
//url: https://api.github.com/gists/517969d0a9548748088e4f0dc42581f7
//owner: https://api.github.com/users/ghasto69

@Mixin(BundleContents.class)
public abstract class BundleContentsMixin implements BundleContentsCapacityMultiplier {
    @Unique
    private float enchantmentExpansion$capacityMultiplier;

    @ModifyExpressionValue(
            method = "<clinit>",
            at = @At(value = "INVOKE", target = "Lcom/mojang/serialization/Codec;flatXmap(Ljava/util/function/Function;Ljava/util/function/Function;)Lcom/mojang/serialization/Codec;")
    )
    private static Codec<BundleContents> modifyCodec(Codec<BundleContents> codec) {
        return Codec.withAlternative(codec, RecordCodecBuilder.create(instance -> instance.group(
                                codec.fieldOf("value").forGetter(Function.identity()),
                                Codec.FLOAT.fieldOf("capacity").forGetter(contents -> (BundleContentsCapacityMultiplier.cast(contents).enchantmentExpansion$getCapacityMultiplier()))
                        ).apply(instance, (contents, capacity) -> (BundleContentsCapacityMultiplier.cast(contents).enchantmentExpansion$setCapacityMultiplier(capacity))
                )
        ));
    }

    @ModifyExpressionValue(
            method = "<clinit>",
            at = @At(
                    value = "INVOKE",
                    target = "Lnet/minecraft/network/codec/StreamCodec;map(Ljava/util/function/Function;Ljava/util/function/Function;)Lnet/minecraft/network/codec/StreamCodec;"
            )
    )
    private static <B, O> StreamCodec<B, O> modifyStreamCodec(StreamCodec<B, O> original) {
        return StreamCodec.composite(
                original,
                Function.identity(),
                (StreamCodec<? super B, ? super Integer>) ByteBufCodecs.VAR_INT,
                (o) -> 3,
                (contents, integer) -> contents
        );
    }

    @Override
    public float enchantmentExpansion$getCapacityMultiplier() {
        return this.enchantmentExpansion$capacityMultiplier;
    }

    @Override
    public BundleContents enchantmentExpansion$setCapacityMultiplier(float capacityMultiplier) {
        this.enchantmentExpansion$capacityMultiplier = capacityMultiplier;
        return (BundleContents) (Object) this;
    }

    @Inject(
            method = "<init>(Ljava/util/List;Lorg/apache/commons/lang3/math/Fraction;I)V",
            at = @At("TAIL")
    )
    private void setDefaultCapacityMultiplier(List<ItemStack> list, Fraction fraction, int i, CallbackInfo ci) {
        this.enchantmentExpansion$capacityMultiplier = 1.0f;
    }

    @ModifyExpressionValue(
            method = "equals",
            at = @At(
                    value = "INVOKE",
                    target = "Lorg/apache/commons/lang3/math/Fraction;equals(Ljava/lang/Object;)Z"
            )
    )
    private boolean equals(boolean original, Object object) {
        return object instanceof BundleContents bundleContents && BundleContentsCapacityMultiplier.cast(bundleContents).enchantmentExpansion$getCapacityMultiplier() == this.enchantmentExpansion$capacityMultiplier;
    }
}