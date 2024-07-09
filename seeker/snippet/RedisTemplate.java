//date: 2024-07-09T17:09:38Z
//url: https://api.github.com/gists/7b28c84aaa859f0d7118fbf2bf070ab8
//owner: https://api.github.com/users/enrinal

public RedisTemplate<String, byte[]> redisTemplateByte(
        LettuceConnectionFactory defaultLettuceConnectionFactory) {
    RedisTemplate<String, byte[]> template = new RedisTemplate<>();
    template.setKeySerializer(RedisSerializer.string());
    template.setHashKeySerializer(RedisSerializer.string());
    template.setValueSerializer(RedisSerializer.byteArray());
    template.setHashValueSerializer(RedisSerializer.byteArray());
    template.setConnectionFactory(defaultLettuceConnectionFactory);
    template.afterPropertiesSet();
    return template;
}
