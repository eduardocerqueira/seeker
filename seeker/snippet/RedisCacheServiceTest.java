//date: 2023-08-23T17:06:52Z
//url: https://api.github.com/gists/014a214f47a25867b7b2cba988b21381
//owner: https://api.github.com/users/snmaddula


import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.ValueOperations;
import static org.springframework.test.util.ReflectionTestUtils.*;

import com.example.redis.poc.dto.CacheDto;

@ExtendWith(MockitoExtension.class)
class RedisCacheServiceTest {

	@Mock
	RedisTemplate<String, Object> redisTemplate;
	@Mock
	RedisTemplate<String, Object> masterTemplate;
	@Mock
	ValueOperations<String, Object> valueOps;

	@InjectMocks
	RedisCacheService redisCacheService;

	@BeforeEach
	void setup() {
		setField("redisCacheService", "redisTemplate", redisTemplate);
		setField("redisCacheService", "masterTemplate", masterTemplate);
	}

	@Test
	void testPutCache() {
		CacheDto cacheDto = CacheDto.builder().name("name").key("key").value("VALUE").ttl(100L).build();
		String key = redisCacheService.prepareKey(cacheDto.getName(), cacheDto.getKey());
		when(redisTemplate.opsForValue()).thenReturn(valueOps);
		doNothing().when(valueOps).set(key, cacheDto.getValue(), cacheDto.getTtl(), TimeUnit.SECONDS);
		redisCacheService.putCache(cacheDto);
		verify(redisTemplate).opsForValue();
		verify(valueOps).set(key, cacheDto.getValue(), cacheDto.getTtl(), TimeUnit.SECONDS);
		verifyNoMoreInteractions(redisTemplate, valueOps);
	}

	@Test
	void testGetCache() {
		when(redisTemplate.opsForValue()).thenReturn(valueOps);
		when(valueOps.get(Mockito.anyString())).thenReturn("VALUE");
		redisCacheService.getCache("name", "key");
		verify(redisTemplate).opsForValue();
		verify(valueOps).get(Mockito.anyString());
		verifyNoMoreInteractions(redisTemplate, valueOps);
	}

	@Test
	void testDeleteCache() {
		redisCacheService.deleteCache("name", "key");
		verifyNoMoreInteractions(masterTemplate);
	}

}
