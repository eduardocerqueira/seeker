//date: 2022-06-06T16:50:04Z
//url: https://api.github.com/gists/24d5d95742cd07164cac8074308b8303
//owner: https://api.github.com/users/patricktran9

package com.tutorial.repository;

import java.util.List;
import java.util.Map;

import org.springframework.stereotype.Component;
import com.tutorial.repository.entity.BuildingEntity;


@Component
public interface BuildingRepository {
	List<BuildingEntity> findByConditions(Map<String, String> params);
}