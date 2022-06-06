//date: 2022-06-06T16:54:56Z
//url: https://api.github.com/gists/e7d14a93b721ad582e024cb91717bc24
//owner: https://api.github.com/users/patricktran9

package com.tutorial.service;

import java.util.List;
import java.util.Map;

import com.tutorial.model.response.BuildingSearchResponse;

public interface BuildingService {
	List<BuildingSearchResponse> findByConditions(Map<String, String> params);
}