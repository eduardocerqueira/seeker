//date: 2022-06-06T16:56:53Z
//url: https://api.github.com/gists/1289ae57c00e0a10842ed8dec5f0b934
//owner: https://api.github.com/users/patricktran9

package com.tutorial.controller;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.tutorial.model.response.BuildingSearchResponse;
import com.tutorial.service.BuildingService;

@RestController
@RequestMapping("/api")
public class BuildingSearchAPI {
	private HashMap<String, String> listInput = new HashMap<String, String>();

	@Autowired
	private BuildingService buildingService;

	@GetMapping("/buildings")
	public Object showBuildings(@RequestParam(required = false) Map<String, String> params,
			@RequestParam(required = false) List<String> types) {
		List<BuildingSearchResponse> returnBuildings = buildingService.findByConditions(params);
		return returnBuildings;
	}

}