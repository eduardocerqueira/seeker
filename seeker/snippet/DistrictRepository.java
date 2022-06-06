//date: 2022-06-06T16:50:54Z
//url: https://api.github.com/gists/94ba02bf009360eb37e9cfa574ed3156
//owner: https://api.github.com/users/patricktran9

package com.tutorial.repository;

import com.tutorial.repository.entity.DistrictEntity;

public interface DistrictRepository {
	DistrictEntity findNameById(Integer districtId);
}