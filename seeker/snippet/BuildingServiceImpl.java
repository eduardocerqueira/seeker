//date: 2022-06-06T16:55:57Z
//url: https://api.github.com/gists/461a95a7200f2a0187d98bfbfbfc97b6
//owner: https://api.github.com/users/patricktran9

package com.tutorial.service.impl;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;


import com.tutorial.model.response.BuildingSearchResponse;
import com.tutorial.repository.BuildingRepository;
import com.tutorial.repository.DistrictRepository;
import com.tutorial.repository.entity.BuildingEntity;
import com.tutorial.repository.entity.DistrictEntity;
import com.tutorial.service.BuildingService;
import com.tutorial.utils.ValidateUtils;
@Service
public class BuildingServiceImpl implements BuildingService{
	@Autowired
	private BuildingRepository buildingRepository;
	@Autowired
	private DistrictRepository districtRepository;

	@Override
	public List<BuildingSearchResponse> findByConditions(Map<String, String> params) {
		List<BuildingSearchResponse> responses = new ArrayList<>();
		if (!params.isEmpty() ) {
			List<BuildingEntity> buildings = buildingRepository.findByConditions(params);

			for (BuildingEntity building : buildings) {
				BuildingSearchResponse buildingSearchResponse = new BuildingSearchResponse();

				if (ValidateUtils.isNotNull(building.getCreatedDate())) {
					buildingSearchResponse.setCreatedDate((building.getCreatedDate()));
				} else {
					buildingSearchResponse.setCreatedDate("No information!");
				}
				if (ValidateUtils.isNotNull(building.getName())) {
					buildingSearchResponse.setName(building.getName());
				} else {
					buildingSearchResponse.setName("No information!");
				}
				String street = building.getStreet();
				String ward = building.getWard();
				Integer districtId = building.getDistrictid();
				buildingSearchResponse.setAddress(buildAddress(street, ward, districtId));
				if (ValidateUtils.isNotNull(building.getStreet()) && ValidateUtils.isNotNull(building.getWard())
						&& ValidateUtils.isNotNull(building.getDistrictid())) {

				} else {
					buildingSearchResponse.setAddress("No information!");
				}
				if (ValidateUtils.isNotNull(building.getManagerName())) {
					buildingSearchResponse.setManagerName(building.getManagerName());
				} else {
					buildingSearchResponse.setManagerName("No information!");
				}
				if (ValidateUtils.isNotNull(building.getManagerPhone())) {
					buildingSearchResponse.setManagerPhone(building.getManagerPhone());
				} else {
					buildingSearchResponse.setManagerPhone("No information!");
				}
				if (ValidateUtils.isNotNull(building.getFloorArea())) {
					buildingSearchResponse.setFloorArea(building.getFloorArea());
				}
				if (ValidateUtils.isNotNull(building.getRentPrice())) {
					buildingSearchResponse.setRentPrice(building.getRentPrice());
				}
				if (ValidateUtils.isNotNull(building.getServiceFee())) {
					buildingSearchResponse.setServiceFee(building.getServiceFee());
				} else {
					buildingSearchResponse.setServiceFee("No information!");
				}
				if (ValidateUtils.isNotNull(building.getBrokerageFee())) {
					buildingSearchResponse.setBrokerageFee(building.getBrokerageFee());
				}
				responses.add(buildingSearchResponse);
			}
		}

		return responses;
	}

	private String buildAddress(String street, String ward, Integer districtId) {
		DistrictEntity districtEntity = districtRepository.findNameById(districtId);
		StringBuilder address = new StringBuilder();

		if (ValidateUtils.isNotBlank(street)) {
			address.append(street);
		}
		if (ValidateUtils.isNotBlank(ward)) {
			address.append(",").append(ward);
		}
		if (null != districtEntity) {
			address.append(",").append(districtEntity.getName());
		}
		return address.toString();
	}

}