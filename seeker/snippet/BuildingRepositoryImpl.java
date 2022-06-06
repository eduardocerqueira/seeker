//date: 2022-06-06T16:51:58Z
//url: https://api.github.com/gists/c68d5ae45b6ab300ae0e9195316441ff
//owner: https://api.github.com/users/patricktran9

package com.tutorial.repository;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.springframework.stereotype.Repository;

import com.tutorial.repository.entity.BuildingEntity;
import com.tutorial.utils.JDBCUtils;
import com.tutorial.utils.ValidateUtils;
@Repository
public class BuildingRepositoryImpl implements BuildingRepository {
	private Connection conn = null;
	private Statement stmt = null;
	private ResultSet rs = null;

	@Override
	public List<BuildingEntity> findByConditions(Map<String, String> params) {
		List<BuildingEntity> buildingEntities = new ArrayList<>();
		StringBuilder QUERY = buildQuery(params);
		try {
			conn = JDBCUtils.getConnections();

			if (conn != null) {
				System.out.println("successfully...");
				stmt = conn.createStatement();
				rs = stmt.executeQuery(QUERY.toString());
				while (rs.next()) {
					BuildingEntity buildingEntity = new BuildingEntity();
					buildingEntity.setId_building(rs.getLong("id_building"));
					buildingEntity.setName(rs.getString("name"));
					buildingEntity.setStreet(rs.getString("street"));
					buildingEntity.setWard(rs.getString("ward"));
					buildingEntity.setDistrictid(rs.getInt("districtid"));
					buildingEntity.setRentPrice(rs.getInt("rentprice"));
					buildingEntity.setManagerName(rs.getString("buildingmanagername"));
					buildingEntity.setManagerPhone(rs.getString("buildingmanagerphone"));
					buildingEntity.setCreatedDate(rs.getString("createddate"));
					buildingEntity.setServiceFee(rs.getString("servicefee"));
					buildingEntity.setBrokerageFee(rs.getFloat("brokeragefee"));
					buildingEntities.add(buildingEntity);
				}
			}

		} catch (SQLException e) {
			System.out.println(e.getMessage());
			// e.printStackTrace();
		} catch (Exception e) {
			System.out.println(e.getMessage());
		} finally {
			JDBCUtils.closeConnections(stmt, rs, conn);
		}
		return buildingEntities;
	}

	private StringBuilder buildQuery(Map<String, String> params) {
		StringBuilder sqlQuery = new StringBuilder(
				"SELECT b.id AS id_building, b.name, b.street, b.ward, b.districtid, b.managername AS buildingmanagername, b.managerphone AS buildingmanagerphone, b.rentprice, b.createddate, b.servicefee, b.brokeragefee \n"
						+ "FROM building AS b\n");
		StringBuilder whereQuery = new StringBuilder();
		buildQueryCommon(params, whereQuery);
		sqlQuery.append("WHERE 1=1\n").append(whereQuery).append("\nGROUP BY b.id");
		System.out.println(sqlQuery);
		return sqlQuery;
	}

	private void buildQueryCommon(Map<String, String> params, StringBuilder whereQuery) {
		String name = params.getOrDefault("name", null);
		if (ValidateUtils.isNotBlank(name)) {
			whereQuery.append("\nAND b.name LIKE '%" + name + "%'\n");
		}
		String floorArea = params.getOrDefault("floorarea", null);
		if (ValidateUtils.isNotBlank(floorArea)) {
			whereQuery.append("\nAND b.floorarea =" + floorArea + "\n");
		}
		String street = params.getOrDefault("street", null);
		if (ValidateUtils.isNotBlank(street)) {
			whereQuery.append("\nAND b.street LIKE '%" + street + "%'\n");
		}
		String ward = params.getOrDefault("ward", null);
		if (ValidateUtils.isNotBlank(ward)) {
			whereQuery.append("\nAND b.ward LIKE '%" + ward + "%'\n");
		}
		String district = params.getOrDefault("district", null);
		if (ValidateUtils.isNotBlank(district)) {
			whereQuery.append("\nAND b.districtid = " + district + "\n");
		}
		String numberOfBasement = params.getOrDefault("numberofbasement", null);
		if (ValidateUtils.isNotBlank(numberOfBasement)) {
			whereQuery.append("\nAND b.numberOfbasement = " + numberOfBasement + "\n");
		}
		String direction = params.getOrDefault("direction", null);
		if (ValidateUtils.isNotBlank(direction)) {
			whereQuery.append("\nAND direction LIKE '%" + direction + "%'\n");
		}
		String level = params.getOrDefault("level", null);
		if (ValidateUtils.isNotBlank(level)) {
			whereQuery.append("\nAND buildinglevel LIKE '%" + level + "%'\n");
		}
		String managerName = params.getOrDefault("managerName", null);
		if (ValidateUtils.isNotBlank(managerName)) {
			whereQuery.append("\nAND b.managername LIKE '%" + managerName + "%'\n");
		}
		String managerPhone = params.getOrDefault("managerphone", null);
		if (ValidateUtils.isNotBlank(managerPhone)) {
			whereQuery.append("\nAND b.managerphone LIKE '%" + managerPhone + "%'\n");
		}
		String rentPriceFrom = params.getOrDefault("rentpricefrom", null);
		if (ValidateUtils.isNotBlank(rentPriceFrom)) {
			whereQuery.append("\nAND b.rentprice >= ").append(rentPriceFrom);
		}
		String rentPriceTo = params.getOrDefault("rentpriceto", null);
		if (ValidateUtils.isNotBlank(rentPriceTo)) {
			whereQuery.append(" \nAND b.rentprice <= ").append(rentPriceTo);
		}
		
	}

}