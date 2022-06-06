//date: 2022-06-06T16:52:56Z
//url: https://api.github.com/gists/a4e096ac5cfc395068c6039c4a1e457a
//owner: https://api.github.com/users/patricktran9

package com.tutorial.repository;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import org.springframework.stereotype.Repository;

import com.tutorial.repository.entity.DistrictEntity;
import com.tutorial.utils.JDBCUtils;
@Repository
public class DistrictRepositoryImpl implements DistrictRepository {
	private Connection conn = null;
	private PreparedStatement preparedStatement;
	private ResultSet rs = null;
	@Override
	public DistrictEntity findNameById(Integer districtId) {
		try {
			conn= JDBCUtils.getConnections();
			if (conn != null) {
				String query= "SELECT name FROM district WHERE id = ?";
				preparedStatement = conn.prepareStatement(query);
				if(districtId!=null && districtId >0) {
					preparedStatement.setInt(1,districtId);
				}
				rs = preparedStatement.executeQuery();	
				while (rs.next()) {
					DistrictEntity districtEntity = new DistrictEntity();
					districtEntity.setName(rs.getString("name"));
					return districtEntity;
				}
			} else {
				System.out.println("cannot connect...");
			}

		} catch ( SQLException e) {
			System.out.println(e.getMessage());
		} catch (Exception e) {
			System.out.println(e.getMessage());
		} finally {
			JDBCUtils.closeConnections(preparedStatement, rs, conn);
	}
		return null;
	}

}