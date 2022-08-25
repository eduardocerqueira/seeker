//date: 2022-08-25T15:31:59Z
//url: https://api.github.com/gists/5fce9f3de630c355ca6843b203bef333
//owner: https://api.github.com/users/DhavalDalal

import java.sql.*;
import java.util.*;

public class Sql {
  public List<Employee> fetchEmployees(String driverFQN, String dburl) throws SQLException {
    Connection connection = null;
    Statement statement = null;
    ResultSet resultSet = null;
    List<Employee> employees = new ArrayList<Employee>();
    try {
      Class.forName(driverFQN);
      connection = DriverManager.getConnection(dburl);
      statement = connection.createStatement();
      statement.execute("select * from employee");
      resultSet = statement.getResultSet();
      while (resultSet.next()) {
        int empId = resultSet.getInt(0);
        String name = resultSet.getString(1); 
        employees.add(new Employee(empId, name));
      }
    } catch (SQLException e) { 
      e.printStackTrace(); 
    } finally {
      if (connection != null) {
        connection.close();
        if (statement != null) {
          statement.close();
          if (resultSet != null)
            resultSet.close();
        }
      }
    }
    return employees;    
  }
}