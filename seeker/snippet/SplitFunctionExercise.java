//date: 2026-03-12T17:35:50Z
//url: https://api.github.com/gists/dc0526e34ccfaa39bd078e50a2df38bf
//owner: https://api.github.com/users/Facugarciar0805

import org.acme.CSVFile; // helper class to generate CSV files
import static org.acme.SQL.executeSQL; // helper method to execute SQL queries
import java.sql.ResultSet;
import java.sql.SQLException;

class SplitFunctionExercise {

  // Generate a CSV file with the final grades of each student
  public static void main(String[] args) throws SQLException {

    CSVFile csv = new CSVFile();
    ResultSet student = executeSQL("select ID, NAME from STUDENT");
    gradesCSVWriter(student, csv);
    csv.writeTo("final_grades.csv");
  }
  
  public static ResultSet getGrades(ResultSet student){
      return executeSQL("select GRADE_VALUE from GRADE"+
                                    " where STUDENT_ID = ?",
                                    student.getInt("ID"));
      
  }
                                   
 public float calculateAverageGrade(ResultSet){
    int gradesCount = 0;
      float gradesSum = 0;
      while (grades.next()) {
        gradesSum += grades.getInt("GRADE_VALUE");
        gradesCount++;
      }

      if(gradesCount > 0) {
        return gradesSum / gradesCount;
      }
      return -1;
 }
  public void gradesCSVWriter(ResultSet student, CSVFile csv){
    while (student.next()) {

      ResultSet grades = getGrades(student.getInt("ID");
      float average = calculateAverage(grades);
      
       csv.addRow(student.getString("NAME"), average);
      
    }
  
  }
                                   
}