//date: 2022-12-22T17:04:36Z
//url: https://api.github.com/gists/39523c0d9d97d75dace2d9b30fc11f0a
//owner: https://api.github.com/users/ridhofauza

import java.util.*;

public class CountMinutes {

   static String[] getFormat(String str) {
      String[] fmtTime = new String[2];
      str = str.replaceAll("[0-9:]*","");
      fmtTime = str.split("-");
      return fmtTime;
   }

   static int[] strToIntArr(String str) {
      str = str.replaceAll("[am|pm]","");
      String[] strArr = str.split(":");
      int[] time = new int[2];

      for (int i = 0; i < strArr.length; i++) {
         time[i] = Integer.parseInt(strArr[i]);
      }
      return time;
   }

   static int countMinutes(String str) {
      String[] fmtTime = getFormat(str); // e.g: ["am", "pm"]
      String[] newStr = str.split("-"); // e.g: ["09:00am", "1:00pm"]
      int[] startTime = strToIntArr(newStr[0]); // e.g: [09, 00]
      int[] endTime = strToIntArr(newStr[1]); // e.g: [01, 00]
      int minutes = startTime[1]+endTime[1];
      int hours = 0;

      if(!fmtTime[0].equals(fmtTime[1]) ) {
         // Jika beda am dengan pm [12+endTime-startTime]
         hours = 12+endTime[0]-startTime[0];
      } else {
         // Jika sama
         hours = endTime[0]-startTime[0];
      }
      minutes += hours*60;
      return minutes;
   }

   public static void main(String[] args) {
      // str = 9:00am-10:00am ==> 60
      // str = 1:00pm-11:00am ==> 1320
      // String str = "09:00am-10:00am";
      String str = "1:00pm-11:00am";
      System.out.println(countMinutes(str));
   }

}