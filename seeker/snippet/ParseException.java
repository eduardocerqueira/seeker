//date: 2023-02-20T16:37:35Z
//url: https://api.github.com/gists/0f557e9d1a86c6f7cedfb35e338ba16a
//owner: https://api.github.com/users/charleslivelyphd

SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
try {
  Date date = dateFormat.parse("2023-02-20");
  // do something with the date
} catch (ParseException e) {
  e.printStackTrace();
  // handle the exception here, such as displaying an error message to the user or logging the error
}
