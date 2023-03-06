//date: 2023-03-06T16:41:16Z
//url: https://api.github.com/gists/d3d2ffb3148275cc8ff62331b3c135ad
//owner: https://api.github.com/users/techisbeautiful

// start session
HttpSession session = request.getSession(true);

// check if user is logged in
if(session.getAttribute("user_id") != null) {
  // user is authenticated
  int user_id = (int) session.getAttribute("user_id");
} else {
  // user is not authenticated, redirect to login page
  response.sendRedirect("login.jsp");
  return;
}