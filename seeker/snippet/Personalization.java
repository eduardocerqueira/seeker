//date: 2023-03-06T16:46:41Z
//url: https://api.github.com/gists/bb6a76a285a6e741632d928acbaf7cad
//owner: https://api.github.com/users/techisbeautiful

// set currency preference
if(request.getParameter("set_currency") != null) {
  // get currency from form
  String currency = request.getParameter("currency");

  // store currency preference in session
  session.setAttribute("currency", currency);
}