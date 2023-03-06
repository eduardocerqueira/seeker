//date: 2023-03-06T16:44:33Z
//url: https://api.github.com/gists/2ada0028218fa73814b6e3a36da6d02c
//owner: https://api.github.com/users/techisbeautiful

// add item to cart
if(request.getParameter("add_to_cart") != null) {
  // get product ID from form
  int product_id = Integer.parseInt(request.getParameter("product_id"));

  // add product to cart
  HashMap<Integer, Integer> cart = (HashMap<Integer, Integer>) session.getAttribute("cart");
  if(cart == null) {
    cart = new HashMap<Integer, Integer>();
  }
  if(cart.containsKey(product_id)) {
    cart.put(product_id, cart.get(product_id) + 1);
  } else {
    cart.put(product_id, 1);
  }
  session.setAttribute("cart", cart);
}