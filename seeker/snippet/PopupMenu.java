//date: 2022-07-04T02:45:06Z
//url: https://api.github.com/gists/24ea9b12c1f0f73e7282d597178f89c4
//owner: https://api.github.com/users/thoinv

PopupMenu popupMenu = new PopupMenu(MainActivity.this, button);
                  
                  // Inflating popup menu from popup_menu.xml file
                popupMenu.getMenuInflater().inflate(R.menu.popup_menu, popupMenu.getMenu());
                popupMenu.setOnMenuItemClickListener(new PopupMenu.OnMenuItemClickListener() {
                    @Override
                    public boolean onMenuItemClick(MenuItem menuItem) {
                        // Toast message on menu item clicked
                        Toast.makeText(MainActivity.this, "You Clicked " + menuItem.getTitle(), Toast.LENGTH_SHORT).show();
                        return true;
                    }
                });
                // Showing the popup menu
                popupMenu.show();