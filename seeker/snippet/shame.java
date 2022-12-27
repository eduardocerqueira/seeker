//date: 2022-12-27T16:31:02Z
//url: https://api.github.com/gists/713eed060c60e3c82deeb78c7a4323fd
//owner: https://api.github.com/users/schipplock

if (autosize) {
    int height = getHeight() + getRootPane().getHeight() + itemPanel.getHeight() + actionPanel.getHeight();
    int width = itemPanel.getWidth() + 40;
    setPreferredSize(new Dimension(width, height));
    pack();
    autosize = false;
}