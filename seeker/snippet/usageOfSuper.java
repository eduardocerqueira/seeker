//date: 2022-09-20T17:10:41Z
//url: https://api.github.com/gists/da033d4134ef27ea22ea169a11bc5201
//owner: https://api.github.com/users/BetterProgramming

void usageOfSuper() {
    List<Father> fathers = new ArrayList<>();
    List<GrandChild> grandChildren = new ArrayList<>();
    copy(fathers, grandChildren);

    // Using the List<Father> after copy(). We can guarantee that Father can be extracted safely, as contravariance
    // would not have allowed any thing above Child to get added. If it would have allowed
    // GrandFather to be added, this usage code would have crashed!!
    for(Father f : fathers) { // GrandFather to father crash would // have happened!!!
        System.out.println(f.toString());
    }
}

void copy(List<? super Child> dest, List<? extends Child> source) { }