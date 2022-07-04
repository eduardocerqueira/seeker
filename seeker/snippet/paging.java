//date: 2022-07-04T03:02:38Z
//url: https://api.github.com/gists/d1f2465ea353cc79dbc6edcb543574c7
//owner: https://api.github.com/users/DoubleDG

package pj3;

public class Pager {
    StringBuffer buffer = new StringBuffer();

    long totalCount;
    long MaxPage = 10;

    long MaxPost = 10;


    public Pager(long totalCount) {
        
        this.totalCount = totalCount;
    }


    public long TotalPageIndex() {
        if (totalCount % MaxPost != 0)
            return totalCount / MaxPost + 1;


        return totalCount / MaxPost;

    }


    public long PrintFirstPage(long a) {
        if (a % MaxPage != 0)
            return (a / MaxPage) * MaxPage + 1;
        return a - (MaxPage - 1);
    }


    public long PrintLastPage(long a) {
        return PrintFirstPage(a) + MaxPage;
    }


    public String html(long pageIndex) {

        buffer.append("<a href='#'>[처음]</a>\n");
        buffer.append("<a href='#'>[이전]</a>\n");
        buffer.append("\n");

        for (long i = PrintFirstPage(pageIndex); i < PrintLastPage(pageIndex); i++) {

            if (i > TotalPageIndex())
                break;

            if (pageIndex == i) {
                buffer.append("<a href='#' class='on'> ");
                buffer.append(i);
                buffer.append(" </a>\n");
            } else {
                buffer.append("<a href='#'>");
                buffer.append(i);
                buffer.append(" </a>\n");
            }
        }

        buffer.append("\n");
        buffer.append("<a href='#'>[다음]</a>\n");
        buffer.append("<a href='#'>[마지막]</a>");
        return buffer.toString();
    }

}

class pj3 {
    public static void main(String[] args) {
        long totalCount = 127;
        long pageIndex = 1;

        Pager pager = new Pager(totalCount);
        System.out.println(pager.html(pageIndex));
    }
}
