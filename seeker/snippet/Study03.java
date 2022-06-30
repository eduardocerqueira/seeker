//date: 2022-06-30T21:32:08Z
//url: https://api.github.com/gists/85fc685f012094f3c3f1a792e9c33d43
//owner: https://api.github.com/users/nuca9

public class Study03 {

    public static void main(String[] args) {

        long totalCount = 127;    // 총 글의 개수
        long pageIndex = 1;      //현재 페이지

        Pager pager = new Pager(totalCount);
        System.out.println(pager.html(pageIndex));
    }
}

class Pager {
    static long totalCount;
    static long pageIndex;
    static final int PAGE_DISPLAY = 10;     //게시판 글 목록 개수
    static final int PAGE_NAVIGATION_BLOCK = 10;    //페이지 개수

    public Pager(long totalCount) {
        this.totalCount = totalCount;
    }

    public static long totalPage() {    //총 페이지 번호
        long totalPage = totalCount / PAGE_DISPLAY;

        if ((totalCount % PAGE_DISPLAY) != 0) {
            totalPage++;
        }
        return totalPage;
    }

    public static long startPage(long pageIndex) {
        //현재 페이지의 시작 페이지 번호
        long startPage = ((pageIndex - 1) / PAGE_NAVIGATION_BLOCK) * PAGE_NAVIGATION_BLOCK + 1;

        return startPage;
    }

    public static long endPage(long startPage) {
        //현재 페이지의 마지막 페이지 번호
        long endPage = (startPage + PAGE_NAVIGATION_BLOCK - 1);

        if (endPage > totalPage()) {
            endPage = totalPage();
        }

        return endPage;
    }

    public String html(long pageIndex) {

        StringBuffer sb = new StringBuffer();

        sb.append("<a href='#'>[처음]</a>").append(System.lineSeparator())
                .append("<a href='#'>[이전]</a>").append(System.lineSeparator())
                .append(System.lineSeparator());

        for (long i = startPage(pageIndex); i <= endPage(startPage(pageIndex)); i++) {
            if (pageIndex == i) {
                sb.append("<a href='#' class='on'>" + i + "</a>").append(System.lineSeparator());
            } else {
                sb.append("<a href='#'>" + i + "</a>").append(System.lineSeparator());
            }
        }

        sb.append(System.lineSeparator())
                .append("<a href='#'>[다음]</a>").append(System.lineSeparator())
                .append("<a href='#'>[마지막]</a>").append("<br>");

        return sb.toString();
    }
}