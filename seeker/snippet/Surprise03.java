//date: 2023-06-08T16:44:12Z
//url: https://api.github.com/gists/a8894406f925681654b3c87f2b138e58
//owner: https://api.github.com/users/DevKnow

/*
  이수아
*/
import java.io.*;

public class Main {


    static final String newLine = "\r\n";
    static final String open = "<a href='#'>";
    static final String close = "</a>";

    public static void main(String[] args) throws IOException {

        long totalCount = 127;
        long pageIndex = -3;

        Pager pager =new Pager(totalCount);
        System.out.println(pager.html(pageIndex));
    }

    static class Pager{
        private final int countPerPage = 10;
        private final int pagePerOnce = 10;

        private long totalCount;

        public Pager(long totalCount){
            this.totalCount = totalCount;
        }

        public String html(long pageIndex){
            StringBuilder sb =new StringBuilder();

            // 총 페이지 수
            long totalPage = totalCount/countPerPage+(totalCount%countPerPage == 0 ? 0 : 1);
            if(pageIndex<=0){
                pageIndex = 1;
            }
            else if(pageIndex>totalPage)
                pageIndex = totalPage;

            final long start = pageIndex / pagePerOnce*pagePerOnce+1;
            final long end = Math.min(start+pagePerOnce -1, totalPage);

            sb.append(open).append("[").append("처음").append("]").append(close);
            sb.append(newLine).append(open).append("[").append("이전").append("]").append(close);
            sb.append(newLine);

            for(long i = start; i<=end; i++){
                sb.append(newLine).append(open);
                if(i == pageIndex){
                    sb.insert(sb.length()-1, " class='on'");
                }
                sb.append(i).append(close);
            }

            sb.append(newLine);
            sb.append(newLine).append(open).append("[").append("다음").append("]").append(close);
            sb.append(newLine).append(open).append("[").append("마지막").append("]").append(close);

            return sb.toString();
        }
    }
}