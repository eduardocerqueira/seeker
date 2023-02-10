//date: 2023-02-10T16:46:45Z
//url: https://api.github.com/gists/2ab9813aacb6bcdc613a2d8dcd025814
//owner: https://api.github.com/users/Mikaeryu

//Task 4.1.6 https://stepik.org/lesson/498100/step/6?unit=489623

package org.stepic.java_func;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

class Main {
    public static void main(String[] args) throws ParseException {
        List<Comment> comments;
        comments = new ArrayList<>();

        comments.add(new Comment(
                CommentUtils.TEXT_FORMATTER.parse("14-03-2020 10:20:34"),
                "What a beautiful photo! Where is it?"
        ));
        comments.add(new Comment(
                CommentUtils.TEXT_FORMATTER.parse("16-03-2020 15:35:18"),
                "I do not know, I just found it on the internet!"
        ));
        comments.add(new Comment(
                CommentUtils.TEXT_FORMATTER.parse("20-03-2020 19:10:22"),
                "Is anyone here?"
        ));

        Date threshold = CommentUtils.TEXT_FORMATTER.parse("15-03-2020 00:00:00");
        int maxTextLength = 30; // it is just an example, do not rely on this number!

        CommentUtils.handleComments(comments, threshold, maxTextLength);
        CommentUtils.printComments(comments);
    }
}

final class CommentUtils {
    /**
     * An example string that fits the format "15-03-2020 10:20:34".
     * Use it to print the comments.
     */
    public static final SimpleDateFormat TEXT_FORMATTER = new SimpleDateFormat("dd-MM-yyyy HH:mm:ss");

    private CommentUtils() { }

    /**
     * It processes a given list of comments by removing old comments and shortening the text length
     */
    public static void handleComments(List<Comment> comments, Date thresholdDate, int maxTextLength) {
        //SOLUTION write your code here
        comments.removeIf(comment -> comment.getCreated().compareTo(thresholdDate) < 0);

        comments.replaceAll(comment -> {
            if (comment.getText().length() > maxTextLength) {
                String shortenedText = comment.getText().substring(0, maxTextLength);
                comment = new Comment(comment.getCreated(), shortenedText); //Замена коммента новым объектом с укороченным текстом
            }
            return comment;
        });
    }

    /**
     * It prints each comment in the following format:
     * [14-03-2020 10:20:34] What a beautiful photo! Where is it?
     * [16-03-2020 15:35:18] I do not know, I just found it on the internet!
     * [20-03-2020 19:10:22] Is anyone here?
     * Please, use the formatter above to fit the format.
     */
    public static void printComments(List<Comment> comments) {
        //SOLUTION write your code here
        SimpleDateFormat dateFormat = new SimpleDateFormat("[dd-MM-yyyy HH:mm:ss]");
        comments.forEach(comment -> System.out.println(dateFormat.format(comment.getCreated())+ " " + comment.getText()));
    }
}

class Comment {
    private final Date created;
    private final String text;

    public Comment(Date created, String text) {
        this.created = created;
        this.text = text;
    }

    public Date getCreated() {
        return created;
    }

    public String getText() {
        return text;
    }
}