//date: 2023-11-24T16:42:46Z
//url: https://api.github.com/gists/270523a17bf412e9388b0820ea601307
//owner: https://api.github.com/users/Dima-Skrypniuk

package org.example;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class Book {
    private String title;
    private String author;
    private String genre;
    private int year;


    public Book(String title, String author, String genre, int year) {
        this.title = title;
        this.author = author;
        this.genre = genre;
        this.year = year;
    }

    public String getTitle() {
        return title;
    }

    public String getAuthor() {
        return author;
    }

    public String getGenre() {
        return genre;
    }

    public int getYear() {
        return year;
    }

    @Override
    public String toString() {
        return "Book{" +
                "title='" + title + '\'' +
                ", author='" + author + '\'' +
                ", genre='" + genre + '\'' +
                ", year=" + year +
                '}';
    }
}

public class BookCollectionOperations {

    public static void main(String[] args) {
        // Create ArrayList
        List<Book> bookCollection1 = new ArrayList<>();
        List<Book> bookCollection2 = new ArrayList<>();

        // Populate book
        bookCollection1.add(new Book("Book1", "Author1", "Fiction", 2000));
        bookCollection1.add(new Book("Book2", "Author2", "Non-Fiction", 2010));
        bookCollection1.add(new Book("Book3", "Author3", "Fantasy", 2020));

        bookCollection2.add(new Book("Book4", "Author4", "Mystery", 2005));
        bookCollection2.add(new Book("Book5", "Author5", "Fiction", 2015));
        bookCollection2.add(new Book("Book6", "Author6", "Science Fiction", 2022));

        // 1. Print List of Authors
        System.out.println("1. List of all authors in the collection:");
        listAuthors(bookCollection1);

        // 2. List Authors by Genre
        System.out.println("\n2. List of authors who have written books in 'Fiction' genre:");
        listAuthorsByGenre(bookCollection1, "Fiction");

        // 3. List Authors by Publication Year
        System.out.println("\n3. List of authors whose books were published in the year 2010:");
        listAuthorsByPublicationYear(bookCollection1, 2010);

        // 4. Find Book by Author
        System.out.println("\n4. Find a book by 'Author2':");
        findBookByAuthor(bookCollection1, "Author2");

        // 5. Find Books by Publication Year
        System.out.println("\n5. Find books written in the year 2020:");
        findBooksByPublicationYear(bookCollection1, 2020);

        // 6. Find Books by Genre
        System.out.println("\n6. Find books in the 'Science Fiction' genre:");
        findBooksByGenre(bookCollection1, "Science Fiction");

        // 7. Remove Books by Author
        System.out.println("\n7. Remove books written by 'Author3':");
        removeBooksByAuthor(bookCollection1, "Author3");
        System.out.println("Remaining books after removal:");
        printBookCollection(bookCollection1);

        // 8. Sort Collection by Criterion
        System.out.println("\n8. Sort collection by title:");
        sortCollectionByCriterion(bookCollection1, "title");
        printBookCollection(bookCollection1);

        // 9. Merge Book Collections
        System.out.println("\n9. Merge book collections:");
        List<Book> mergedCollection = mergeBookCollections(bookCollection1, bookCollection2);
        printBookCollection(mergedCollection);

        // 10. Subcollection of Books by Genre
        System.out.println("\n10. Create a subcollection of books from 'Fiction' genre:");
        List<Book> subCollection = getSubcollectionByGenre(bookCollection1, "Fiction");
        printBookCollection(subCollection);
    }

    // 1. Print List of Authors
    private static void listAuthors(List<Book> bookCollection) {
        bookCollection.forEach(book -> System.out.println(book.getAuthor()));
    }

    // 2. List Authors by Genre
    private static void listAuthorsByGenre(List<Book> bookCollection, String genre) {
        bookCollection.stream()
                .filter(book -> book.getGenre().equals(genre))
                .map(Book::getAuthor)
                .distinct()
                .forEach(System.out::println);
    }

    // 3. List Authors by Publication Year
    private static void listAuthorsByPublicationYear(List<Book> bookCollection, int year) {
        bookCollection.stream()
                .filter(book -> book.getYear() == year)
                .map(Book::getAuthor)
                .distinct()
                .forEach(System.out::println);
    }

    // 4. Find Book by Author
    private static void findBookByAuthor(List<Book> bookCollection, String author) {
        bookCollection.stream()
                .filter(book -> book.getAuthor().equals(author))
                .findFirst()
                .ifPresent(System.out::println);
    }

    // 5. Find Books by Publication Year
    private static void findBooksByPublicationYear(List<Book> bookCollection, int year) {
        bookCollection.stream()
                .filter(book -> book.getYear() == year)
                .forEach(System.out::println);
    }

    // 6. Find Books by Genre
    private static void findBooksByGenre(List<Book> bookCollection, String genre) {
        bookCollection.stream()
                .filter(book -> book.getGenre().equals(genre))
                .forEach(System.out::println);
    }

    // 7. Remove Books by Author
    private static void removeBooksByAuthor(List<Book> bookCollection, String author) {
        bookCollection.removeIf(book -> book.getAuthor().equals(author));
    }

    // 8. Sort Collection by Criterion
    private static void sortCollectionByCriterion(List<Book> bookCollection, String criterion) {
        switch (criterion) {
            case "title":
                Collections.sort(bookCollection, (b1, b2) -> b1.getTitle().compareTo(b2.getTitle()));
                break;
            case "author":
                Collections.sort(bookCollection, (b1, b2) -> b1.getAuthor().compareTo(b2.getAuthor()));
                break;
            case "year":
                Collections.sort(bookCollection, (b1, b2) -> Integer.compare(b1.getYear(), b2.getYear()));
                break;
            default:
                System.out.println("Invalid criterion for sorting.");
        }
    }

    // 9. Merge Book Collections
    private static List<Book> mergeBookCollections(List<Book> collection1, List<Book> collection2) {
        List<Book> mergedCollection = new ArrayList<>(collection1);
        mergedCollection.addAll(collection2);
        return mergedCollection;
    }

    // 10. Subcollection of Books by Genre
    private static List<Book> getSubcollectionByGenre(List<Book> bookCollection, String genre) {
        return bookCollection.stream()
                .filter(book -> book.getGenre().equals(genre))
                .toList();
    }

    // Helper method to print the book collection
    private static void printBookCollection(List<Book> bookCollection) {
        bookCollection.forEach(System.out::println);
    }
}

