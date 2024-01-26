//date: 2024-01-26T17:06:59Z
//url: https://api.github.com/gists/bc65c570503566bb67f9cc64382ee289
//owner: https://api.github.com/users/delta-dev-software

public class Main {
    public static void main(String[] args) {
        // Create an EntityManagerFactory
        EntityManagerFactory emf = Persistence.createEntityManagerFactory("examplePU");

        // Create an EntityManager
        EntityManager em = emf.createEntityManager();

        // Create a new book entity
        Book book = new Book();
        book.setId(1L);
        book.setTitle("Java Persistence with JPA");
        book.setAuthor("John Doe");

        // Persist the book entity (CREATE)
        EntityTransaction transaction = em.getTransaction();
        transaction.begin();
        em.persist(book);
        transaction.commit();

        // Find a book by ID (READ)
        Book retrievedBook = em.find(Book.class, 1L);
        System.out.println("Retrieved Book: " + retrievedBook.getTitle());

        // Update the book entity (UPDATE)
        transaction.begin();
        retrievedBook.setAuthor("Jane Doe");
        transaction.commit();

        // Remove the book entity (DELETE)
        transaction.begin();
        em.remove(retrievedBook);
        transaction.commit();

        // Close the EntityManager and EntityManagerFactory
        em.close();
        emf.close();
    }
}