//date: 2023-10-10T17:06:18Z
//url: https://api.github.com/gists/fa555d7d47498d53ac6dfa8b14426fb5
//owner: https://api.github.com/users/swayam-learning

package Projects;

import java.util.*;

class Expense {
    private Date date;
    private double amount;
    private String category;
    private String description;

    public Expense(Date date, double amount, String category, String description) {
        this.date = date;
        this.amount = amount;
        this.category = category;
        this.description = description;
    }

    public double getAmount() {
        return amount;
    }

    public String getCategory() {
        return category;
    }

    public String getDescription() {
        return description;
    }

    public Date getDate() {
        return date;
    }

    @Override
    public String toString() {
        return "Expense{" +
                "date=" + date +
                ", amount=" + amount +
                ", category='" + category + '\'' +
                ", description='" + description + '\'' +
                '}';
    }
}

class ExpenseTracker {
    private List<Expense> expenses;
    private Map<String, LinkedList<Expense>> expenseCategories;

    public ExpenseTracker() {
        expenses = new LinkedList<>();
        expenseCategories = new HashMap<>();
    }

    public void addExpense(Expense expense) {
        expenses.add(expense);

        // Categorize the expense
        String category = expense.getCategory();
        LinkedList<Expense> categoryExpenses = expenseCategories.getOrDefault(category, new LinkedList<>());
        categoryExpenses.add(expense);
        expenseCategories.put(category, categoryExpenses);
    }

    public List<Expense> viewExpensesByCategory(String category) {
        return expenseCategories.getOrDefault(category, new LinkedList<>());
    }

    public double getTotalExpenditure() {
        return expenses.stream().mapToDouble(Expense::getAmount).sum();
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        ExpenseTracker tracker = new ExpenseTracker();

        while (true) {
            try {
                System.out.print("Enter expense amount (0 to exit): $");
                double amount = scanner.nextDouble();
                scanner.nextLine();  // Consume the newline

                if (amount == 0)
                    break;

                System.out.print("Enter expense category: ");
                String category = scanner.nextLine();

                System.out.print("Enter expense description: ");
                String description = scanner.nextLine();

                Expense expense = new Expense(new Date(), amount, category, description);
                tracker.addExpense(expense);
            } catch (InputMismatchException e) {
                System.out.println("Invalid input. Please enter a valid amount (a number).");
                scanner.nextLine();  // Consume the invalid input to prevent an infinite loop
            }
        }

        System.out.println("\nExpense Summary:");
        for (Expense expense : tracker.expenses) {
            System.out.println(expense);
        }

        System.out.println("\nTotal Expenditure: $" + tracker.getTotalExpenditure());
    }
}