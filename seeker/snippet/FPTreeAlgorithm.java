//date: 2025-11-14T16:44:22Z
//url: https://api.github.com/gists/e0851c27777ecf9f42ff7dec97e70419
//owner: https://api.github.com/users/metaloozee

import java.util.*;

/**
 * FP-Growth (Frequent Pattern Growth) Algorithm
 * 
 * FP-Growth is an efficient algorithm for mining frequent itemsets without
 * candidate generation (unlike Apriori). It uses a compact tree structure
 * called FP-Tree to represent the database and mines patterns recursively.
 * 
 * Key advantages over Apriori:
 * - No candidate generation (more efficient)
 * - Compressed representation of database
 * - Only scans database twice
 * 
 * Algorithm Steps:
 * 1. Scan database to find frequent 1-itemsets
 * 2. Build FP-Tree (compressed representation)
 * 3. Mine FP-Tree recursively to find all frequent patterns
 * 
 * Example Domain: Market basket analysis (finding products frequently bought together)
 */
public class FPTreeAlgorithm {
    
    /**
     * Node in the FP-Tree structure.
     * Each node represents an item and tracks its count and links.
     */
    static class FPNode {
        String item;
        int count;
        FPNode parent;
        List<FPNode> children;
        FPNode nextSameItem; // Link to next node with same item (for header table)
        
        public FPNode(String item, int count, FPNode parent) {
            this.item = item;
            this.count = count;
            this.parent = parent;
            this.children = new ArrayList<>();
            this.nextSameItem = null;
        }
        
        /**
         * Find or create a child node with the given item.
         */
        public FPNode findChild(String item) {
            for (FPNode child : children) {
                if (child.item.equals(item)) {
                    return child;
                }
            }
            return null;
        }
    }
    
    /**
     * Header table entry for efficient tree traversal.
     * Maintains a linked list of all nodes with the same item.
     */
    static class HeaderEntry {
        String item;
        int frequency;
        FPNode firstNode; // Head of linked list of nodes
        
        public HeaderEntry(String item, int frequency) {
            this.item = item;
            this.frequency = frequency;
            this.firstNode = null;
        }
    }
    
    private int minSupport; // Minimum support threshold
    private Map<String, Integer> itemFrequency; // Frequency of each item
    private List<HeaderEntry> headerTable; // Header table for FP-Tree
    private FPNode root; // Root of FP-Tree
    private List<Set<String>> frequentPatterns; // Discovered frequent patterns
    
    public FPTreeAlgorithm(int minSupport) {
        this.minSupport = minSupport;
        this.itemFrequency = new HashMap<>();
        this.headerTable = new ArrayList<>();
        this.frequentPatterns = new ArrayList<>();
    }
    
    /**
     * Main method to find all frequent patterns in the transaction database.
     * 
     * @param transactions List of transactions, where each transaction is a set of items
     * @return List of frequent itemsets
     */
    public List<Set<String>> findFrequentPatterns(List<Set<String>> transactions) {
        System.out.println("Step 1: Scanning database to find frequent items...");
        
        // First scan: Count frequency of each item
        for (Set<String> transaction : transactions) {
            for (String item : transaction) {
                itemFrequency.put(item, itemFrequency.getOrDefault(item, 0) + 1);
            }
        }
        
        // Remove infrequent items
        itemFrequency.entrySet().removeIf(entry -> entry.getValue() < minSupport);
        
        System.out.println("Found " + itemFrequency.size() + " frequent items");
        System.out.println("Frequent items: " + itemFrequency);
        
        if (itemFrequency.isEmpty()) {
            System.out.println("No frequent items found!");
            return frequentPatterns;
        }
        
        // Create header table sorted by frequency (descending)
        createHeaderTable();
        
        System.out.println("\nStep 2: Building FP-Tree...");
        
        // Build FP-Tree
        buildFPTree(transactions);
        
        System.out.println("FP-Tree built successfully");
        
        System.out.println("\nStep 3: Mining FP-Tree for frequent patterns...");
        
        // Mine patterns recursively
        mineTree(new HashSet<>(), root, headerTable);
        
        System.out.println("Found " + frequentPatterns.size() + " frequent patterns");
        
        return frequentPatterns;
    }
    
    /**
     * Create header table sorted by item frequency (descending order).
     * This ordering ensures better tree compression.
     */
    private void createHeaderTable() {
        for (Map.Entry<String, Integer> entry : itemFrequency.entrySet()) {
            headerTable.add(new HeaderEntry(entry.getKey(), entry.getValue()));
        }
        
        // Sort by frequency (descending)
        headerTable.sort((a, b) -> Integer.compare(b.frequency, a.frequency));
    }
    
    /**
     * Build the FP-Tree from transactions.
     * The tree compresses the database by sharing common prefixes.
     */
    private void buildFPTree(List<Set<String>> transactions) {
        root = new FPNode("root", 0, null);
        
        for (Set<String> transaction : transactions) {
            // Filter and sort items by frequency order
            List<String> sortedItems = new ArrayList<>();
            for (HeaderEntry entry : headerTable) {
                if (transaction.contains(entry.item)) {
                    sortedItems.add(entry.item);
                }
            }
            
            // Insert sorted transaction into tree
            if (!sortedItems.isEmpty()) {
                insertTransaction(sortedItems, root);
            }
        }
    }
    
    /**
     * Insert a transaction into the FP-Tree.
     * Reuses existing paths when possible (tree compression).
     */
    private void insertTransaction(List<String> items, FPNode node) {
        if (items.isEmpty()) {
            return;
        }
        
        String firstItem = items.get(0);
        FPNode child = node.findChild(firstItem);
        
        if (child != null) {
            // Path exists, increment count
            child.count++;
        } else {
            // Create new node
            child = new FPNode(firstItem, 1, node);
            node.children.add(child);
            
            // Link to header table
            linkToHeaderTable(child);
        }
        
        // Recursively insert remaining items
        if (items.size() > 1) {
            insertTransaction(items.subList(1, items.size()), child);
        }
    }
    
    /**
     * Link a new node to the header table's linked list.
     * This allows efficient traversal of all nodes with the same item.
     */
    private void linkToHeaderTable(FPNode node) {
        for (HeaderEntry entry : headerTable) {
            if (entry.item.equals(node.item)) {
                if (entry.firstNode == null) {
                    entry.firstNode = node;
                } else {
                    // Find last node in chain
                    FPNode current = entry.firstNode;
                    while (current.nextSameItem != null) {
                        current = current.nextSameItem;
                    }
                    current.nextSameItem = node;
                }
                break;
            }
        }
    }
    
    /**
     * Recursively mine the FP-Tree to find all frequent patterns.
     * Uses a divide-and-conquer approach, mining conditional FP-Trees.
     * 
     * @param suffix Current pattern being extended
     * @param node Current tree node
     * @param localHeaderTable Header table for current tree
     */
    private void mineTree(Set<String> suffix, FPNode node, List<HeaderEntry> localHeaderTable) {
        // Process items in reverse frequency order (bottom-up)
        for (int i = localHeaderTable.size() - 1; i >= 0; i--) {
            HeaderEntry entry = localHeaderTable.get(i);
            
            // Create new pattern by adding this item to suffix
            Set<String> newPattern = new HashSet<>(suffix);
            newPattern.add(entry.item);
            
            // Calculate support by traversing linked list
            int support = 0;
            FPNode current = entry.firstNode;
            while (current != null) {
                support += current.count;
                current = current.nextSameItem;
            }
            
            // If pattern is frequent, add it
            if (support >= minSupport) {
                frequentPatterns.add(new HashSet<>(newPattern));
            }
            
            // Build conditional pattern base
            List<List<String>> conditionalPatternBase = new ArrayList<>();
            List<Integer> counts = new ArrayList<>();
            
            current = entry.firstNode;
            while (current != null) {
                List<String> path = new ArrayList<>();
                FPNode ancestor = current.parent;
                
                // Trace path from current node to root
                while (ancestor != null && ancestor.item != null && !ancestor.item.equals("root")) {
                    path.add(0, ancestor.item);
                    ancestor = ancestor.parent;
                }
                
                if (!path.isEmpty()) {
                    conditionalPatternBase.add(path);
                    counts.add(current.count);
                }
                
                current = current.nextSameItem;
            }
            
            // If conditional pattern base is not empty, build conditional FP-Tree
            if (!conditionalPatternBase.isEmpty()) {
                // Count items in conditional pattern base
                Map<String, Integer> conditionalItemFreq = new HashMap<>();
                for (int j = 0; j < conditionalPatternBase.size(); j++) {
                    List<String> pattern = conditionalPatternBase.get(j);
                    int count = counts.get(j);
                    for (String item : pattern) {
                        conditionalItemFreq.put(item, 
                            conditionalItemFreq.getOrDefault(item, 0) + count);
                    }
                }
                
                // Remove infrequent items
                conditionalItemFreq.entrySet().removeIf(e -> e.getValue() < minSupport);
                
                if (!conditionalItemFreq.isEmpty()) {
                    // Create new header table for conditional tree
                    List<HeaderEntry> newHeaderTable = new ArrayList<>();
                    for (Map.Entry<String, Integer> e : conditionalItemFreq.entrySet()) {
                        newHeaderTable.add(new HeaderEntry(e.getKey(), e.getValue()));
                    }
                    newHeaderTable.sort((a, b) -> Integer.compare(b.frequency, a.frequency));
                    
                    // Build conditional FP-Tree
                    FPNode conditionalRoot = new FPNode("root", 0, null);
                    for (int j = 0; j < conditionalPatternBase.size(); j++) {
                        List<String> pattern = conditionalPatternBase.get(j);
                        List<String> filteredPattern = new ArrayList<>();
                        for (HeaderEntry he : newHeaderTable) {
                            if (pattern.contains(he.item)) {
                                filteredPattern.add(he.item);
                            }
                        }
                        
                        if (!filteredPattern.isEmpty()) {
                            insertTransactionWithCount(filteredPattern, conditionalRoot, 
                                counts.get(j), newHeaderTable);
                        }
                    }
                    
                    // Recursively mine conditional tree
                    mineTree(newPattern, conditionalRoot, newHeaderTable);
                }
            }
        }
    }
    
    /**
     * Insert transaction with a specific count (for conditional trees).
     */
    private void insertTransactionWithCount(List<String> items, FPNode node, 
                                           int count, List<HeaderEntry> localHeaderTable) {
        if (items.isEmpty()) {
            return;
        }
        
        String firstItem = items.get(0);
        FPNode child = node.findChild(firstItem);
        
        if (child != null) {
            child.count += count;
        } else {
            child = new FPNode(firstItem, count, node);
            node.children.add(child);
            linkToLocalHeaderTable(child, localHeaderTable);
        }
        
        if (items.size() > 1) {
            insertTransactionWithCount(items.subList(1, items.size()), child, 
                count, localHeaderTable);
        }
    }
    
    private void linkToLocalHeaderTable(FPNode node, List<HeaderEntry> localHeaderTable) {
        for (HeaderEntry entry : localHeaderTable) {
            if (entry.item.equals(node.item)) {
                if (entry.firstNode == null) {
                    entry.firstNode = node;
                } else {
                    FPNode current = entry.firstNode;
                    while (current.nextSameItem != null) {
                        current = current.nextSameItem;
                    }
                    current.nextSameItem = node;
                }
                break;
            }
        }
    }
    
    public static void main(String[] args) {
        System.out.println("FP-Growth Algorithm: Frequent Pattern Mining");
        System.out.println("=" .repeat(60) + "\n");
        
        // Dummy dataset: Supermarket transactions
        // Each set represents items bought together in one transaction
        List<Set<String>> transactions = Arrays.asList(
            new HashSet<>(Arrays.asList("Bread", "Milk", "Eggs")),
            new HashSet<>(Arrays.asList("Bread", "Butter", "Milk")),
            new HashSet<>(Arrays.asList("Bread", "Butter")),
            new HashSet<>(Arrays.asList("Bread", "Milk", "Butter", "Eggs")),
            new HashSet<>(Arrays.asList("Milk", "Eggs")),
            new HashSet<>(Arrays.asList("Bread", "Milk")),
            new HashSet<>(Arrays.asList("Bread", "Eggs")),
            new HashSet<>(Arrays.asList("Butter", "Eggs", "Milk")),
            new HashSet<>(Arrays.asList("Bread", "Butter", "Milk")),
            new HashSet<>(Arrays.asList("Bread", "Milk", "Eggs"))
        );
        
        System.out.println("Transaction Database:");
        for (int i = 0; i < transactions.size(); i++) {
            System.out.println("  Transaction " + (i+1) + ": " + transactions.get(i));
        }
        
        int minSupport = 3; // Item must appear in at least 3 transactions
        System.out.println("\nMinimum Support: " + minSupport + " transactions");
        System.out.println("\n" + "=".repeat(60) + "\n");
        
        // Run FP-Growth algorithm
        FPTreeAlgorithm fpGrowth = new FPTreeAlgorithm(minSupport);
        List<Set<String>> frequentPatterns = fpGrowth.findFrequentPatterns(transactions);
        
        // Display results
        System.out.println("\n" + "=".repeat(60));
        System.out.println("FREQUENT PATTERNS DISCOVERED:");
        System.out.println("=".repeat(60));
        
        // Sort patterns by size for better readability
        frequentPatterns.sort((a, b) -> {
            if (a.size() != b.size()) {
                return Integer.compare(a.size(), b.size());
            }
            return a.toString().compareTo(b.toString());
        });
        
        // Group by pattern size
        Map<Integer, List<Set<String>>> patternsBySize = new HashMap<>();
        for (Set<String> pattern : frequentPatterns) {
            patternsBySize.computeIfAbsent(pattern.size(), k -> new ArrayList<>()).add(pattern);
        }
        
        for (int size = 1; size <= patternsBySize.keySet().stream().max(Integer::compare).orElse(0); size++) {
            if (patternsBySize.containsKey(size)) {
                System.out.println("\n" + size + "-itemsets:");
                for (Set<String> pattern : patternsBySize.get(size)) {
                    System.out.println("  " + pattern);
                }
            }
        }
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("INSIGHTS:");
        System.out.println("- These patterns represent items frequently bought together");
        System.out.println("- Can be used for product placement and cross-selling");
        System.out.println("- Larger itemsets indicate strong purchase associations");
    }
}