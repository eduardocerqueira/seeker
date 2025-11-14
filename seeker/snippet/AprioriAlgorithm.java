//date: 2025-11-14T16:44:22Z
//url: https://api.github.com/gists/e0851c27777ecf9f42ff7dec97e70419
//owner: https://api.github.com/users/metaloozee

import java.util.*;

/**
 * Apriori Algorithm for Association Rule Mining
 * 
 * The Apriori algorithm discovers association rules in transactional databases.
 * It finds frequent itemsets (sets of items that appear together frequently)
 * and generates rules showing relationships between items.
 * 
 * Key Principle: Apriori Property
 * - If an itemset is frequent, all its subsets must also be frequent
 * - If an itemset is infrequent, all its supersets must be infrequent
 * 
 * Algorithm Steps:
 * 1. Find frequent 1-itemsets (single items)
 * 2. Generate candidate k-itemsets from frequent (k-1)-itemsets
 * 3. Scan database to count support of candidates
 * 4. Keep only frequent k-itemsets
 * 5. Repeat until no more frequent itemsets can be found
 * 6. Generate association rules from frequent itemsets
 * 
 * Example Domain: Market basket analysis (retail store transactions)
 */
public class AprioriAlgorithm {
    
    /**
     * Represents an itemset (a set of items that appear together).
     */
    static class Itemset {
        Set<String> items;
        int support; // How many transactions contain this itemset
        
        public Itemset(Set<String> items, int support) {
            this.items = new HashSet<>(items);
            this.support = support;
        }
        
        public Itemset(Set<String> items) {
            this(items, 0);
        }
        
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Itemset itemset = (Itemset) o;
            return items.equals(itemset.items);
        }
        
        @Override
        public int hashCode() {
            return items.hashCode();
        }
        
        @Override
        public String toString() {
            List<String> sorted = new ArrayList<>(items);
            Collections.sort(sorted);
            return sorted.toString();
        }
    }
    
    /**
     * Represents an association rule: antecedent => consequent
     * Example: {Bread, Butter} => {Milk} means "if customer buys bread and butter,
     * they're likely to buy milk"
     */
    static class AssociationRule {
        Set<String> antecedent; // Left side (if part)
        Set<String> consequent; // Right side (then part)
        double confidence;       // P(consequent | antecedent)
        double lift;            // How much more likely consequent is when antecedent is present
        int support;            // Number of transactions containing both
        
        public AssociationRule(Set<String> antecedent, Set<String> consequent, 
                              double confidence, double lift, int support) {
            this.antecedent = antecedent;
            this.consequent = consequent;
            this.confidence = confidence;
            this.lift = lift;
            this.support = support;
        }
        
        @Override
        public String toString() {
            return String.format("%s => %s (Confidence: %.2f%%, Lift: %.2f, Support: %d)",
                new ArrayList<>(antecedent), new ArrayList<>(consequent), 
                confidence * 100, lift, support);
        }
    }
    
    private int minSupport;      // Minimum support count
    private double minConfidence; // Minimum confidence (0-1)
    private List<Set<String>> transactions; // Transaction database
    private Map<Integer, List<Itemset>> frequentItemsets; // Frequent itemsets by size
    
    public AprioriAlgorithm(int minSupport, double minConfidence) {
        this.minSupport = minSupport;
        this.minConfidence = minConfidence;
        this.frequentItemsets = new HashMap<>();
    }
    
    /**
     * Run the Apriori algorithm to find frequent itemsets and generate rules.
     * 
     * @param transactions List of transactions (each transaction is a set of items)
     * @return List of association rules
     */
    public List<AssociationRule> run(List<Set<String>> transactions) {
        this.transactions = transactions;
        
        System.out.println("Step 1: Finding frequent itemsets...");
        
        // Find all frequent itemsets
        findFrequentItemsets();
        
        System.out.println("\nStep 2: Generating association rules...");
        
        // Generate association rules from frequent itemsets
        List<AssociationRule> rules = generateAssociationRules();
        
        return rules;
    }
    
    /**
     * Find all frequent itemsets using the Apriori algorithm.
     * Starts with 1-itemsets and grows larger itemsets level by level.
     */
    private void findFrequentItemsets() {
        int k = 1; // Current itemset size
        
        // Step 1: Find frequent 1-itemsets
        List<Itemset> frequentK = findFrequent1Itemsets();
        frequentItemsets.put(k, frequentK);
        
        System.out.println("  Found " + frequentK.size() + " frequent " + k + "-itemsets");
        
        // Continue finding larger itemsets until no more can be found
        while (!frequentK.isEmpty()) {
            k++;
            
            // Generate candidate k-itemsets from frequent (k-1)-itemsets
            List<Itemset> candidateK = generateCandidates(frequentK, k);
            
            // Count support for each candidate
            countSupport(candidateK);
            
            // Keep only frequent candidates
            frequentK = new ArrayList<>();
            for (Itemset candidate : candidateK) {
                if (candidate.support >= minSupport) {
                    frequentK.add(candidate);
                }
            }
            
            if (!frequentK.isEmpty()) {
                frequentItemsets.put(k, frequentK);
                System.out.println("  Found " + frequentK.size() + " frequent " + k + "-itemsets");
            }
        }
    }
    
    /**
     * Find all frequent 1-itemsets (individual items that meet minimum support).
     */
    private List<Itemset> findFrequent1Itemsets() {
        Map<String, Integer> itemCount = new HashMap<>();
        
        // Count occurrence of each item
        for (Set<String> transaction : transactions) {
            for (String item : transaction) {
                itemCount.put(item, itemCount.getOrDefault(item, 0) + 1);
            }
        }
        
        // Create itemsets for frequent items
        List<Itemset> frequent = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : itemCount.entrySet()) {
            if (entry.getValue() >= minSupport) {
                Set<String> items = new HashSet<>();
                items.add(entry.getKey());
                frequent.add(new Itemset(items, entry.getValue()));
            }
        }
        
        return frequent;
    }
    
    /**
     * Generate candidate k-itemsets from frequent (k-1)-itemsets.
     * Uses the Apriori property: combines two (k-1)-itemsets that share
     * (k-2) common items to create a k-itemset candidate.
     */
    private List<Itemset> generateCandidates(List<Itemset> frequentPrev, int k) {
        List<Itemset> candidates = new ArrayList<>();
        
        // Try combining every pair of frequent (k-1)-itemsets
        for (int i = 0; i < frequentPrev.size(); i++) {
            for (int j = i + 1; j < frequentPrev.size(); j++) {
                Set<String> union = new HashSet<>(frequentPrev.get(i).items);
                union.addAll(frequentPrev.get(j).items);
                
                // If union has exactly k items, it's a valid candidate
                if (union.size() == k) {
                    Itemset candidate = new Itemset(union);
                    
                    // Prune using Apriori property: all (k-1)-subsets must be frequent
                    if (hasInfrequentSubset(candidate, frequentPrev)) {
                        continue; // Skip this candidate
                    }
                    
                    // Avoid duplicates
                    if (!candidates.contains(candidate)) {
                        candidates.add(candidate);
                    }
                }
            }
        }
        
        return candidates;
    }
    
    /**
     * Check if any (k-1)-subset of the candidate is infrequent.
     * This is the pruning step that makes Apriori efficient.
     */
    private boolean hasInfrequentSubset(Itemset candidate, List<Itemset> frequentPrev) {
        // Generate all (k-1)-subsets
        for (String item : candidate.items) {
            Set<String> subset = new HashSet<>(candidate.items);
            subset.remove(item);
            
            Itemset subsetItemset = new Itemset(subset);
            if (!frequentPrev.contains(subsetItemset)) {
                return true; // Found an infrequent subset
            }
        }
        return false;
    }
    
    /**
     * Count the support (number of transactions) for each candidate itemset.
     * This requires scanning the transaction database.
     */
    private void countSupport(List<Itemset> candidates) {
        for (Itemset candidate : candidates) {
            int count = 0;
            for (Set<String> transaction : transactions) {
                // Check if transaction contains all items in candidate
                if (transaction.containsAll(candidate.items)) {
                    count++;
                }
            }
            candidate.support = count;
        }
    }
    
    /**
     * Generate association rules from frequent itemsets.
     * For each frequent itemset, try all possible ways to split it into
     * antecedent and consequent, and calculate confidence and lift.
     */
    private List<AssociationRule> generateAssociationRules() {
        List<AssociationRule> rules = new ArrayList<>();
        
        // Only generate rules from itemsets with 2+ items
        for (int k = 2; k <= frequentItemsets.size(); k++) {
            if (!frequentItemsets.containsKey(k)) continue;
            
            for (Itemset itemset : frequentItemsets.get(k)) {
                // Try all possible splits of the itemset
                rules.addAll(generateRulesForItemset(itemset));
            }
        }
        
        return rules;
    }
    
    /**
     * Generate all possible association rules for a single frequent itemset.
     * Example: For {A, B, C}, generate rules like {A,B}=>{C}, {A}=>{B,C}, etc.
     */
    private List<AssociationRule> generateRulesForItemset(Itemset itemset) {
        List<AssociationRule> rules = new ArrayList<>();
        
        // Generate all non-empty subsets as potential antecedents
        List<Set<String>> subsets = generateSubsets(new ArrayList<>(itemset.items));
        
        for (Set<String> antecedent : subsets) {
            // Skip empty set and the full set
            if (antecedent.isEmpty() || antecedent.size() == itemset.items.size()) {
                continue;
            }
            
            // Consequent is the remaining items
            Set<String> consequent = new HashSet<>(itemset.items);
            consequent.removeAll(antecedent);
            
            // Calculate confidence: support(itemset) / support(antecedent)
            int antecedentSupport = getSupport(antecedent);
            double confidence = (double) itemset.support / antecedentSupport;
            
            // Only keep rules that meet minimum confidence
            if (confidence >= minConfidence) {
                // Calculate lift: confidence / support(consequent)
                int consequentSupport = getSupport(consequent);
                double lift = confidence / ((double) consequentSupport / transactions.size());
                
                rules.add(new AssociationRule(antecedent, consequent, 
                    confidence, lift, itemset.support));
            }
        }
        
        return rules;
    }
    
    /**
     * Get the support count for a given itemset.
     */
    private int getSupport(Set<String> items) {
        int size = items.size();
        if (frequentItemsets.containsKey(size)) {
            for (Itemset itemset : frequentItemsets.get(size)) {
                if (itemset.items.equals(items)) {
                    return itemset.support;
                }
            }
        }
        return 0;
    }
    
    /**
     * Generate all subsets of a list of items.
     * Uses bit manipulation to enumerate all 2^n subsets.
     */
    private List<Set<String>> generateSubsets(List<String> items) {
        List<Set<String>> subsets = new ArrayList<>();
        int n = items.size();
        int totalSubsets = (int) Math.pow(2, n);
        
        for (int i = 0; i < totalSubsets; i++) {
            Set<String> subset = new HashSet<>();
            for (int j = 0; j < n; j++) {
                // Check if j-th bit is set in i
                if ((i & (1 << j)) > 0) {
                    subset.add(items.get(j));
                }
            }
            subsets.add(subset);
        }
        
        return subsets;
    }
    
    /**
     * Print all discovered frequent itemsets.
     */
    public void printFrequentItemsets() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("FREQUENT ITEMSETS:");
        System.out.println("=".repeat(60));
        
        for (int k = 1; k <= frequentItemsets.size(); k++) {
            if (!frequentItemsets.containsKey(k)) continue;
            
            System.out.println("\n" + k + "-itemsets:");
            for (Itemset itemset : frequentItemsets.get(k)) {
                double supportPercent = (double) itemset.support / transactions.size() * 100;
                System.out.printf("  %s (Support: %d, %.1f%%)\n", 
                    itemset, itemset.support, supportPercent);
            }
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Apriori Algorithm: Association Rule Mining");
        System.out.println("=" .repeat(60) + "\n");
        
        // Dummy dataset: Grocery store transactions
        List<Set<String>> transactions = Arrays.asList(
            new HashSet<>(Arrays.asList("Bread", "Milk", "Eggs", "Butter")),
            new HashSet<>(Arrays.asList("Bread", "Milk", "Butter")),
            new HashSet<>(Arrays.asList("Bread", "Eggs")),
            new HashSet<>(Arrays.asList("Milk", "Eggs", "Butter", "Cheese")),
            new HashSet<>(Arrays.asList("Bread", "Milk", "Cheese")),
            new HashSet<>(Arrays.asList("Bread", "Milk", "Eggs")),
            new HashSet<>(Arrays.asList("Milk", "Eggs", "Butter")),
            new HashSet<>(Arrays.asList("Bread", "Butter", "Cheese")),
            new HashSet<>(Arrays.asList("Bread", "Milk", "Butter", "Eggs")),
            new HashSet<>(Arrays.asList("Bread", "Eggs", "Cheese"))
        );
        
        System.out.println("Transaction Database (" + transactions.size() + " transactions):");
        for (int i = 0; i < transactions.size(); i++) {
            System.out.println("  T" + (i+1) + ": " + transactions.get(i));
        }
        
        int minSupport = 3; // Item(set) must appear in at least 3 transactions
        double minConfidence = 0.6; // 60% confidence threshold for rules
        
        System.out.println("\nParameters:");
        System.out.println("  Minimum Support: " + minSupport + " transactions");
        System.out.println("  Minimum Confidence: " + (minConfidence * 100) + "%");
        System.out.println("\n" + "=".repeat(60) + "\n");
        
        // Run Apriori algorithm
        AprioriAlgorithm apriori = new AprioriAlgorithm(minSupport, minConfidence);
        List<AssociationRule> rules = apriori.run(transactions);
        
        // Display frequent itemsets
        apriori.printFrequentItemsets();
        
        // Display association rules
        System.out.println("\n" + "=".repeat(60));
        System.out.println("ASSOCIATION RULES:");
        System.out.println("=".repeat(60));
        
        if (rules.isEmpty()) {
            System.out.println("No rules found meeting the confidence threshold.");
        } else {
            // Sort rules by confidence (descending)
            rules.sort((a, b) -> Double.compare(b.confidence, a.confidence));
            
            System.out.println("\nTop Association Rules:");
            for (int i = 0; i < Math.min(10, rules.size()); i++) {
                System.out.println((i+1) + ". " + rules.get(i));
            }
            
            System.out.println("\n" + "=".repeat(60));
            System.out.println("INTERPRETATION:");
            System.out.println("- Confidence: Probability of consequent given antecedent");
            System.out.println("- Lift > 1: Items occur together more than by chance");
            System.out.println("- Lift = 1: Items are independent");
            System.out.println("- Lift < 1: Items occur together less than by chance");
            System.out.println("\nBUSINESS APPLICATIONS:");
            System.out.println("- Product recommendations");
            System.out.println("- Store layout optimization");
            System.out.println("- Promotional bundling");
            System.out.println("- Cross-selling strategies");
        }
    }
}