//date: 2025-11-14T16:44:22Z
//url: https://api.github.com/gists/e0851c27777ecf9f42ff7dec97e70419
//owner: https://api.github.com/users/metaloozee

import java.util.*;

/**
 * Decision Tree Classifier using ID3 Algorithm
 * 
 * This classifier builds a tree structure where each internal node represents
 * a decision on a feature, and each leaf node represents a class label.
 * The tree is constructed by recursively selecting the feature that provides
 * the highest information gain (reduces entropy the most).
 * 
 * Example Domain: Customer purchase prediction
 * Features: Age, Income, Student, Credit Rating
 */
public class DecisionTreeClassifier {
    
    // Node class to represent each node in the decision tree
    static class TreeNode {
        String featureName;           // Feature to split on (null for leaf nodes)
        String classLabel;            // Class label (only for leaf nodes)
        Map<String, TreeNode> children; // Child nodes for each feature value
        boolean isLeaf;
        
        public TreeNode() {
            children = new HashMap<>();
            isLeaf = false;
        }
    }
    
    private TreeNode root;
    private List<String> featureNames;
    
    public DecisionTreeClassifier(List<String> featureNames) {
        this.featureNames = new ArrayList<>(featureNames);
    }
    
    /**
     * Train the decision tree using the ID3 algorithm.
     * Data format: Each row is a feature vector with the last element being the class label.
     */
    public void train(List<String[]> data) {
        List<Integer> availableFeatures = new ArrayList<>();
        for (int i = 0; i < featureNames.size(); i++) {
            availableFeatures.add(i);
        }
        root = buildTree(data, availableFeatures);
    }
    
    /**
     * Recursively build the decision tree using information gain.
     * This is the core of the ID3 algorithm.
     */
    private TreeNode buildTree(List<String[]> data, List<Integer> availableFeatures) {
        TreeNode node = new TreeNode();
        
        // Base case 1: All instances have the same class
        Set<String> classes = getClassLabels(data);
        if (classes.size() == 1) {
            node.isLeaf = true;
            node.classLabel = classes.iterator().next();
            return node;
        }
        
        // Base case 2: No more features to split on
        if (availableFeatures.isEmpty()) {
            node.isLeaf = true;
            node.classLabel = getMajorityClass(data);
            return node;
        }
        
        // Find the best feature to split on (highest information gain)
        int bestFeature = selectBestFeature(data, availableFeatures);
        node.featureName = featureNames.get(bestFeature);
        
        // Get all possible values for the selected feature
        Set<String> featureValues = getFeatureValues(data, bestFeature);
        
        // Create a subset for each feature value and recursively build subtrees
        for (String value : featureValues) {
            List<String[]> subset = filterDataByFeature(data, bestFeature, value);
            
            if (subset.isEmpty()) {
                // If no examples, create a leaf with majority class of parent
                TreeNode leaf = new TreeNode();
                leaf.isLeaf = true;
                leaf.classLabel = getMajorityClass(data);
                node.children.put(value, leaf);
            } else {
                // Recursively build subtree
                List<Integer> remainingFeatures = new ArrayList<>(availableFeatures);
                remainingFeatures.remove(Integer.valueOf(bestFeature));
                node.children.put(value, buildTree(subset, remainingFeatures));
            }
        }
        
        return node;
    }
    
    /**
     * Select the feature that provides the highest information gain.
     * Information Gain = Entropy(parent) - Weighted Average of Entropy(children)
     */
    private int selectBestFeature(List<String[]> data, List<Integer> availableFeatures) {
        double baseEntropy = calculateEntropy(data);
        double maxInfoGain = Double.NEGATIVE_INFINITY;
        int bestFeature = availableFeatures.get(0);
        
        for (int feature : availableFeatures) {
            double infoGain = calculateInformationGain(data, feature, baseEntropy);
            if (infoGain > maxInfoGain) {
                maxInfoGain = infoGain;
                bestFeature = feature;
            }
        }
        
        return bestFeature;
    }
    
    /**
     * Calculate information gain for a specific feature.
     */
    private double calculateInformationGain(List<String[]> data, int featureIndex, double baseEntropy) {
        Set<String> values = getFeatureValues(data, featureIndex);
        double weightedEntropy = 0.0;
        
        for (String value : values) {
            List<String[]> subset = filterDataByFeature(data, featureIndex, value);
            double weight = (double) subset.size() / data.size();
            weightedEntropy += weight * calculateEntropy(subset);
        }
        
        return baseEntropy - weightedEntropy;
    }
    
    /**
     * Calculate entropy of a dataset.
     * Entropy = -Σ(p(x) * log2(p(x))) for each class x
     * Lower entropy means more homogeneous data.
     */
    private double calculateEntropy(List<String[]> data) {
        Map<String, Integer> classCount = new HashMap<>();
        
        // Count occurrences of each class
        for (String[] row : data) {
            String classLabel = row[row.length - 1];
            classCount.put(classLabel, classCount.getOrDefault(classLabel, 0) + 1);
        }
        
        // Calculate entropy
        double entropy = 0.0;
        int total = data.size();
        for (int count : classCount.values()) {
            if (count > 0) {
                double probability = (double) count / total;
                entropy -= probability * (Math.log(probability) / Math.log(2));
            }
        }
        
        return entropy;
    }
    
    /**
     * Get all unique class labels in the dataset.
     */
    private Set<String> getClassLabels(List<String[]> data) {
        Set<String> classes = new HashSet<>();
        for (String[] row : data) {
            classes.add(row[row.length - 1]);
        }
        return classes;
    }
    
    /**
     * Get all unique values for a specific feature.
     */
    private Set<String> getFeatureValues(List<String[]> data, int featureIndex) {
        Set<String> values = new HashSet<>();
        for (String[] row : data) {
            values.add(row[featureIndex]);
        }
        return values;
    }
    
    /**
     * Filter dataset to only include rows where the feature has a specific value.
     */
    private List<String[]> filterDataByFeature(List<String[]> data, int featureIndex, String value) {
        List<String[]> filtered = new ArrayList<>();
        for (String[] row : data) {
            if (row[featureIndex].equals(value)) {
                filtered.add(row);
            }
        }
        return filtered;
    }
    
    /**
     * Find the most common class label in the dataset.
     */
    private String getMajorityClass(List<String[]> data) {
        Map<String, Integer> classCount = new HashMap<>();
        for (String[] row : data) {
            String classLabel = row[row.length - 1];
            classCount.put(classLabel, classCount.getOrDefault(classLabel, 0) + 1);
        }
        
        String majorityClass = null;
        int maxCount = 0;
        for (Map.Entry<String, Integer> entry : classCount.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                majorityClass = entry.getKey();
            }
        }
        return majorityClass;
    }
    
    /**
     * Predict the class label for a new instance by traversing the tree.
     */
    public String predict(String[] instance) {
        return predictHelper(root, instance);
    }
    
    private String predictHelper(TreeNode node, String[] instance) {
        if (node.isLeaf) {
            return node.classLabel;
        }
        
        // Find the feature index for this node
        int featureIndex = featureNames.indexOf(node.featureName);
        String featureValue = instance[featureIndex];
        
        // Traverse to the appropriate child
        if (node.children.containsKey(featureValue)) {
            return predictHelper(node.children.get(featureValue), instance);
        } else {
            // If value not seen in training, return most common class at this node
            return getMostCommonLeafClass(node);
        }
    }
    
    private String getMostCommonLeafClass(TreeNode node) {
        if (node.isLeaf) {
            return node.classLabel;
        }
        
        Map<String, Integer> classCount = new HashMap<>();
        for (TreeNode child : node.children.values()) {
            String classLabel = getMostCommonLeafClass(child);
            classCount.put(classLabel, classCount.getOrDefault(classLabel, 0) + 1);
        }
        
        String majorityClass = null;
        int maxCount = 0;
        for (Map.Entry<String, Integer> entry : classCount.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                majorityClass = entry.getKey();
            }
        }
        return majorityClass;
    }
    
    /**
     * Print the decision tree structure.
     */
    public void printTree() {
        System.out.println("Decision Tree Structure:");
        printTreeHelper(root, "", true);
    }
    
    private void printTreeHelper(TreeNode node, String prefix, boolean isRoot) {
        if (node.isLeaf) {
            System.out.println(prefix + "→ Class: " + node.classLabel);
            return;
        }
        
        if (isRoot) {
            System.out.println(prefix + node.featureName);
        }
        
        List<Map.Entry<String, TreeNode>> entries = new ArrayList<>(node.children.entrySet());
        for (int i = 0; i < entries.size(); i++) {
            Map.Entry<String, TreeNode> entry = entries.get(i);
            boolean isLast = (i == entries.size() - 1);
            String connector = isLast ? "└── " : "├── ";
            String childPrefix = isLast ? "    " : "│   ";
            
            System.out.print(prefix + connector + entry.getKey());
            if (!entry.getValue().isLeaf) {
                System.out.println(" → " + entry.getValue().featureName);
                printTreeHelper(entry.getValue(), prefix + childPrefix, false);
            } else {
                System.out.println();
                printTreeHelper(entry.getValue(), prefix + childPrefix, false);
            }
        }
    }
    
    public static void main(String[] args) {
        // Define feature names
        List<String> features = Arrays.asList("Age", "Income", "Student", "CreditRating");
        DecisionTreeClassifier classifier = new DecisionTreeClassifier(features);
        
        // Dummy dataset: Customer information and whether they bought a computer
        // Format: [Age, Income, Student, CreditRating, BuysComputer]
        List<String[]> trainingData = Arrays.asList(
            new String[]{"Youth", "High", "No", "Fair", "No"},
            new String[]{"Youth", "High", "No", "Excellent", "No"},
            new String[]{"MiddleAged", "High", "No", "Fair", "Yes"},
            new String[]{"Senior", "Medium", "No", "Fair", "Yes"},
            new String[]{"Senior", "Low", "Yes", "Fair", "Yes"},
            new String[]{"Senior", "Low", "Yes", "Excellent", "No"},
            new String[]{"MiddleAged", "Low", "Yes", "Excellent", "Yes"},
            new String[]{"Youth", "Medium", "No", "Fair", "No"},
            new String[]{"Youth", "Low", "Yes", "Fair", "Yes"},
            new String[]{"Senior", "Medium", "Yes", "Fair", "Yes"},
            new String[]{"Youth", "Medium", "Yes", "Excellent", "Yes"},
            new String[]{"MiddleAged", "Medium", "No", "Excellent", "Yes"},
            new String[]{"MiddleAged", "High", "Yes", "Fair", "Yes"},
            new String[]{"Senior", "Medium", "No", "Excellent", "No"}
        );
        
        // Train the classifier
        System.out.println("Training Decision Tree Classifier...\n");
        classifier.train(trainingData);
        
        // Print the tree structure
        classifier.printTree();
        
        // Test with new instances
        System.out.println("\n--- Making Predictions ---");
        String[][] testInstances = {
            {"Youth", "Medium", "Yes", "Fair"},
            {"Senior", "High", "No", "Excellent"},
            {"MiddleAged", "Low", "No", "Fair"}
        };
        
        for (String[] instance : testInstances) {
            String prediction = classifier.predict(instance);
            System.out.println("Customer: " + Arrays.toString(instance) + 
                             " -> Prediction: " + prediction);
        }
    }
}