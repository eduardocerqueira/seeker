//date: 2025-11-14T16:44:22Z
//url: https://api.github.com/gists/e0851c27777ecf9f42ff7dec97e70419
//owner: https://api.github.com/users/metaloozee

import java.util.*;

/**
 * Naive Bayes Classifier Implementation
 * 
 * This classifier uses Bayes' theorem with the "naive" assumption that features
 * are independent of each other. It calculates the probability of each class
 * given the input features and selects the class with the highest probability.
 * 
 * Example Domain: Weather prediction (will it rain or not?)
 * Features: Outlook, Temperature, Humidity, Wind
 */
public class NaiveBayesClassifier {
    
    // Store training data
    private List<String[]> trainingData;
    private Map<String, Integer> classCount; // Count of each class
    private Map<String, Map<String, Map<String, Integer>>> featureCount; // Feature value counts per class
    private int totalSamples;
    
    public NaiveBayesClassifier() {
        trainingData = new ArrayList<>();
        classCount = new HashMap<>();
        featureCount = new HashMap<>();
        totalSamples = 0;
    }
    
    /**
     * Train the classifier with the provided dataset.
     * Each row contains: [outlook, temperature, humidity, wind, playTennis]
     */
    public void train(List<String[]> data) {
        trainingData = data;
        totalSamples = data.size();
        
        // Count occurrences of each class and feature values
        for (String[] row : data) {
            String classValue = row[row.length - 1]; // Last column is the class
            
            // Count classes
            classCount.put(classValue, classCount.getOrDefault(classValue, 0) + 1);
            
            // Count feature values for each class
            for (int i = 0; i < row.length - 1; i++) {
                String featureName = "feature_" + i;
                String featureValue = row[i];
                
                // Initialize nested maps if needed
                featureCount.putIfAbsent(classValue, new HashMap<>());
                featureCount.get(classValue).putIfAbsent(featureName, new HashMap<>());
                
                // Increment count
                Map<String, Integer> valueMap = featureCount.get(classValue).get(featureName);
                valueMap.put(featureValue, valueMap.getOrDefault(featureValue, 0) + 1);
            }
        }
    }
    
    /**
     * Predict the class for a new instance using Naive Bayes formula.
     * Returns the class with the highest probability.
     */
    public String predict(String[] instance) {
        double maxProbability = Double.NEGATIVE_INFINITY;
        String predictedClass = null;
        
        // Calculate probability for each class
        for (String classValue : classCount.keySet()) {
            double probability = calculateClassProbability(instance, classValue);
            
            if (probability > maxProbability) {
                maxProbability = probability;
                predictedClass = classValue;
            }
        }
        
        return predictedClass;
    }
    
    /**
     * Calculate P(Class | Features) using Bayes theorem:
     * P(Class | Features) = P(Class) * P(Feature1 | Class) * P(Feature2 | Class) * ...
     * 
     * We use log probabilities to avoid underflow with very small numbers.
     */
    private double calculateClassProbability(String[] instance, String classValue) {
        // Start with prior probability: P(Class)
        double logProbability = Math.log((double) classCount.get(classValue) / totalSamples);
        
        // Multiply by conditional probabilities: P(Feature | Class)
        for (int i = 0; i < instance.length; i++) {
            String featureName = "feature_" + i;
            String featureValue = instance[i];
            
            // Get count of this feature value for this class
            int featureValueCount = 0;
            if (featureCount.get(classValue).containsKey(featureName) &&
                featureCount.get(classValue).get(featureName).containsKey(featureValue)) {
                featureValueCount = featureCount.get(classValue).get(featureName).get(featureValue);
            }
            
            // Apply Laplace smoothing (add 1) to handle zero probabilities
            int classTotal = classCount.get(classValue);
            int numUniqueValues = getUniqueFeatureValues(i).size();
            double conditionalProb = (double) (featureValueCount + 1) / (classTotal + numUniqueValues);
            
            logProbability += Math.log(conditionalProb);
        }
        
        return logProbability;
    }
    
    /**
     * Get all unique values for a specific feature across all training data.
     * Used for Laplace smoothing calculation.
     */
    private Set<String> getUniqueFeatureValues(int featureIndex) {
        Set<String> uniqueValues = new HashSet<>();
        for (String[] row : trainingData) {
            uniqueValues.add(row[featureIndex]);
        }
        return uniqueValues;
    }
    
    /**
     * Calculate and display accuracy on a test dataset.
     */
    public double evaluateAccuracy(List<String[]> testData) {
        int correct = 0;
        for (String[] row : testData) {
            String[] features = Arrays.copyOf(row, row.length - 1);
            String actualClass = row[row.length - 1];
            String predictedClass = predict(features);
            
            if (predictedClass.equals(actualClass)) {
                correct++;
            }
        }
        return (double) correct / testData.size();
    }
    
    // Main method with dummy dataset for demonstration
    public static void main(String[] args) {
        NaiveBayesClassifier classifier = new NaiveBayesClassifier();
        
        // Dummy dataset: Weather conditions and whether to play tennis
        // Format: [Outlook, Temperature, Humidity, Wind, PlayTennis]
        List<String[]> trainingData = Arrays.asList(
            new String[]{"Sunny", "Hot", "High", "Weak", "No"},
            new String[]{"Sunny", "Hot", "High", "Strong", "No"},
            new String[]{"Overcast", "Hot", "High", "Weak", "Yes"},
            new String[]{"Rain", "Mild", "High", "Weak", "Yes"},
            new String[]{"Rain", "Cool", "Normal", "Weak", "Yes"},
            new String[]{"Rain", "Cool", "Normal", "Strong", "No"},
            new String[]{"Overcast", "Cool", "Normal", "Strong", "Yes"},
            new String[]{"Sunny", "Mild", "High", "Weak", "No"},
            new String[]{"Sunny", "Cool", "Normal", "Weak", "Yes"},
            new String[]{"Rain", "Mild", "Normal", "Weak", "Yes"},
            new String[]{"Sunny", "Mild", "Normal", "Strong", "Yes"},
            new String[]{"Overcast", "Mild", "High", "Strong", "Yes"},
            new String[]{"Overcast", "Hot", "Normal", "Weak", "Yes"},
            new String[]{"Rain", "Mild", "High", "Strong", "No"}
        );
        
        // Train the classifier
        System.out.println("Training Naive Bayes Classifier...");
        classifier.train(trainingData);
        
        // Test with new instances
        System.out.println("\n--- Making Predictions ---");
        String[][] testInstances = {
            {"Sunny", "Cool", "High", "Strong"},
            {"Overcast", "Hot", "Normal", "Weak"},
            {"Rain", "Cool", "High", "Weak"}
        };
        
        for (String[] instance : testInstances) {
            String prediction = classifier.predict(instance);
            System.out.println("Weather: " + Arrays.toString(instance) + 
                             " -> Prediction: " + prediction);
        }
        
        // Evaluate on training data (for demonstration)
        System.out.println("\n--- Accuracy Evaluation ---");
        double accuracy = classifier.evaluateAccuracy(trainingData);
        System.out.printf("Training Accuracy: %.2f%%\n", accuracy * 100);
    }
}