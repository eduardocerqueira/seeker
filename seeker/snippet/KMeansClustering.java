//date: 2025-11-14T16:44:22Z
//url: https://api.github.com/gists/e0851c27777ecf9f42ff7dec97e70419
//owner: https://api.github.com/users/metaloozee

import java.util.*;

/**
 * K-Means Clustering Algorithm
 * 
 * K-Means is an unsupervised learning algorithm that partitions data into K clusters.
 * Each data point belongs to the cluster with the nearest centroid (mean).
 * 
 * Algorithm Steps:
 * 1. Initialize K centroids randomly
 * 2. Assign each point to the nearest centroid
 * 3. Recalculate centroids as the mean of assigned points
 * 4. Repeat steps 2-3 until convergence (centroids don't change significantly)
 * 
 * Example Domain: Customer segmentation based on spending patterns
 * Features: Annual Income, Spending Score (both normalized 0-100)
 */
public class KMeansClustering {
    
    // Class to represent a 2D point (can be extended to N dimensions)
    static class Point {
        double x, y;
        int clusterId; // Which cluster this point belongs to
        
        public Point(double x, double y) {
            this.x = x;
            this.y = y;
            this.clusterId = -1; // -1 means not assigned yet
        }
        
        /**
         * Calculate Euclidean distance to another point.
         * Distance = sqrt((x1-x2)² + (y1-y2)²)
         */
        public double distanceTo(Point other) {
            double dx = this.x - other.x;
            double dy = this.y - other.y;
            return Math.sqrt(dx * dx + dy * dy);
        }
        
        @Override
        public String toString() {
            return String.format("(%.2f, %.2f)", x, y);
        }
    }
    
    // Class to represent a cluster centroid
    static class Centroid {
        Point location;
        int id;
        
        public Centroid(int id, double x, double y) {
            this.id = id;
            this.location = new Point(x, y);
        }
        
        /**
         * Update centroid location to be the mean of all assigned points.
         */
        public void updateLocation(List<Point> assignedPoints) {
            if (assignedPoints.isEmpty()) {
                return; // Keep current location if no points assigned
            }
            
            double sumX = 0, sumY = 0;
            for (Point p : assignedPoints) {
                sumX += p.x;
                sumY += p.y;
            }
            
            location.x = sumX / assignedPoints.size();
            location.y = sumY / assignedPoints.size();
        }
        
        @Override
        public String toString() {
            return String.format("Centroid %d: %s", id, location);
        }
    }
    
    private int k; // Number of clusters
    private List<Centroid> centroids;
    private List<Point> dataPoints;
    private int maxIterations;
    private double convergenceThreshold;
    
    public KMeansClustering(int k, int maxIterations, double convergenceThreshold) {
        this.k = k;
        this.maxIterations = maxIterations;
        this.convergenceThreshold = convergenceThreshold;
        this.centroids = new ArrayList<>();
        this.dataPoints = new ArrayList<>();
    }
    
    /**
     * Run the K-Means clustering algorithm on the provided data points.
     */
    public void fit(List<Point> data) {
        this.dataPoints = data;
        
        // Step 1: Initialize centroids randomly from the data points
        initializeCentroids();
        
        System.out.println("Initial centroids:");
        for (Centroid c : centroids) {
            System.out.println("  " + c);
        }
        
        // Iterate until convergence or max iterations reached
        int iteration = 0;
        boolean converged = false;
        
        while (iteration < maxIterations && !converged) {
            iteration++;
            
            // Step 2: Assign each point to the nearest centroid
            assignPointsToClusters();
            
            // Step 3: Update centroid positions
            List<Point> oldCentroidLocations = new ArrayList<>();
            for (Centroid c : centroids) {
                oldCentroidLocations.add(new Point(c.location.x, c.location.y));
            }
            
            updateCentroids();
            
            // Check for convergence (centroids didn't move much)
            converged = hasConverged(oldCentroidLocations);
            
            System.out.println("\nIteration " + iteration + ":");
            printClusterSizes();
        }
        
        System.out.println("\nConverged after " + iteration + " iterations.");
        System.out.println("\nFinal centroids:");
        for (Centroid c : centroids) {
            System.out.println("  " + c);
        }
    }
    
    /**
     * Initialize centroids by randomly selecting K points from the dataset.
     * More sophisticated methods like K-Means++ can be used for better initialization.
     */
    private void initializeCentroids() {
        Random random = new Random(42); // Fixed seed for reproducibility
        List<Point> shuffled = new ArrayList<>(dataPoints);
        Collections.shuffle(shuffled, random);
        
        for (int i = 0; i < k; i++) {
            Point p = shuffled.get(i);
            centroids.add(new Centroid(i, p.x, p.y));
        }
    }
    
    /**
     * Assign each data point to the cluster with the nearest centroid.
     */
    private void assignPointsToClusters() {
        for (Point point : dataPoints) {
            double minDistance = Double.MAX_VALUE;
            int nearestCentroidId = -1;
            
            // Find the nearest centroid
            for (Centroid centroid : centroids) {
                double distance = point.distanceTo(centroid.location);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestCentroidId = centroid.id;
                }
            }
            
            point.clusterId = nearestCentroidId;
        }
    }
    
    /**
     * Update each centroid to be the mean position of all points assigned to it.
     */
    private void updateCentroids() {
        // Group points by cluster
        Map<Integer, List<Point>> clusterPoints = new HashMap<>();
        for (int i = 0; i < k; i++) {
            clusterPoints.put(i, new ArrayList<>());
        }
        
        for (Point point : dataPoints) {
            clusterPoints.get(point.clusterId).add(point);
        }
        
        // Update each centroid
        for (Centroid centroid : centroids) {
            centroid.updateLocation(clusterPoints.get(centroid.id));
        }
    }
    
    /**
     * Check if the algorithm has converged by measuring how much centroids moved.
     */
    private boolean hasConverged(List<Point> oldLocations) {
        for (int i = 0; i < centroids.size(); i++) {
            Point oldLoc = oldLocations.get(i);
            Point newLoc = centroids.get(i).location;
            double movement = oldLoc.distanceTo(newLoc);
            
            if (movement > convergenceThreshold) {
                return false; // At least one centroid moved significantly
            }
        }
        return true; // All centroids are stable
    }
    
    /**
     * Calculate the Within-Cluster Sum of Squares (WCSS) - a measure of cluster quality.
     * Lower WCSS indicates tighter, more compact clusters.
     */
    public double calculateWCSS() {
        double wcss = 0.0;
        
        for (Point point : dataPoints) {
            Centroid assignedCentroid = centroids.get(point.clusterId);
            double distance = point.distanceTo(assignedCentroid.location);
            wcss += distance * distance;
        }
        
        return wcss;
    }
    
    /**
     * Print the size of each cluster.
     */
    private void printClusterSizes() {
        Map<Integer, Integer> clusterSizes = new HashMap<>();
        for (int i = 0; i < k; i++) {
            clusterSizes.put(i, 0);
        }
        
        for (Point point : dataPoints) {
            clusterSizes.put(point.clusterId, clusterSizes.get(point.clusterId) + 1);
        }
        
        for (int i = 0; i < k; i++) {
            System.out.println("  Cluster " + i + ": " + clusterSizes.get(i) + " points");
        }
    }
    
    /**
     * Get all points assigned to a specific cluster.
     */
    public List<Point> getCluster(int clusterId) {
        List<Point> cluster = new ArrayList<>();
        for (Point point : dataPoints) {
            if (point.clusterId == clusterId) {
                cluster.add(point);
            }
        }
        return cluster;
    }
    
    /**
     * Print detailed cluster information.
     */
    public void printClusters() {
        System.out.println("\n=== Cluster Details ===");
        for (int i = 0; i < k; i++) {
            List<Point> cluster = getCluster(i);
            System.out.println("\nCluster " + i + " (" + cluster.size() + " points):");
            System.out.println("  Centroid: " + centroids.get(i).location);
            System.out.println("  Sample points:");
            for (int j = 0; j < Math.min(5, cluster.size()); j++) {
                System.out.println("    " + cluster.get(j));
            }
            if (cluster.size() > 5) {
                System.out.println("    ... and " + (cluster.size() - 5) + " more");
            }
        }
        
        System.out.printf("\nWithin-Cluster Sum of Squares (WCSS): %.2f\n", calculateWCSS());
    }
    
    public static void main(String[] args) {
        // Dummy dataset: Customer segmentation based on income and spending score
        // Creating 3 distinct groups for better visualization
        List<Point> customerData = new ArrayList<>();
        
        // Group 1: Low income, low spending (15 customers)
        customerData.add(new Point(15, 20));
        customerData.add(new Point(18, 25));
        customerData.add(new Point(12, 18));
        customerData.add(new Point(20, 22));
        customerData.add(new Point(16, 19));
        customerData.add(new Point(14, 23));
        customerData.add(new Point(17, 21));
        customerData.add(new Point(19, 24));
        customerData.add(new Point(13, 20));
        customerData.add(new Point(21, 26));
        customerData.add(new Point(15, 22));
        customerData.add(new Point(18, 19));
        customerData.add(new Point(16, 25));
        customerData.add(new Point(14, 21));
        customerData.add(new Point(17, 23));
        
        // Group 2: High income, low spending (15 customers)
        customerData.add(new Point(75, 25));
        customerData.add(new Point(80, 30));
        customerData.add(new Point(72, 22));
        customerData.add(new Point(78, 28));
        customerData.add(new Point(76, 24));
        customerData.add(new Point(82, 27));
        customerData.add(new Point(74, 26));
        customerData.add(new Point(79, 29));
        customerData.add(new Point(77, 23));
        customerData.add(new Point(81, 25));
        customerData.add(new Point(73, 28));
        customerData.add(new Point(78, 26));
        customerData.add(new Point(76, 27));
        customerData.add(new Point(80, 24));
        customerData.add(new Point(75, 29));
        
        // Group 3: High income, high spending (15 customers)
        customerData.add(new Point(70, 75));
        customerData.add(new Point(75, 80));
        customerData.add(new Point(68, 72));
        customerData.add(new Point(73, 78));
        customerData.add(new Point(71, 76));
        customerData.add(new Point(77, 82));
        customerData.add(new Point(69, 74));
        customerData.add(new Point(74, 79));
        customerData.add(new Point(72, 77));
        customerData.add(new Point(76, 81));
        customerData.add(new Point(70, 73));
        customerData.add(new Point(75, 78));
        customerData.add(new Point(73, 76));
        customerData.add(new Point(71, 80));
        customerData.add(new Point(74, 75));
        
        System.out.println("K-Means Clustering: Customer Segmentation");
        System.out.println("Dataset: " + customerData.size() + " customers");
        System.out.println("Features: Annual Income (x-axis), Spending Score (y-axis)");
        System.out.println("=" .repeat(60) + "\n");
        
        // Create K-Means clusterer with k=3 clusters
        KMeansClustering kmeans = new KMeansClustering(
            3,      // number of clusters
            100,    // max iterations
            0.01    // convergence threshold
        );
        
        // Fit the model
        kmeans.fit(customerData);
        
        // Display results
        kmeans.printClusters();
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Interpretation:");
        System.out.println("- Cluster 0: Likely represents one customer segment");
        System.out.println("- Cluster 1: Likely represents another customer segment");
        System.out.println("- Cluster 2: Likely represents a third customer segment");
        System.out.println("\nBusinesses can use these clusters to:");
        System.out.println("- Target marketing campaigns");
        System.out.println("- Personalize product recommendations");
        System.out.println("- Optimize pricing strategies");
    }
}