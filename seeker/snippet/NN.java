//date: 2023-06-13T17:09:01Z
//url: https://api.github.com/gists/a5b31509566719bab7995925001b41a0
//owner: https://api.github.com/users/FriedGil

import java.util.Arrays;
import java.util.Random;

public class NN {

    private final int featuresize;
    private final int hiddenSize;
    private final int outputSize;
    private double[][] weights1;
    private double[][] weights2;
    private double[] hiddenLayer;
    private double[] outputLayer;
    private double[] hiddenLayerError;
    private double[] outputLayerError;
    private final double learningRate;

    //https://towardsdatascience.com/machine-learning-for-beginners-an-introduction-to-neural-networks-d49f22d238f9 https://www.youtube.com/watch?v=aircAruvnKk
    public NN(int featuresize, int hiddenSize, int outputSize, double learningRate) {
        this.featuresize = featuresize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        weights1 = new double[featuresize][hiddenSize];
        weights2 = new double[hiddenSize][outputSize];

        for (int i = 0; i < featuresize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weights1[i][j] = Math.random() - 0.5;
            }
        }
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights2[i][j] = Math.random() - 0.5;
            }
        }
        
        hiddenLayer = new double[hiddenSize];
        outputLayer = new double[outputSize];
        hiddenLayerError = new double[hiddenSize];
        outputLayerError = new double[outputSize];
    }

    //https://www.youtube.com/watch?v=IHZwWFHWa-w

    public double[] feedForward(double[] input) {
        for (int i = 0; i < hiddenSize; i++) {
            double sum = 0;
            for (int j = 0; j < featuresize; j++) {
                sum += input[j] * weights1[j][i];
            }
            hiddenLayer[i] = sigmoid(sum);
        }

        for (int i = 0; i < outputSize; i++) {
            double sum = 0;
            for (int j = 0; j < hiddenSize; j++) {
                sum += hiddenLayer[j] * weights2[j][i];
            }
            outputLayer[i] = sigmoid(sum);
        }

        return outputLayer;
    }

    public void train(double[] input, double[] target) {
        double[] output = feedForward(input);

        for (int i = 0; i < outputSize; i++) {
            outputLayerError[i] = (target[i] - output[i]) * sigmoidDerivative(output[i]);
        }

        for (int i = 0; i < hiddenSize; i++) {
            double sum = 0;
            for (int j = 0; j < outputSize; j++) {
                sum += weights2[i][j] * outputLayerError[j];
            }
            hiddenLayerError[i] = sum * sigmoidDerivative(hiddenLayer[i]);
        }

        for (int i = 0; i < featuresize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weights1[i][j] += learningRate * input[i] * hiddenLayerError[j];
            }
        }
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights2[i][j] += learningRate * hiddenLayer[i] * outputLayerError[j];
            }
        }
    }

    //Sigmoid is an activation function. https://medium.com/@itcalderon11/a-simple-guide-to-activation-functions-in-neural-networks-422995f14b9 https://en.wikipedia.org/wiki/Sigmoid_function
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        double sig = sigmoid(x);
        return sig * (1.0 - sig);
    }

    public static void main(String[] args) {
        //In this example, the input data is an array of two numbers and the output is 0 or 1. 
        //In this case the output is only dependent on the first number, which the model manages to understand fairly well.
        NN neuralNetwork = new NN(2, 2, 1, 0.1);
        //Features are your input and labels are your output. features.length and labels.length must be the same.
        double[][] features = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {0,2}, {0,3}};
        double[][] labels = {{0}, {0}, {1}, {1}, {0}, {0}};
        for (int i = 0; i < 10000; i++) {
            for (int j = 0; j < features.length; j++) {
                neuralNetwork.train(features[j], labels[j]);
            }
        }
        //Arrays in test should have the same dimensions as your features.
        double[][] test = {{0, 1}, {1, 0}, {1, 1}, {0,2}, {0,5}, {0,4}};
        for (int i = 0; i < test.length; i++) {
            double[] input = test[i];
            double[] output = neuralNetwork.feedForward(input);
            System.out.println("Input: " + Arrays.toString(input) + ", Output: " + Arrays.toString(output));
        }
    }
}