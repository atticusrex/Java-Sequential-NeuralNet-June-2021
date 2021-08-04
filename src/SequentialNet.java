public class SequentialNet {
    public static InputLayer inputLayer;
    public static Layer[] hiddenLayers;
    public static OutputLayer outputLayer;
    public static double learningRate = .05;

    /**
     * Constructor method initializes the neural networks to have a hidden layer that is 0 layers
     * long
     */
    public SequentialNet() {
        hiddenLayers = new Layer[0];
    }

    public double[] runBackwards(double[] outputs) {
        Matrix outputMatrix = new Matrix(outputLayer.size, 1);

        for (int i = 0; i < outputLayer.size; i++) {
            outputMatrix.set(i, 0, outputs[i]);
        }
        return outputs;
    }

    /**
     * This sets the number of input nodes for the neural network
     * @param n The number of inputs
     */
    public void setInputDim(int n) {
        inputLayer = new InputLayer(n);
    }

    /**
     * This sets the number of output nodes for the nn
     * @param n The number of outputs
     */
    public void setOutputDim(int n) {
        outputLayer = new OutputLayer(n);
    }

    /**
     * This sets the input values as an array of data
     * @param data The actual input values of the network derived from whatever data is being used. 1d array.
     */
    public void setInputs(double[] data) {
        inputLayer.setValues(data);
    }

    /**
     * This sets the target outputs that the network should be able to hit
     *
     * @param data - A 1d array of the outputs the network is supposed to have.
     */
    public void setExpected(double[] data) {
        outputLayer.setExpected(data);
    }

    /**
     * This adds a hidden layer to the network based on a specific number of nodes
     * @param size The number of nodes in the new hidden layer.
     */
    public void addHiddenLayer(int size) {
        int n = hiddenLayers.length;
        Layer[] tempLayers = new Layer[n + 1];
        System.arraycopy(hiddenLayers, 0, tempLayers, 0, n);
        hiddenLayers = tempLayers;
        hiddenLayers[n] = new Layer(size);
        if (n != 0) {
            hiddenLayers[n - 1].addOutputLayer(hiddenLayers[n]);
        }
        inputLayer.addOutputLayer(hiddenLayers[0]);
        hiddenLayers[n].addOutputLayer(outputLayer);
    }

    /**
     * This propagates the input forward in the neural network.
     *
     * @return a 1d array representing the output of the network
     */
    public double[] think() {
        inputLayer.sendPulse();

        for (Layer hiddenLayer : hiddenLayers) {
            hiddenLayer.sendPulse();
        }

        int n = outputLayer.size;

        double[] values = new double[n];

        for (int i = 0; i < n; i++) {
            values[i] = outputLayer.neurons[i].value;
        }

        return values;
    }

    /**
     * This returns the calculated error between the actual output of the nn and the expected output of the neural net
     *
     * @return the numerical error to be optimized.
     */
    public double error() {
        return outputLayer.error();
    }

    /**
     * This sets the learning rate of the neural network
     *
     * @param num value of the learning rate
     */
    public void setLearningRate(double num) {
        learningRate = num;
    }

    /**
     * Master function to backpropagate the neural network.
     */
    public void backpropagate() {
        Matrix delta = adjustOutputWeights();

        for (int i = hiddenLayers.length - 1; i > 0; i--) {
            delta = adjustWeights(delta, hiddenLayers[i], hiddenLayers[i - 1]);
        }

        adjustWeights(delta, hiddenLayers[0], inputLayer);
    }

    public static Matrix adjustOutputWeights() {
        Matrix a = outputLayer.preValues;
        Matrix z = outputLayer.values;
        Matrix T = outputLayer.expectedVals;
        Matrix dCdz = Matrix.subtract(z, T);
        Matrix dzda = a.dsigmoid();
        Matrix dCda = Matrix.elemMultiply(dCdz, dzda);

        Layer prevLayer = hiddenLayers[hiddenLayers.length - 1];

        Matrix z_prev = prevLayer.values;

        Matrix dCdw = new Matrix(outputLayer.size, prevLayer.size);
        for (int i = 0; i < outputLayer.size; i++) {
            for (int j = 0; j < prevLayer.size; j++) {
                double val = dCda.get(i, 0) * z_prev.get(j, 0);
                dCdw.set(i, j, val);
            }
        }

        Matrix delta = dCda;
        dCdw.multiply(learningRate);
        prevLayer.weights = Matrix.subtract(prevLayer.weights, dCdw);
        return delta;
    }

    public Matrix adjustWeights(Matrix delta, Layer layer, Layer prevLayer) {
        Matrix weights = Matrix.transpose(layer.weights);
        Matrix a = layer.preValues;
        delta = Matrix.elemMultiply(Matrix.multiply(weights, delta), a.sigmoid());

        Matrix adj = new Matrix(delta.rows, prevLayer.size);

        for (int i = 0; i < delta.rows; i++) {
            for (int j = 0; j < prevLayer.size; j++) {
                adj.data[i][j] = delta.get(i, 0) * prevLayer.values.get(j, 0);
            }
        }

        adj.multiply(learningRate);
        prevLayer.weights = Matrix.subtract(prevLayer.weights, adj);
        return delta;
    }

    private static double[][] transpose(double A[][]) {
        double trans[][] = new double[A[0].length][A.length];
        for (int row = 0; row < A.length; row++) {
            for (int col = 0; col < A[row].length; col++) {
                trans[col][row] = A[row][col];
            }
        }
        return trans;
    }

    private static double[][] multiply(double first[][], double second[][]) {
        int r1 = first.length;
        int c1 = first[0].length;
        int c2 = second[0].length;
        double[][] product = new double[r1][c2];
        for(int i = 0; i < r1; i++) {
            for (int j = 0; j < c2; j++) {
                for (int k = 0; k < c1; k++) {
                    product[i][j] += first[i][k] * second[k][j];
                }
            }
        }
        return product;
    }

    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-1 * x));
    }
}
