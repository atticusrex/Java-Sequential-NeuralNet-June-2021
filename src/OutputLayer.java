public class OutputLayer extends Layer {
    public Matrix expectedVals;

    public OutputLayer(int n) {
        super(n);
        outputLayer = new Layer(0);
        neurons = new Neuron[n];
        for (int i = 0; i < size; i++) {
            neurons[i] = new Neuron(outputLayer);
        }
    }

    public double error() {
        double error = 0;
        for (int i = 0; i < size; i++) {
            error += .5 * Math.pow(expectedVals.get(i, 0) - values.get(i, 0), 2);
        }
        return error;
    }

    public void setExpected(double vals[]) {
        if (vals.length != size) {
            System.out.println("Dimension mismatch!");
        } else {
            expectedVals = Matrix.fromArray(vals);
        }
    }
}
