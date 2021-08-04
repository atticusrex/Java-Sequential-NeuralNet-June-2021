public class Layer {
    private static int seed = 1000000;
    public int size;
    public Neuron neurons[];
    public Layer outputLayer;
    public double bias;
    public Matrix weights;
    public Matrix values;
    public Matrix preValues;

    public Layer(int n) {
        size = n;
        neurons = new Neuron[n];
        bias = randGauss();
    }

    public void addOutputLayer(Layer l) {
        outputLayer = l;
        for (int i = 0; i < size; i++) {
            neurons[i] = new Neuron(l);
        }

        weights = new Matrix(outputLayer.size, size);
    }

    public void sendPulse() {
        Matrix pulse = Matrix.multiply(weights, values);
        outputLayer.receivePulse(pulse);
    }

    public void receivePulse(Matrix pulse) {
        this.preValues = pulse;
        this.values = this.preValues.sigmoid();
    }

    private double negativeSigmoid(double x) {
        return 2 / (1 + Math.exp(-1 * x)) - 1;
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-1 * x));
    }

    private static double ranf() {
        final int a = 16807, c = 2147483647, q = 127773, r = 2836;
        final double cd = c;
        int h = seed / q;
        int l = seed %q;
        int t = a * l - r * h;
        if (t > 0) seed = t;
        else seed = c + t;
        return seed / cd;
    }

    private static double randGauss() {

        double r1 = - Math.log(1 - ranf());
        double r2 = 2 * Math.PI * ranf();
        r1 = Math.sqrt(2 * r1);
        double x = r1 * Math.cos(r2);
        return x;
    }

}
