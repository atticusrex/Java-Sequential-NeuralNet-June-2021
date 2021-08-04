public class InputLayer extends Layer {

    public InputLayer(int n) {
        super(n);
    }

    public void setValues(double[] vals) {
        if (vals.length != this.size) {
            System.out.println("Dimension mismatch!");
        } else {
            this.values = Matrix.fromArray(vals);
        }
    }
}