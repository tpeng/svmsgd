import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.jet.math.Functions;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;

/**
 * A SVM solver with SGD (stochastic gradient descent). the data is being used is RCV1-V2.
 * see http://leon.bottou.org/projects/sgd
 * <p/>
 * The format of one training example per line looks like:
 * <p/>
 * y f_1:v_1 f_2:v_2 ... f_n:v_n
 * <p/>
 * where y is either 1 or -1, each f_i is an integer representing a feature index
 * and each v_i a float.
 * <p/>
 * Author: tpeng <pengtaoo@gmail.com>
 * Date: 8/18/12 3:08 PM
 */
public class SvmSgd {

    public final static int MAX_FEATURE = 47153;

    private double lambda = 0.0001;

    // the count of instance being trained
    private int t = 0;

    // the count of errors
    private int errors = 0;

    // the weight to learn
    private DenseDoubleMatrix1D weight = new DenseDoubleMatrix1D(MAX_FEATURE);

    private String dataPath;

    public SvmSgd(String data) {
        this.dataPath = data;
    }

    public double inner(DoubleMatrix1D a, DoubleMatrix1D b) {
        return a.zDotProduct(b);
    }

    // a[i] = a[i] * b;
    public void scaleInPlace(DoubleMatrix1D a, double b) {
        a.assign(Functions.mult(b));
    }

    public DoubleMatrix1D scale(DoubleMatrix1D a, double b) {
        return a.copy().assign(Functions.mult(b));
    }

    // a[i] = a[i] + b[i]
    public void addInPlace(DoubleMatrix1D a, DoubleMatrix1D b) {
        a.assign(b, Functions.plus);
    }

    /**
     * get the hinge loss with current model
     */
    public double hinge(Instance instance) {
        int label = instance.getLabel();
        SparseDoubleMatrix1D v = instance.getFeatures();
        return Math.max(0, 1 - label * inner(weight, v));
    }

    /**
     * correct the model with gradient descent method
     */
    private void correct(Instance instance) {
        // w(t+1) = w(t) - learning_rate * gradient of loss function
        // learning_rate is 1 / lambda * t
        // gradient is lambda * weight - sum(y*x) / t
        int y = instance.getLabel();
        SparseDoubleMatrix1D xs = instance.getFeatures();
        scaleInPlace(weight, 1 - 1.0 / t);
        addInPlace(weight, scale(xs, y * 1.0 / (lambda * t)));
    }

    /**
     * update the model base on current model and predication
     */
    private void update(Instance instance) {
        double error = hinge(instance);
        t += 1;
        if (error > 0) {
            errors += 1;
            correct(instance);
        }
        status(100);
    }

    private void status(int interval) {
        if (t % interval == 0) {
            System.out.print("step: " + t);
            System.out.print("\terrors: " + errors);
            System.out.print("\taccuracy: " + (1 - (1.0 * errors / t)));
            System.out.println();
        }
    }

    public void train() throws IOException {
        GZIPInputStream zip = new GZIPInputStream(new FileInputStream(dataPath));
        BufferedReader br = new BufferedReader(new InputStreamReader(zip));
        String line;
        while ((line = br.readLine()) != null) {
            Instance feature = DataSet.create(line, MAX_FEATURE);
            update(feature);
        }
    }

    public double[] getWeight() {
        return weight.toArray();
    }

    public static void main(String... args) throws IOException {
        SvmSgd sgd = new SvmSgd("data/train2000.dat.gz");
//        SvmSgd sgd = new SvmSgd("//Users/tpeng/bottou-sgd/svm/rcv1.train.txt.gz");
        sgd.train();
        System.out.println(Arrays.toString(sgd.getWeight()));
    }
}
