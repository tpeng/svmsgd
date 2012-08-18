import cern.colt.matrix.impl.SparseDoubleMatrix1D;

/**
 * Created with IntelliJ IDEA.
 * User: tpeng
 * Date: 8/18/12
 * Time: 3:29 PM
 * To change this template use File | Settings | File Templates.
 */
public class Instance {

    public int label;
    public SparseDoubleMatrix1D features;

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public SparseDoubleMatrix1D getFeatures() {
        return features;
    }

    public void setFeatures(SparseDoubleMatrix1D features) {
        this.features = features;
    }

    @Override
    public String toString() {
        return "Instance{" +
                "label=" + label +
                ", features=" + features +
                '}';
    }
}
