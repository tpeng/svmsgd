import cern.colt.matrix.impl.SparseDoubleMatrix1D;

/**
 * Created with IntelliJ IDEA.
 * User: tpeng
 * Date: 8/18/12
 * Time: 3:36 PM
 * To change this template use File | Settings | File Templates.
 */
public class DataSet {

    /** create a feature from RCV1-V2 data set */
    public static Instance create(String line, int size) {
        String[] values = line.split("\\s+");
        Instance instance = new Instance();

        SparseDoubleMatrix1D vector = new SparseDoubleMatrix1D(size);
        for (int i=1; i<values.length; i++) {
            int index = Integer.parseInt(values[i].split(":")[0]);
            double value = Double.parseDouble(values[i].split(":")[1]);
            vector.set(index, value);
        }
        instance.setLabel(Integer.parseInt(values[0]));
        instance.setFeatures(vector);
        return instance;
    }
}
