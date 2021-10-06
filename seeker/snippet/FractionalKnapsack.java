//date: 2021-10-06T16:51:49Z
//url: https://api.github.com/gists/aeece6b48f46756e610f6f17ac41b5a2
//owner: https://api.github.com/users/ravindraranwala

class FractionalKnapsack {
    FractionalKnapsack() {
	throw new AssertionError();
    }

    public static void main(String[] args) {
	final int[] w = { 20, 10, 30 };
	final int[] v = { 100, 60, 120 };
	final double p = knapsack(w, v, 50);
	System.out.println(p);
    }

    static double knapsack(int[] w, int[] v, int c) {
	final int n = w.length;
	final double[] d = new double[n];
	for (int i = 0; i < n; i++)
	    d[i] = (double) v[i] / w[i];
	return knapsack(w, v, c, 0, n - 1, d);
    }

    static double knapsack(int[] w, int[] v, int c, int p, int r, double[] d) {
	if (r < p)
	    return 0;
	final int[] k = randomizedPartition(w, v, d, p, r);
	final int q = k[0];
	if (c < k[1])
	    return knapsack(w, v, c, q + 1, r, d);
	else if (k[1] + w[q] < c)
	    return k[2] + v[q] + knapsack(w, v, c - k[1] - w[q], p, q - 1, d);
	else
	    return k[2] + (double) (c - k[1]) / w[q] * v[q];
    }

    static int[] partition(int[] w, int[] v, double[] d, int p, int r) {
	final double x = d[r];
	int i = p - 1;
	int lw = 0;
	int lv = 0;
	int wSum = 0;
	int vSum = 0;
	for (int j = p; j < r; j++) {
	    wSum = wSum + w[j];
	    vSum = vSum + v[j];
	    if (d[j] <= x) {
		i = i + 1;
		lw = lw + w[j];
		lv = lv + v[j];
		// exchange d[i] with d[j]
		final double tmpD = d[i];
		d[i] = d[j];
		d[j] = tmpD;
		// exchange w[i] with w[j]
		final int tmpW = w[i];
		w[i] = w[j];
		w[j] = tmpW;
		// exchange v[i] with v[j]
		final int tmpV = v[i];
		v[i] = v[j];
		v[j] = tmpV;
	    }
	}
	// exchange d[i + 1] with d[r]
	final double rPivot = d[r];
	d[r] = d[i + 1];
	d[i + 1] = rPivot;

	// exchange w[i + 1] with w[r]
	final int wPivot = w[r];
	w[r] = w[i + 1];
	w[i + 1] = wPivot;

	// exchange v[i + 1] with v[r]
	final int vPivot = v[r];
	v[r] = v[i + 1];
	v[i + 1] = vPivot;

	final int[] res = { i + 1, wSum - lw, vSum - lv };
	return res;
    }

    static int[] randomizedPartition(int[] w, int[] v, double[] d, int p, int r) {
	final int i = ThreadLocalRandom.current().nextInt(p, r + 1);
	final double pivotD = d[r];
	d[r] = d[i];
	d[i] = pivotD;

	final int pivotW = w[r];
	w[r] = w[i];
	w[i] = pivotW;

	final int pivotV = v[r];
	v[r] = v[i];
	v[i] = pivotV;

	return partition(w, v, d, p, r);
    }
}