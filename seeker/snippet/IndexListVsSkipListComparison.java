//date: 2024-05-27T17:04:06Z
//url: https://api.github.com/gists/57ed08551cbea8b812f8740c9a3ddb74
//owner: https://api.github.com/users/coderodde

package fi.helsinki.cs.rodionef.msc;

import com.github.coderodde.util.IndexedLinkedList;
import java.util.Map;
import java.util.concurrent.ConcurrentSkipListMap;


public class IndexListVsSkipListComparison {

    private static final IndexedLinkedList<Integer> iil = 
            new IndexedLinkedList<>();
    
    private static final Map<Integer, Boolean> cslm = 
            new ConcurrentSkipListMap<>();
    
    public static void main(String[] args) {
        System.out.println("Warming up...");
        System.out.println();
        warmup();
        System.out.println();
        benchmark();
    }
    
    private static void warmup() {
        runBenchmark(false);
    }
    
    private static void benchmark() {
        runBenchmark(true);
    }
    
    private static void runBenchmark(boolean doPrint) {
        iil.clear();
        cslm.clear();
        
        long durationIll = 0L;
        long durationCslm = 0L;
        
        // Ascending put:
        long ta = System.currentTimeMillis();
        
        for (int i = 0; i < 1_000_000; i++) {
            cslm.put(i, Boolean.TRUE);
        }
        
        long tb = System.currentTimeMillis();
        
        durationCslm += tb - ta;
        
        if (doPrint) {
            System.out.printf(
                    "ConcurrentSkipListMap.put in ascending order " + 
                            "%d milliseconds.\n",
                    tb - ta);
        }
        
        // Descending remove:
        ta = System.currentTimeMillis();
        
        for (int i = 999_999; i >= 0; i--) {
            cslm.remove(i);
        }
        
        tb = System.currentTimeMillis();
        
        durationCslm += tb - ta;
        
        if (doPrint) {
            System.out.printf(
                    "ConcurrentSkipListMap.remove (reverse order) "+ 
                            "in %d milliseconds.\n",
                    tb - ta);
        }
        
        // Descending put:
        ta = System.currentTimeMillis();
        
        for (int i = 999_999; i >= 0; i--) {
            cslm.put(i, Boolean.TRUE);
        }
        
        tb = System.currentTimeMillis();
        
        durationCslm += tb - ta;
        
        if (doPrint) {
            System.out.printf(
                    "ConcurrentSkipListMap.put (reverse order) "+ 
                            "in %d milliseconds.\n",
                    tb - ta);
        }
        
        // Ascending addLast:
        ta = System.currentTimeMillis();
        
        for (int i = 0; i < 1_000_000; i++) {
            iil.addLast(i);
        }
        
        tb = System.currentTimeMillis();
        
        durationIll += tb - ta;
        
        if (doPrint) {
            System.out.printf("IndexedLinkedList.addLast in %d milliseconds.\n",
                              tb - ta);
        }
        
        // Descending removeLast:
        ta = System.currentTimeMillis();
        
        for (int i = 999_999; i >= 0; i--) {
            iil.removeLast();
        }
        
        tb = System.currentTimeMillis();
        
        durationIll += tb - ta;
        
        if (doPrint) {
            System.out.printf(
                    "IndexedLinkedList.removeLast in %d milliseconds.\n",
                    tb - ta);
            
        }
        
        // Descending addFirst:
        ta = System.currentTimeMillis();
        
        for (int i = 999_999; i >= 0; i--) {
            iil.addFirst(i);
        }
        
        tb = System.currentTimeMillis();
        
        durationIll += tb - ta;
        
        if (doPrint) {
            System.out.printf(
                    "IndexedLinkedList.addFirst (reverse order) "+ 
                            "in %d milliseconds.\n",
                    tb - ta);
        }
        
        if (doPrint) {
            System.out.println("--- TOTAL ---");
            System.out.printf(
                    "ConcurrentSkipListMap: %d milliseconds.\n",
                    durationCslm);
            
            System.out.printf(
                    "IndexedLinkedList: %d milliseconds.\n",
                    durationIll);
        }
    }
}
