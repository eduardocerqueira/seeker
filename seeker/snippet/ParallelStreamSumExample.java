//date: 2025-07-07T16:39:28Z
//url: https://api.github.com/gists/3781a22242576e7494d33874bd2d1383
//owner: https://api.github.com/users/yjAlpaka

import java.util.stream.LongStream;

public class ParallelStreamSumExample {
    public static void main(String[] args){

        long start=System.currentTimeMillis();
        long summSequential=LongStream.rangeClosed(1,500_000_000).sum();
        long end=System.currentTimeMillis();
        System.out.println("순차 스트림 합계: "+summSequential+" (시간: "+(end-start)+")");


        start=System.currentTimeMillis();
        long sumParallel= LongStream.rangeClosed(1,500_000_000).parallel().sum();
        end=System.currentTimeMillis();
        System.out.println("병렬 스트림 합계: "+sumParallel+" (시간: "+(end-start)+")");
    }
}
