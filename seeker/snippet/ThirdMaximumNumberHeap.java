//date: 2023-07-24T16:45:04Z
//url: https://api.github.com/gists/f2615c8764410b65dc07bb0ed8c2ca0f
//owner: https://api.github.com/users/Andres-Fuentes-Encora

class ThirdMaximumNumberHeap {
    public int thirdMax(int[] nums) {
        PriorityQueue<Integer> heap = new PriorityQueue<>();
        Set<Integer> set = new HashSet<>();
        for(int num : nums){
            if(set.contains(num)){
                continue;
            }
            if(heap.size() == 3){
                if(heap.peek() < num){
                    set.remove(heap.poll());
                    heap.add(num);
                    set.add(num);
                }
            } else {
                heap.add(num);
                set.add(num);
            }
        }


        if(heap.size() == 1){
            return heap.poll();
        } else if(heap.size() == 2){
            return Math.max(heap.poll(), heap.poll());
        }

        return heap.poll();
    }
}