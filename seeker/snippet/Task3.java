//date: 2023-02-28T17:08:15Z
//url: https://api.github.com/gists/eb58f202d7b94f2558b194c6a4ed60c1
//owner: https://api.github.com/users/IliyanOstrovski

public class Task3 {
    public void main(String[] args) {
        Range r1 = new Range(1, 3);
        Range r2 = new Range(2, 5);
        r1.merge(r2);
        System.out.println(r1);
    }

    public static class Range {
        private int start;
        private int end;

        public Range(int start, int end) {
            this.start = start;
            this.end = end;
        }

        public boolean contains(int n) {
            return n >= start && n <= end;
        }

        public boolean overlaps(Range r) {
            return this.contains(r.start)
                    || this.contains(r.end)
                    || r.contains(this.start)
                    || r.contains(this.end);
        }

        public boolean merge(Range r) {
            if (this.overlaps(r)) {
                this.start = Math.min(this.start, r.start);
                this.end = Math.max(this.end, r.end);
                return true;
            }
            return false;
        }

        @Override
        public String toString() {
            return "[" + start + ", " + end + "]";
        }
    }

}
