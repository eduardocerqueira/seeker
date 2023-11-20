//date: 2023-11-20T16:27:30Z
//url: https://api.github.com/gists/4edc6d8dd01c27216407cfaf71b96e24
//owner: https://api.github.com/users/Johannaeden

For example, if you have a Student class with one attribute (private int grade), and you have an array of these students (Student[] students = {new Student(...), ...}) then you could sort these students in ascending order through this:

static Student[] insertionSort(Student[] array) {
        Student[] result = new Student[array.length];
        for (int endIndex = 0; endIndex < array.length; endIndex++) {
            // begin of insert
            int insertIndex = 0;
            while (insertIndex < endIndex && array[endIndex].getGrade() > result[insertIndex].getGrade()) {
                insertIndex ++;
            }
            for (int i = endIndex - 1; i >= insertIndex; i--) {
                result[i + 1] = result[i];
            }
            result[insertIndex] = array[endIndex];
            // end of insert
        }
        return result;
    }
This would return an array of students sorted in ascending order by their grades. The only thing that's different from the int[] array is that I'm now getting the attribute that I want the sort the students by in the while loop there, so instead of array[endIndex]) > result[insertIndex], I'm using array[endIndex].getAttribute() > result[insertIndex].getAttribute().
This is it for descending order:

static Student[] insertionSort(Student[] array) {
        Student[] result = new Student[array.length];
        for (int endIndex = 0; endIndex < array.length; endIndex++) {
            // begin of insert
            int insertIndex = 0;
            while (insertIndex < endIndex && array[endIndex].getGrade() < result[insertIndex].getGrade()) {
                insertIndex ++;
            }
            for (int i = endIndex - 1; i >= insertIndex; i--) {
                result[i + 1] = result[i];
            }
            result[insertIndex] = array[endIndex];
            // end of insert
        }
        return result;
    }

The only thing that's changed is the rightmost > in the while loop, which is a < now that it's descending
You can use this for any other type that has an attribute, not just for students and grades (for example you can sort engines by their horsepower)