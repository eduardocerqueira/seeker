//date: 2023-03-06T16:43:54Z
//url: https://api.github.com/gists/a5160c7563813205658aebb370e4e77a
//owner: https://api.github.com/users/IliyanOstrovski

public class MyStringBuilder {
    private char[] value;
    private int count;
    private int capacity;


    public MyStringBuilder() {
        value = new char[25];
    }

    public MyStringBuilder(int capacity) {
        value = new char[capacity];
        this.capacity = capacity;
    }

    public MyStringBuilder(String str) {
        this(str.length() + 16);
        append(str);
    }

    public MyStringBuilder(MyStringBuilder sb) {
        this(sb.length() + 16);
        append(sb);
    }

    public int capacity() {
        return capacity;
    }

    public int length() {
        return count;
    }

    public void ensureCapacity(int minimumCapacity) {
        if (minimumCapacity > capacity) {
            capacity = Math.max(minimumCapacity, capacity * 2);
            char[] newValue = new char[capacity];
            System.arraycopy(value, 0, newValue, 0, count);
            value = newValue;
        }
    }

    public void setCharAt(int index, char ch) {
        if (index < 0 || index >= count) {
            throw new StringIndexOutOfBoundsException(index);
        }
        value[index] = ch;
    }

    public void append(String str) {
        int len = str.length();
        ensureCapacity(count + len);
        str.getChars(0, len, value, count);
        count += len;
    }

    public void append(int num) {
        append(String.valueOf(num));
    }

    public void append(MyStringBuilder sb) {
        if (sb == null) {
            return;
        }
        int len = sb.length();
        ensureCapacity(count + len);
        System.arraycopy(sb.value, 0, value, count, len);
        count += len;
    }


    public int indexOf(String str) {
        return indexOf(str, 0);
    }

    public int indexOf(String str, int fromIndex) {
        int strLen = str.length();
        int max = count - strLen;
        for (int i = Math.min(fromIndex, max); i <= max; i++) {
            boolean found = true;
            for (int j = 0; j < strLen; j++) {
                if (value[i + j] != str.charAt(j)) {
                    found = false;
                    break;
                }
            }
            if (found) {
                return i;
            }
        }
        return -1;
    }

    public int lastIndexOf(String str) {
        return lastIndexOf(str, count);
    }

    public int lastIndexOf(String str, int fromIndex) {
        int min = fromIndex - str.length();
        if (min < 0) {
            min = 0;
        }
        String sub = str;
        for (int i = count - sub.length(); i >= min; i--) {
            if (substring(i, i + sub.length()).equals(sub)) {
                return i;
            }
        }
        return -1;
    }

    public void insert(int offset, String str) {
        if (offset < 0 || offset > count) {
            throw new StringIndexOutOfBoundsException(offset);
        }
        int len = str.length();
        ensureCapacity(count + len);
        System.arraycopy(value, offset, value, offset + len, count - offset);
        str.getChars(0, len, value, offset);
        count += len;
    }

    public void insert(int index, String str, int offset, int len) {
        if (index < 0 || index > count) {
            throw new StringIndexOutOfBoundsException(index);
        }
        if (offset < 0 || len < 0 || offset + len > str.length()) {
            throw new StringIndexOutOfBoundsException("offset " + offset + ", len " + len + ", str.length() " + str.length());
        }
        ensureCapacity(count + len);
        System.arraycopy(value, index, value, index + len, count - index);
        str.getChars(offset, offset + len, value, index);
        count += len;
    }

    public MyStringBuilder reverse() {
        int left = 0;
        int right = count - 1;
        while (left < right) {
            char temp = value[left];
            value[left] = value[right];
            value[right] = temp;
            left++;
            right--;
        }
        return this;
    }

    public void trimToSize() {
        if (count < value.length) {
            char[] newValue = new char[count];
            System.arraycopy(value, 0, newValue, 0, count);
            value = newValue;
        }
    }

    public MyStringBuilder replace(int start, int end, String str) {
        if (start < 0 || start > this.length() || end < start || end > this.length()) {
            throw new StringIndexOutOfBoundsException();
        }

        int len = end - start;
        int strLen = str.length();
        int newLength = this.length() - len + strLen;

        this.ensureCapacity(newLength);

        for (int i = 0; i < strLen; i++) {
            this.setCharAt(start + i, str.charAt(i));
        }

        // Delete any remaining characters after the end index
        this.delete(start + strLen, end);

        return this;
    }

    public String substring(int start, int end) {
        if (start < 0 || end > length() || start > end) {
            throw new StringIndexOutOfBoundsException();
        }
        return new String(value, start, end - start);
    }

    public String substring(int start) {
        if (start < 0 || start >= length()) {
            throw new StringIndexOutOfBoundsException();
        }
        return new String(value, start, length() - start);
    }

    public MyStringBuilder delete(int start, int end) {
        if (start < 0 || start > end || end > length()) {
            throw new StringIndexOutOfBoundsException();
        }
        int len = end - start;
        if (len > 0) {
            System.arraycopy(value, start + len, value, start, count - end);
            count -= len;
        }
        return this;
    }

    public MyStringBuilder delete(int start) {
        if (start < 0 || start >= length()) {
            throw new StringIndexOutOfBoundsException();
        }
        int len = length() - start;
        if (len > 0) {
            System.arraycopy(value, start + len, value, start, count - (start + len));
            count -= len;
        }
        return this;
    }

    @Override
    public String toString() {
        char[] nonNullValue = Arrays.copyOf(value, length());
        return new String(nonNullValue);
    }
}