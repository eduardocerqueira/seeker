//date: 2025-11-21T17:12:20Z
//url: https://api.github.com/gists/0c63eed77f4ab81a21718d18a984dbc4
//owner: https://api.github.com/users/Pratiksha02-hub

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.Comparator;
import java.util.PriorityQueue;

class Student {
    private int id;
    private String name;
    private double cgpa;

    public Student(int id, String name, double cgpa) {
        this.id = id;
        this.name = name;
        this.cgpa = cgpa;
    }

    public int getID() {
        return id;
    }

    public String getName() {
        return name;
    }

    public double getCGPA() {
        return cgpa;
    }
}


class Priorities {

    public List<Student> getStudents(List<String> events) {

        Comparator<Student> comp = new Comparator<Student>() {
            public int compare(Student a, Student b) {
                if (a.getCGPA() < b.getCGPA()) return 1;
                if (a.getCGPA() > b.getCGPA()) return -1;

                int nameCompare = a.getName().compareTo(b.getName());
                if (nameCompare != 0) return nameCompare;

                return a.getID() - b.getID();
            }
        };

        // FIX: include initial capacity + comparator
        PriorityQueue<Student> pq = new PriorityQueue<Student>(1000, comp);

        for (String e : events) {
            String[] parts = e.split(" ");
            String type = parts[0];

            if (type.equals("ENTER")) {
                String name = parts[1];
                double cgpa = Double.parseDouble(parts[2]);
                int id = Integer.parseInt(parts[3]);
                pq.add(new Student(id, name, cgpa));
            } else if (type.equals("SERVED")) {
                pq.poll();
            }
        }

        List<Student> remaining = new ArrayList<Student>();
        while (!pq.isEmpty()) {
            remaining.add(pq.poll());
        }

        return remaining;
    }
}