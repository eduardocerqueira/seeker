//date: 2025-05-28T17:11:53Z
//url: https://api.github.com/gists/79aea3866f2fceb85f59a3e938a2a993
//owner: https://api.github.com/users/thinkphp

class Student {

    //doua atribute
    private String name;
    private int matrikelnummer;

    public String getName() {
        return name;
    }

    public int getMatrikelNummer() {

        return matrikelnummer;
    }
}


class StudentOperations {

      //task2 a)
       public static boolean istEingetragen(Student student, List<Student> list) {

              //cautare secventiala
              for(Student s: list) {

                   if(s.getMatrikelNummer() == student.getMatrikelNummer()) {

                       return true;
                   }
              }
              return false;
       }

       //task2 b)
       public static boolean istEingetrangenOpt(Student student, List<Student> lista_sortata) {

              int index = 0;//pornim cu indexul 0

              //cautam pozitia de inserat in lista sortata
              while(index <= lista_sortata.size()) {

                     lista_sortata.get(index).getMatrikelNummer() < student.getMatrikelNummer() {

                           index++;
                     }
              }

              lista_sortata.add(index, student);
       }

       //cautare binara

       public static boolean cautare_binara(Student student, List<Student> lista_sortata) {

              int left = 0;
              int right = lista_sortata.size() - 1;

              int target = student.getMatrikelNummer();

              while(left <= right) {

                    int middle = left + (right - left) / 2;

                    int middle_matrikeNummer = student.getMatrikelNummer();

                    if(middle_matrikeNummer == target) {

                         return true;
                         
                    } else if(middle_matrikeNummer < target) {

                         left = middle + 1;                      
                         
                    } else {

                         right = middle - 1;
                    }
              }

              return false;

       }
}

/*
            lista_sortata = [George, Mihai, Cristian] ---lista_sortata.size() -->3
                             0         1      2


            lista = [ 11, 12, 13, 14, 15, 16, 17, 18, 19, 100, 111]

          index       0                                         10
          first = 0
          last = 10

          mijloc = 0 + 10

          search = 19

          lista[5] == search 16 == 19 ?

          cautare binara = algoritm performant



*/
