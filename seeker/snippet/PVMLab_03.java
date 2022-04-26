//date: 2022-04-26T16:58:34Z
//url: https://api.github.com/gists/284b7730791ef72599048587fbeee435
//owner: https://api.github.com/users/BLADUS

package com.company;
import java.io.*;
import java.lang.reflect.Array;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Scanner;


public class Lab_03 {
    public static Scanner s_int = new Scanner(System.in);
    public static Scanner s_str = new Scanner(System.in);

    public static void mainMenu(File file) throws IOException {
        String textFile = "";
        String packageName;
        int swch;

        System.out.println("Выберете нужный пункт меню:");
        System.out.println("1.Вывод абсолютного пути для текущего файла или каталога");
        System.out.println("2. Вывод содержимого каталога и выбор нового файла");
        System.out.println("3. Вывод всей возможной информации для заданного файла");
        System.out.println("4. Создание нового каталога или файла по заданному пути");
        System.out.println("5. Создание копии файла по заданному пути");
        System.out.println("6. Вывод списка файлов текущего каталога, имеющих разрешение, задаваемое пользователем");
        System.out.println("7. Удаление файла или каталога");
        System.out.println("8. Поиск файла в каталоге");
        System.out.println("9.Запись в файл");
        System.out.println("10.Вывод из файла");
        System.out.println("11.Выход");
        swch= s_int.nextInt();
        switch (swch){
            case 1:
                System.out.println(file.getAbsoluteFile());
                break;

            case 2:
                file = serchFile();
            break;
            case 3:
                System.out.println("Имя файла:"+file.getName());
                System.out.println("Родительская папка:"+file.toPath());
                break;
            case 4:
                createFileorDirectory();
                break;
            case 5:
               File file_copy =copyFile();
               break;
            case 6:
                extensions();
                break;
            case 7:
                deleteFileOrDirectory();
                break;
            case 8:
                 detectedFile();
                break;
            case 9:
                writeFile();
                break;
            case 10:
                readFile();
                break;
            case 11:return;
            default:
                System.out.println("Вы выбрали не указанный тип меню");
                break;
        }
        mainMenu(file);
    }

    public static void createFileorDirectory() throws IOException {
        System.out.println("Что вы желаете создать:");
        System.out.println("1.Файл");
        System.out.println("2.Директория");
        int i = s_int.nextInt();
        if(i==1){
        File direcroty=searchPackage();
        String pathDir=direcroty.getAbsolutePath();
            System.out.print("Введите имя файла:");
            String file_name=s_str.nextLine();
            File file = new File(pathDir+"\\"+file_name);
            if(!file.exists()){
                file.createNewFile();
            }
        }
        if(i==2) {
            System.out.println("Дайте название каталога:");
            String directoryName = s_str.nextLine();
            File directory = new File("C:\\Users\\Osada\\IdeaProjects\\PVM\\" + directoryName);
            if (!directory.exists()) {
                directory.mkdirs();
            }
        }
    }

    public static File searchPackage(){
        System.out.println("Выберите каталог:");
        File ideaDirectory = new File("C:\\Users\\Osada\\IdeaProjects\\PVM");
        for (File file:ideaDirectory.listFiles()){
            System.out.print(file.getName()+" || ");
        }
        String packageName= s_str.nextLine();
        String pathPackage="C:\\Users\\Osada\\IdeaProjects\\PVM\\"+packageName;
        File folder = new File(pathPackage);
        return folder;
    }

    public static File serchFile(){
        File folder=searchPackage();
        System.out.print("Выберите нужный файл:");
        for (File file : folder.listFiles())
        {
            System.out.print(file.getName()+" || ");
        }
         String fileName=s_str.nextLine();
         File file = new File(fileName);
         return file;
    }

    public static void deleteFileOrDirectory(){
        System.out.println("Выберите что хотите удалить:");
        System.out.println("1.Директорию");
        System.out.println("2.Файл");
        int i = s_int.nextInt();
        switch (i){
            case 1:
                File directory=searchPackage();
                System.out.println("Директория "+directory.getName()+" была удалена");
                directory.delete();
                break;
            case 2:
                File file=serchFile();
                System.out.println("Файл "+file.getName()+" был удален");
                file.delete();
                break;
            default:break;
        }

    }

    public static File copyFile() throws IOException {
        System.out.println("Выберите файл для копирования:");
        File file =serchFile();
        String nameFile=file.getName();
        System.out.println("Выберите в какую папку вы хотите скопировать:");
        File directory=searchPackage();
        String pathDir=directory.getAbsolutePath();
        File file_copy=new File(pathDir+"\\"+nameFile);
        file_copy.createNewFile();
        return file_copy;
    }

    public static void detectedFile(){
       File folder =searchPackage();
       String pathFolder=folder.getAbsolutePath();
        System.out.println("Введите имя файла который вы хотите найти:");
        String nameFile=s_str.nextLine();
        File file = new File(pathFolder+"\\"+nameFile);
        if(file.exists())
            System.out.println("Файл найден!");
        else System.out.println("Файл не найден :(");


    }

    public static void extensions(){
        System.out.println("Введите нужное расширение для файла:");
        String ext = s_str.nextLine();
        String separator=".";
        File folder = searchPackage();
        System.out.print("Файлы с вашим расширением:");
        for (File file:folder.listFiles()) {
            String fileName=file.getName();
           if(fileName.endsWith(separator+ext))
               System.out.print(fileName+" || ");
           else continue;
            System.out.println();
        }

    }

    public static void writeFile() throws FileNotFoundException {
        File file=serchFile();
        PrintWriter pw =new PrintWriter(file);
        System.out.println("Введите нужный вам текст:");
        String text=s_str.nextLine();
        pw.write(text);
        pw.close();
    }

    public static void readFile() throws IOException {
        File file = serchFile();
      BufferedReader bf = new BufferedReader(new FileReader(file));
      String line;
        System.out.println("Текст из файла:");
      while((line = bf.readLine()) != null){
          System.out.println(line);
      }

    }

    public static void main(String[] args) throws IOException {
        File file = serchFile();
        mainMenu(file);
    }
}