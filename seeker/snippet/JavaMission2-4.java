//date: 2025-05-13T17:06:48Z
//url: https://api.github.com/gists/1e2c15c7e5ed11e85c9442f9341ad1a6
//owner: https://api.github.com/users/jyh1108

//정용화

import java.util.ArrayList;
import java.util.Scanner;

class Contact {
    private String name;
    private String phoneNumber;

    public Contact(String name, String phoneNumber) {
        this.name = name;
        this.phoneNumber = phoneNumber;
    }

    public String getName() {
        return name;
    }

    public String getPhoneNumber() {
        return phoneNumber;
    }

    public void setName(String name) {
        this.name = name;

    }

    public void setPhoneNumber(String phoneNumber) {
        this.phoneNumber = phoneNumber;
    }

}

public class Mission {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        ArrayList<Contact> contacts = new ArrayList<>();

        while (true) {
            System.out.println("연락처 관리 시스템 (add, remove, list, search, exit)");
            System.out.print("명령 입력 : ");
            String cm = sc.nextLine();
            switch (cm) {
                case "add":
                    System.out.print("이름 입력 :");
                    String name = sc.nextLine();
                    System.out.print("전화번호 : ");
                    String phone = sc.nextLine();
                    contacts.add(new Contact(name, phone));
                    System.out.println("연락처가 추가되었습니다.");
                    break;
                case "search":
                    System.out.print("검색할 이름 :");
                    String searchName = sc.nextLine();
                    boolean found = false;

                    for (Contact contact : contacts) {
                        if (contact.getName().equals(searchName)) {
                            System.out.println("찾은 연락처 : " + contact.getPhoneNumber());
                            found = true;
                        }
                    }
                    if (!found) {
                        System.out.println("연락처를 찾을 수 없습니다.");
                    }
                    break;
                case "list":
                    for (Contact contact : contacts) {
                        System.out.println(contact.getName() + " - " + contact.getPhoneNumber());
                    }
                    break;
                case "remove":
                    System.out.println("삭제할 이름 : ");
                    String removeName = sc.nextLine();
                    boolean removed = false;
                    for (int i = 0; i < contacts.size(); i++) {
                        if (contacts.get(i).getName().equals(removeName)) {
                            contacts.remove(i);
                            System.out.print("삭제 완료!");
                            removed = true;
                            break;
                        }
                    }
                    if (!removed) {
                        System.out.println("연락처를 찾을 수 없습니다.");
                    }
                    break;
                case "exit":
                    System.out.println("종료합니다.");
                    return;
                default:
                    System.out.println("명령어만 입력해 주세요");
            }
        }
    }
}
