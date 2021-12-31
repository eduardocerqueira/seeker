//date: 2021-12-31T16:37:14Z
//url: https://api.github.com/gists/d7b8a7ed67c12a4b4239de3e4f867523
//owner: https://api.github.com/users/rgbk21

class Host {
  private Symbiote symbiote;
  Host() {}
  public void setSymbiote(Symbiote symbiote) {this.symbiote = symbiote;}
  public void printHost() {System.out.println("Host loaded.");}
}

public class Symbiote {
  private Host host;
  Symbiote() {}
  public void setHost(Host host) {this.host = host;}
  public void printSymbiote() {System.out.println("Symbiote loaded.");}
}

class Test {
  public static void main(String[] args) {
    BeanFactory injector = new FileSystemXmlApplicationContext("src/main/resources/chapter3/setter/symbioteSetter.xml");
    Host host = (Host) injector.getBean("host");
    Symbiote symbiote = (Symbiote) injector.getBean("symbiote");
    host.printHost();
    symbiote.printSymbiote();
  }
}
