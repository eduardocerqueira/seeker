//date: 2023-02-03T17:10:38Z
//url: https://api.github.com/gists/a94c18a5fc3b5cde32cd16c585871a11
//owner: https://api.github.com/users/muhamedoufi

class CheesePizza implements Pizza {
  @Override
  public void prepare() {
    System.out.println("Préparation de la pizza au fromage");
  }

  @Override
  public void bake() {
    System.out.println("Cuisson de la pizza au fromage");
  }

  @Override
  public void cut() {
    System.out.println("Coupe de la pizza au fromage");
  }
}