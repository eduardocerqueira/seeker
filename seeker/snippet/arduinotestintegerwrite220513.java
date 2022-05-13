//date: 2022-05-13T17:10:07Z
//url: https://api.github.com/gists/bc31f91cfeda6c0d1866d7275eca678e
//owner: https://api.github.com/users/iani

int testval;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  testval = 1;
  Serial.write(testval);
  delay(1000);
  testval = -1;
  delay(1000);
  Serial.write(testval);
  delay(1000);
  testval = 123456;
  Serial.write(testval);
  

}
