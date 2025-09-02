#include <Arduino.h>

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);  // Set the LED_BUILTIN pin as output
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH);  // Turn the LED on
  delay(1000);                      // Wait for 1 second
  digitalWrite(LED_BUILTIN, LOW);   // Turn the LED off
  delay(1000);                      // Wait for 1 second
}
