#include <Wire.h>

void setup() {
  Wire.begin();
  Serial.begin(115200);
  delay(500);

  Serial.println("I2C scan...");
  byte found = 0;

  for (byte addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    byte err = Wire.endTransmission();
    if (err == 0) {
      Serial.print("Found: 0x");
      if (addr < 16) Serial.print("0");
      Serial.println(addr, HEX);
      found++;
    }
  }

  if (found == 0) Serial.println("No I2C devices found");
}

void loop() {}
