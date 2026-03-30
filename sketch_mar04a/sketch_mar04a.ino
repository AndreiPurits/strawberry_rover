#include <Wire.h>
#include <Adafruit_MS_PWMServoDriver.h>

Adafruit_MS_PWMServoDriver pwm = Adafruit_MS_PWMServoDriver(); // адрес по умолчанию 0x40

// Под твои сервы можно потом подстроить:
static const uint16_t SERVO_MIN = 60;  // ~0.6ms
static const uint16_t SERVO_MAX = 800;  // ~2.4ms

// angle: 0..180 -> ticks 0..4095 при 50 Hz
uint16_t angleToTicks(int angle) {
  angle = constrain(angle, 0, 180);
  return map(angle, 0, 180, SERVO_MIN, SERVO_MAX);
}

void setup() {
  Wire.begin();
  pwm.begin();
  pwm.setPWMFreq(50);     // частота серв
  delay(10);

  // стартовая позиция
  pwm.setPWM(0, 0, angleToTicks(90));   // канал 0 = Servo0
}

void loop() {
  // небольшой "луп" вокруг центра
  pwm.setPWM(0, 0, angleToTicks(80));
  delay(400);
  pwm.setPWM(0, 0, angleToTicks(100));
  delay(400);
}
