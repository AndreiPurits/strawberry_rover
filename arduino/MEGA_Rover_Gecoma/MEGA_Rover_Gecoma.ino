/*
  MEGA Rover — Gecoma (dual track H-bridge, serial control)
  ---------------------------------------------------------
  Motor driver (red module) — same wiring as working test sketch:
    D0 -> Mega D4  LEFT  IN1
    D1 -> Mega D5  LEFT  IN2
    D2 -> Mega D6  RIGHT IN1
    D3 -> Mega D7  RIGHT IN2
    GND -> GND, 3V3 -> 3.3V, VT -> 5V

  Forward (both tracks):
    LEFT:  D4=HIGH D5=LOW   |  RIGHT: D6=HIGH D7=LOW
  Backward:
    LEFT:  D4=LOW  D5=HIGH |  RIGHT: D6=LOW  D7=HIGH
  Stop:
    all D4..D7 = LOW

  Peripherals (Gecoma field wiring — see project photos):
    Current sensor (blue):  A0 analog + module D0 -> Mega D22
    DHT11 (white):          OUT -> D23, + -> 5V, - -> GND
    Vibration / RFID (blk): IN -> D24, VCC -> 5V, GND -> GND

  STATUS JSON: armed, tracks, current_a0, current_d22, temp_c, humidity_pct, vibration_d24
    PING                          -> PONG
    ARM / DISARM                  -> OK
    M FL=.. FR=.. RL=.. RR=..     -> OK  (only when ARMED)
      Orin sends FL=RL=left, FR=RR=right (1000..2000, neutral 1500)
    STATUS                        -> JSON

  SAFE BOOT: DISARMED, stopMotors() — no auto cycle, no movement until ARM.
*/

#include <Arduino.h>

static const uint32_t BAUD = 115200;

// --- Track H-bridge pins (swap LEFT<->RIGHT pairs here after drive test) ---
static const uint8_t LEFT_IN1  = 4;  // was RIGHT: use 6
static const uint8_t LEFT_IN2  = 5;  // was RIGHT: use 7
static const uint8_t RIGHT_IN1 = 6;  // was LEFT:  use 4
static const uint8_t RIGHT_IN2 = 7;  // was LEFT:  use 5

static const uint8_t PIN_CURRENT_D22 = 22;  // current module D0 (digital)
static const uint8_t PIN_DHT11 = 23;       // DHT11 OUT
static const uint8_t PIN_VIBRATION = 24;   // vibration / RFID IN (active HIGH)
static const uint32_t DHT_INTERVAL_MS = 2000;

static float gTempC = NAN;
static float gHumPct = NAN;
static bool gDhtOk = false;
static uint32_t gLastDhtMs = 0;

static const int PWM_MIN = 1000;
static const int PWM_NEU = 1500;
static const int PWM_MAX = 2000;
static const int DEADBAND = 25;
static const uint32_t FAILSAFE_TIMEOUT_MS = 500;
static const uint32_t MOTOR_CYCLE_MS = 50;

static volatile int gFL = PWM_NEU;
static volatile int gFR = PWM_NEU;
static volatile int gRL = PWM_NEU;
static volatile int gRR = PWM_NEU;
static volatile uint32_t gLastCmdMs = 0;
static bool gArmed = false;

static char line[128];
static uint8_t idx = 0;

static inline int clamp_us(int v) {
  if (v < PWM_MIN) return PWM_MIN;
  if (v > PWM_MAX) return PWM_MAX;
  return v;
}

static void stopMotors() {
  digitalWrite(LEFT_IN1,  LOW);
  digitalWrite(LEFT_IN2,  LOW);
  digitalWrite(RIGHT_IN1, LOW);
  digitalWrite(RIGHT_IN2, LOW);
}

static int motionPct(int us) {
  int d = abs(us - PWM_NEU);
  if (d <= DEADBAND) return 0;
  int span = (PWM_MAX - PWM_NEU) - DEADBAND;
  if (span <= 0) return 0;
  return constrain((d - DEADBAND) * 100 / span, 1, 100);
}

static void driveSide(uint8_t in1, uint8_t in2, int us) {
  if (!gArmed || motionPct(us) == 0) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    return;
  }
  const bool forward = us > PWM_NEU;
  const int pct = motionPct(us);
  const uint32_t phase = millis() % MOTOR_CYCLE_MS;
  const uint32_t onMs = (MOTOR_CYCLE_MS * (uint32_t)pct) / 100U;
  if (phase >= onMs) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    return;
  }
  if (forward) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
  }
}

static void applyTracks() {
  if (!gArmed) {
    stopMotors();
    return;
  }
  if ((uint32_t)(millis() - gLastCmdMs) > FAILSAFE_TIMEOUT_MS) {
    gFL = gFR = gRL = gRR = PWM_NEU;
  }
  // Orin: FL=RL=left track, FR=RR=right track
  int leftUs  = (gFL + gRL) / 2;
  int rightUs = (gFR + gRR) / 2;
  // Keep failsafe alive while a non-neutral command is active (no need to spam M).
  if (gArmed && (abs(leftUs - PWM_NEU) > DEADBAND || abs(rightUs - PWM_NEU) > DEADBAND)) {
    gLastCmdMs = millis();
  }
  driveSide(LEFT_IN1,  LEFT_IN2,  leftUs);
  driveSide(RIGHT_IN1, RIGHT_IN2, rightUs);
}

static bool is_digit(char c) { return c >= '0' && c <= '9'; }

static bool parse_int_after(const char* s, int start, int& out) {
  int i = start;
  while (s[i] && !is_digit(s[i]) && s[i] != '-') i++;
  if (!s[i]) return false;
  bool neg = (s[i] == '-');
  if (neg) i++;
  if (!is_digit(s[i])) return false;
  long v = 0;
  while (s[i] && is_digit(s[i])) {
    v = v * 10 + (s[i] - '0');
    i++;
  }
  out = neg ? (int)-v : (int)v;
  return true;
}

static bool parse_cmd_m(const char* s, int& FL, int& FR, int& RL, int& RR) {
  bool hasM = false, hasFL = false, hasFR = false, hasRL = false, hasRR = false;
  int fl = PWM_NEU, fr = PWM_NEU, rl = PWM_NEU, rr = PWM_NEU;
  for (int i = 0; s[i]; i++) {
    if (s[i] == 'M' || s[i] == 'm') { hasM = true; break; }
  }
  if (!hasM) return false;
  for (int i = 0; s[i]; i++) {
    int v;
    if ((s[i] == 'F' || s[i] == 'f') && (s[i + 1] == 'L' || s[i + 1] == 'l') && !hasFL) {
      if (parse_int_after(s, i + 2, v)) { fl = v; hasFL = true; }
    }
    if ((s[i] == 'F' || s[i] == 'f') && (s[i + 1] == 'R' || s[i + 1] == 'r') && !hasFR) {
      if (parse_int_after(s, i + 2, v)) { fr = v; hasFR = true; }
    }
    if ((s[i] == 'R' || s[i] == 'r') && (s[i + 1] == 'L' || s[i + 1] == 'l') && !hasRL) {
      if (parse_int_after(s, i + 2, v)) { rl = v; hasRL = true; }
    }
    if ((s[i] == 'R' || s[i] == 'r') && (s[i + 1] == 'R' || s[i + 1] == 'r') && !hasRR) {
      if (parse_int_after(s, i + 2, v)) { rr = v; hasRR = true; }
    }
  }
  if (!(hasFL && hasFR && hasRL && hasRR)) return false;
  FL = clamp_us(fl); FR = clamp_us(fr); RL = clamp_us(rl); RR = clamp_us(rr);
  return true;
}

static bool str_eq_i(const char* a, const char* b) {
  while (*a && *b) {
    char ca = *a, cb = *b;
    if (ca >= 'A' && ca <= 'Z') ca += 32;
    if (cb >= 'A' && cb <= 'Z') cb += 32;
    if (ca != cb) return false;
    a++; b++;
  }
  return *a == 0 && *b == 0;
}

static void pollSensors() {
  if ((uint32_t)(millis() - gLastDhtMs) < DHT_INTERVAL_MS) return;
  gLastDhtMs = millis();

  uint8_t data[5] = {0};
  pinMode(PIN_DHT11, OUTPUT);
  digitalWrite(PIN_DHT11, LOW);
  delay(20);
  digitalWrite(PIN_DHT11, HIGH);
  delayMicroseconds(40);
  pinMode(PIN_DHT11, INPUT_PULLUP);

  if (digitalRead(PIN_DHT11) != LOW) {
    gDhtOk = false;
    return;
  }
  unsigned long t0 = micros();
  while (digitalRead(PIN_DHT11) == LOW) {
    if (micros() - t0 > 100) { gDhtOk = false; return; }
  }
  t0 = micros();
  while (digitalRead(PIN_DHT11) == HIGH) {
    if (micros() - t0 > 100) { gDhtOk = false; return; }
  }

  for (int i = 0; i < 40; i++) {
    t0 = micros();
    while (digitalRead(PIN_DHT11) == LOW) {
      if (micros() - t0 > 100) { gDhtOk = false; return; }
    }
    unsigned long hi = micros();
    while (digitalRead(PIN_DHT11) == HIGH) {
      if (micros() - hi > 100) { gDhtOk = false; return; }
    }
    unsigned long dur = micros() - hi;
    data[i / 8] <<= 1;
    if (dur > 40) data[i / 8] |= 1;
  }

  uint8_t sum = data[0] + data[1] + data[2] + data[3];
  if (sum != data[4]) {
    gDhtOk = false;
    return;
  }
  gHumPct = data[0];
  gTempC = data[2] + data[3] * 0.1f;
  gDhtOk = true;
}

static void printStatus() {
  int leftUs  = (gFL + gRL) / 2;
  int rightUs = (gFR + gRR) / 2;
  int leftPct = motionPct(leftUs);
  int rightPct = motionPct(rightUs);
  int d22 = digitalRead(PIN_CURRENT_D22);
  int vib = digitalRead(PIN_VIBRATION) ? 1 : 0;
  int a0 = analogRead(A0);
  Serial.print(F("{\"armed\":"));
  Serial.print(gArmed ? F("true") : F("false"));
  Serial.print(F(",\"fl_us\":"));
  Serial.print(gFL);
  Serial.print(F(",\"fr_us\":"));
  Serial.print(gFR);
  Serial.print(F(",\"rl_us\":"));
  Serial.print(gRL);
  Serial.print(F(",\"rr_us\":"));
  Serial.print(gRR);
  Serial.print(F(",\"left_us\":"));
  Serial.print(leftUs);
  Serial.print(F(",\"right_us\":"));
  Serial.print(rightUs);
  Serial.print(F(",\"left_power_pct\":"));
  Serial.print(leftPct);
  Serial.print(F(",\"right_power_pct\":"));
  Serial.print(rightPct);
  Serial.print(F(",\"current_a0\":"));
  Serial.print(a0);
  Serial.print(F(",\"current_d22\":"));
  Serial.print(d22);
  Serial.print(F(",\"current_d0\":"));
  Serial.print(d22);
  Serial.print(F(",\"vibration_d24\":"));
  Serial.print(vib);
  Serial.print(F(",\"dht_ok\":"));
  Serial.print(gDhtOk ? F("true") : F("false"));
  if (gDhtOk) {
    Serial.print(F(",\"temp_c\":"));
    Serial.print(gTempC, 1);
    Serial.print(F(",\"humidity_pct\":"));
    Serial.print(gHumPct, 1);
  }
  Serial.println(F("}"));
}

static void handleLine() {
  line[idx] = 0;
  idx = 0;

  if (str_eq_i(line, "PING")) {
    Serial.println("PONG");
    return;
  }
  if (str_eq_i(line, "ARM")) {
    gArmed = true;
    gLastCmdMs = millis();
    gFL = gFR = gRL = gRR = PWM_NEU;
    stopMotors();
    Serial.println("OK");
    return;
  }
  if (str_eq_i(line, "DISARM")) {
    gArmed = false;
    gFL = gFR = gRL = gRR = PWM_NEU;
    stopMotors();
    Serial.println("OK");
    return;
  }
  if (str_eq_i(line, "STATUS")) {
    printStatus();
    return;
  }

  if (!gArmed) {
    Serial.println("ERR:DISARMED");
    return;
  }

  int nfl, nfr, nrl, nrr;
  if (parse_cmd_m(line, nfl, nfr, nrl, nrr)) {
    gFL = nfl; gFR = nfr; gRL = nrl; gRR = nrr;
    gLastCmdMs = millis();
    Serial.println("OK");
    return;
  }
  Serial.println("ERR:PARSE");
}

void setup() {
  pinMode(LEFT_IN1,  OUTPUT);
  pinMode(LEFT_IN2,  OUTPUT);
  pinMode(RIGHT_IN1, OUTPUT);
  pinMode(RIGHT_IN2, OUTPUT);
  stopMotors();

  pinMode(PIN_CURRENT_D22, INPUT);
  pinMode(PIN_DHT11, INPUT_PULLUP);
  pinMode(PIN_VIBRATION, INPUT);

  gArmed = false;
  gLastCmdMs = millis();

  Serial.begin(BAUD);
  delay(300);
  Serial.println("MEGA_ROVER_GECOMA_READY DISARMED");
}

void loop() {
  applyTracks();
  pollSensors();

  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      handleLine();
    } else if (idx < sizeof(line) - 1) {
      line[idx++] = c;
    } else {
      idx = 0;
    }
  }
}
