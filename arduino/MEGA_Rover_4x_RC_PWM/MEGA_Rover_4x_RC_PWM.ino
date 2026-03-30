/*
  MEGA Rover 4x RC PWM (1000..2000 us)
  ------------------------------------
  Serial command format:
    M FL=1000..2000 FR=1000..2000 RL=1000..2000 RR=1000..2000\n

  Outputs (RC/servo PWM, ~50 Hz):
    D5  = Front Left
    D6  = Front Right
    D7  = Rear Left
    D8  = Rear Right

  Notes:
  - Works ONLY if your motor controllers accept RC servo PWM input.
  - Arduino does NOT drive motors directly, only controller inputs.
  - Failsafe: returns all channels to neutral if command timeout expires.
*/

#include <Arduino.h>

static const uint32_t BAUD = 115200;

static const uint8_t PIN_FL = 5;
static const uint8_t PIN_FR = 6;
static const uint8_t PIN_RL = 7;
static const uint8_t PIN_RR = 8;

static const int PWM_MIN = 1000;
static const int PWM_NEU = 1500;
static const int PWM_MAX = 2000;
static const uint32_t FAILSAFE_TIMEOUT_MS = 500;

static volatile int gFL = PWM_NEU;
static volatile int gFR = PWM_NEU;
static volatile int gRL = PWM_NEU;
static volatile int gRR = PWM_NEU;
static volatile uint32_t gLastCmdMs = 0;

static inline int clamp_us(int v) {
  if (v < PWM_MIN) return PWM_MIN;
  if (v > PWM_MAX) return PWM_MAX;
  return v;
}

// ---------- parser ----------
static char line[128];
static uint8_t idx = 0;

static bool is_digit(char c) {
  return c >= '0' && c <= '9';
}

static bool parse_int_after(const char* s, int start, int& out) {
  int i = start;
  while (s[i] && !is_digit(s[i]) && s[i] != '-') i++;
  if (!s[i]) return false;

  bool neg = false;
  if (s[i] == '-') {
    neg = true;
    i++;
  }
  if (!is_digit(s[i])) return false;

  long v = 0;
  while (s[i] && is_digit(s[i])) {
    v = v * 10 + (s[i] - '0');
    i++;
  }
  out = neg ? (int)-v : (int)v;
  return true;
}

static bool parse_cmd(const char* s, int& FL, int& FR, int& RL, int& RR) {
  bool hasM = false;
  bool hasFL = false, hasFR = false, hasRL = false, hasRR = false;
  int fl = PWM_NEU, fr = PWM_NEU, rl = PWM_NEU, rr = PWM_NEU;

  for (int i = 0; s[i]; i++) {
    if (s[i] == 'M' || s[i] == 'm') {
      hasM = true;
      break;
    }
  }
  if (!hasM) return false;

  for (int i = 0; s[i]; i++) {
    if ((s[i] == 'F' || s[i] == 'f') && (s[i + 1] == 'L' || s[i + 1] == 'l') && !hasFL) {
      int v;
      if (parse_int_after(s, i + 2, v)) {
        fl = v;
        hasFL = true;
      }
    }
    if ((s[i] == 'F' || s[i] == 'f') && (s[i + 1] == 'R' || s[i + 1] == 'r') && !hasFR) {
      int v;
      if (parse_int_after(s, i + 2, v)) {
        fr = v;
        hasFR = true;
      }
    }
    if ((s[i] == 'R' || s[i] == 'r') && (s[i + 1] == 'L' || s[i + 1] == 'l') && !hasRL) {
      int v;
      if (parse_int_after(s, i + 2, v)) {
        rl = v;
        hasRL = true;
      }
    }
    if ((s[i] == 'R' || s[i] == 'r') && (s[i + 1] == 'R' || s[i + 1] == 'r') && !hasRR) {
      int v;
      if (parse_int_after(s, i + 2, v)) {
        rr = v;
        hasRR = true;
      }
    }
  }

  if (!(hasFL && hasFR && hasRL && hasRR)) return false;

  FL = clamp_us(fl);
  FR = clamp_us(fr);
  RL = clamp_us(rl);
  RR = clamp_us(rr);
  return true;
}

static void pulse4(int FL, int FR, int RL, int RR) {
  digitalWrite(PIN_FL, HIGH);
  digitalWrite(PIN_FR, HIGH);
  digitalWrite(PIN_RL, HIGH);
  digitalWrite(PIN_RR, HIGH);

  unsigned long t0 = micros();

  bool fl_on = true, fr_on = true, rl_on = true, rr_on = true;

  while (fl_on || fr_on || rl_on || rr_on) {
    unsigned long dt = micros() - t0;

    if (fl_on && dt >= (unsigned long)FL) { digitalWrite(PIN_FL, LOW); fl_on = false; }
    if (fr_on && dt >= (unsigned long)FR) { digitalWrite(PIN_FR, LOW); fr_on = false; }
    if (rl_on && dt >= (unsigned long)RL) { digitalWrite(PIN_RL, LOW); rl_on = false; }
    if (rr_on && dt >= (unsigned long)RR) { digitalWrite(PIN_RR, LOW); rr_on = false; }
  }

  while ((micros() - t0) < 20000UL) {
    // idle to complete 20 ms frame
  }
}

void setup() {
  pinMode(PIN_FL, OUTPUT);
  pinMode(PIN_FR, OUTPUT);
  pinMode(PIN_RL, OUTPUT);
  pinMode(PIN_RR, OUTPUT);

  digitalWrite(PIN_FL, LOW);
  digitalWrite(PIN_FR, LOW);
  digitalWrite(PIN_RL, LOW);
  digitalWrite(PIN_RR, LOW);

  Serial.begin(BAUD);
  gLastCmdMs = millis();
  Serial.println("MEGA_ROVER_4CH_RC_PWM_READY");
}

void loop() {
  uint32_t now = millis();
  if ((uint32_t)(now - gLastCmdMs) > FAILSAFE_TIMEOUT_MS) {
    gFL = PWM_NEU;
    gFR = PWM_NEU;
    gRL = PWM_NEU;
    gRR = PWM_NEU;
  }

  int FL = gFL, FR = gFR, RL = gRL, RR = gRR;
  pulse4(FL, FR, RL, RR);

  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;

    if (c == '\n') {
      line[idx] = 0;
      idx = 0;

      int nfl, nfr, nrl, nrr;
      if (parse_cmd(line, nfl, nfr, nrl, nrr)) {
        gFL = nfl;
        gFR = nfr;
        gRL = nrl;
        gRR = nrr;
        gLastCmdMs = millis();
        Serial.println("OK");
      }
    } else {
      if (idx < sizeof(line) - 1) line[idx++] = c;
      else idx = 0;
    }
  }
}
