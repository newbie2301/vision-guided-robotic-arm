/*
 * arm_controller.ino
 * Vision-Guided Robotic Arm – Arduino Firmware
 *
 * Listens for newline-terminated comma-separated joint angle commands
 * from the host PC over USB serial (115200 baud).
 * Format: "J1,J2,J3,J4\n"   e.g. "90,125,180,60\n"
 *
 * Executes smooth interpolated motion via smoothMove() to prevent
 * mechanical shock and servo stall.
 *
 * Pin assignments:
 *   J1 – Base Yaw      → Pin 3  (PWM)
 *   J2 – Shoulder Pitch → Pin 5  (PWM)
 *   J3 – Elbow Pitch    → Pin 6  (PWM)
 *   J4 – Gripper        → Pin 9  (PWM)
 *
 * Joint limits:
 *   J1:  0  – 180 deg
 *   J2:  10 – 170 deg
 *   J3:  40 – 180 deg
 *   J4:  42 – 100 deg   (100 = open, 42 = closed)
 */

#include <Servo.h>

// ── Pin assignments ──────────────────────────────────────────────────────────
const int PIN_J1 = 3;
const int PIN_J2 = 5;
const int PIN_J3 = 6;
const int PIN_J4 = 9;

// ── Joint limits ─────────────────────────────────────────────────────────────
const int J1_MIN = 0,   J1_MAX = 180;
const int J2_MIN = 10,  J2_MAX = 170;
const int J3_MIN = 40,  J3_MAX = 180;
const int J4_MIN = 42,  J4_MAX = 100;

// ── Home / initial pose ───────────────────────────────────────────────────────
const int HOME_J1 = 90;
const int HOME_J2 = 125;
const int HOME_J3 = 180;
const int HOME_J4 = 60;

// ── Motion parameters ─────────────────────────────────────────────────────────
const int   SMOOTH_STEPS    = 80;    // interpolation steps per command
const int   SMOOTH_DELAY_MS = 20;    // milliseconds per step  → 1.6 s total

// ── Servo objects ─────────────────────────────────────────────────────────────
Servo servo_j1, servo_j2, servo_j3, servo_j4;

// ── Current angles (tracked for interpolation) ────────────────────────────────
int cur_j1, cur_j2, cur_j3, cur_j4;

// ─────────────────────────────────────────────────────────────────────────────
// Clamp helper
// ─────────────────────────────────────────────────────────────────────────────
int clamp(int val, int lo, int hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

// ─────────────────────────────────────────────────────────────────────────────
// Smooth interpolated move for all four joints simultaneously
// ─────────────────────────────────────────────────────────────────────────────
void smoothMove(int t1, int t2, int t3, int t4) {
    t1 = clamp(t1, J1_MIN, J1_MAX);
    t2 = clamp(t2, J2_MIN, J2_MAX);
    t3 = clamp(t3, J3_MIN, J3_MAX);
    t4 = clamp(t4, J4_MIN, J4_MAX);

    for (int step = 1; step <= SMOOTH_STEPS; step++) {
        float ratio = (float)step / (float)SMOOTH_STEPS;
        servo_j1.write(cur_j1 + (int)((t1 - cur_j1) * ratio));
        servo_j2.write(cur_j2 + (int)((t2 - cur_j2) * ratio));
        servo_j3.write(cur_j3 + (int)((t3 - cur_j3) * ratio));
        servo_j4.write(cur_j4 + (int)((t4 - cur_j4) * ratio));
        delay(SMOOTH_DELAY_MS);
    }
    cur_j1 = t1;
    cur_j2 = t2;
    cur_j3 = t3;
    cur_j4 = t4;
}

// ─────────────────────────────────────────────────────────────────────────────
// setup
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);

    servo_j1.attach(PIN_J1);
    servo_j2.attach(PIN_J2);
    servo_j3.attach(PIN_J3);
    servo_j4.attach(PIN_J4);

    // Initialise tracked positions to home
    cur_j1 = HOME_J1;
    cur_j2 = HOME_J2;
    cur_j3 = HOME_J3;
    cur_j4 = HOME_J4;

    // Move to home on power-up
    servo_j1.write(HOME_J1);
    servo_j2.write(HOME_J2);
    servo_j3.write(HOME_J3);
    servo_j4.write(HOME_J4);
    delay(1000);

    Serial.println("ARM_READY");
}

// ─────────────────────────────────────────────────────────────────────────────
// loop – parse incoming commands and execute
// ─────────────────────────────────────────────────────────────────────────────
void loop() {
    if (Serial.available()) {
        String line = Serial.readStringUntil('\n');
        line.trim();

        if (line.length() == 0) return;

        // Parse four comma-separated integers
        int angles[4];
        int count = 0;
        int start = 0;

        for (int i = 0; i <= (int)line.length() && count < 4; i++) {
            if (i == (int)line.length() || line[i] == ',') {
                String token = line.substring(start, i);
                token.trim();
                angles[count++] = token.toInt();
                start = i + 1;
            }
        }

        if (count < 4) {
            Serial.println("ERR_PARSE");
            return;
        }

        smoothMove(angles[0], angles[1], angles[2], angles[3]);
        Serial.println("DONE");
    }
}
