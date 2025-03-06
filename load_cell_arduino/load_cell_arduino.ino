/*
Adapted from: 

 Example using the SparkFun HX711 breakout board with a scale
 By: Nathan Seidle
 SparkFun Electronics
 Date: November 19th, 2014
 License: This code is public domain but you buy me a beer if you use this and we meet someday (Beerware license).

 This example demonstrates basic scale output. See the calibration sketch to get the calibration_factor for your
 specific load cell setup.

 This example code uses bogde's excellent library: https://github.com/bogde/HX711
 bogde's library is released under a GNU GENERAL PUBLIC LICENSE

 The HX711 does one thing well: read load cells. The breakout board is compatible with any wheat-stone bridge
 based load cell which should allow a user to measure everything from a few grams to tens of tons.
 Arduino pin 2 -> HX711 CLK
 3 -> DAT
 5V -> VCC
 GND -> GND

 The HX711 board can be powered from 2.7V to 5V so the Arduino 5V power should be fine.

*/

#include "HX711.h"

#define calibration_factor -21050.0 //This value is obtained using the SparkFun_HX711_Calibration sketch
// above is calibrated to kgs

#define DOUT  3
#define CLK  2
#define BUTT 4

HX711 scale;

unsigned long lastDebounceTime = 0;  // the last time the output pin was toggled
unsigned long debounceDelay = 50;

int buttonState;            // the current reading from the input pin
int lastButtonState = LOW;  // the previous reading from the input pin
int tare = 0;

void setup() {
    pinMode(BUTT, INPUT);

    Serial.begin(9600);
    scale.begin(DOUT, CLK);
    scale.set_scale(calibration_factor); //This value is obtained by using the SparkFun_HX711_Calibration sketch
    scale.tare(); //Assuming there is no weight on the scale at start up, reset the scale to 0

    Serial.println("Load cells ready");
}

void loop() {
    tare = 0;
    int reading = digitalRead(BUTT);
    // If the switch changed, due to noise or pressing:
    if (reading != lastButtonState) {
      // reset the debouncing timer
      lastDebounceTime = millis();
    }

    if ((millis() - lastDebounceTime) > debounceDelay) {

      if (reading != buttonState) {
        buttonState = reading;

        if (buttonState == HIGH) {
          tare = 1;
        }
      }
    }

    if (tare) {
        delay(500);
        Serial.print("Taring");
        scale.tare();
        Serial.println();
    }
    
    Serial.print(scale.get_units(), 3); //scale.get_units() returns a float
    Serial.print(" kgs"); //You can change this to kg but you'll need to refactor the calibration_factor
    Serial.println();

    lastButtonState = reading;
}
