#include <FastLED.h>
#include <string.h>
#include <stdio.h>

#define SERIAL_RX_BUFFER_SIZE 2048
#define SERIAL_TX_BUFFER_SIZE 2048

// Define NUM of LEDs
#define NUM_LEDS_0 12
#define NUM_LEDS_1 12
#define NUM_LEDS_2 12

// Define PIN for LEDs
#define LED_0_PIN 6
#define LED_1_PIN 10
#define LED_2_PIN 9

// Define LED arrays
CRGB leds_0[NUM_LEDS_0];
CRGB leds_1[NUM_LEDS_1];
CRGB leds_2[NUM_LEDS_1];

// Define parameters
String com_str = "";
const char s[4] = ",";
char *token;
int count = 0;
int led_info[2] = {0};
int l_1 = 4;
int l_2 = 2;

void setup()
{
    // Add LEDs
    FastLED.addLeds<WS2812, LED_0_PIN, GRB>(leds_0, NUM_LEDS_0);
    FastLED.addLeds<WS2812, LED_1_PIN, GRB>(leds_1, NUM_LEDS_1);
    FastLED.addLeds<WS2812, LED_2_PIN, GRB>(leds_2, NUM_LEDS_2);

    // Init LED arrays
    for (int i = 0; i < NUM_LEDS_0; i++)
    {
        leds_0[i] = CRGB(l_1, l_1, l_1);
    }
    for (int i = 0; i < NUM_LEDS_1; i++)
    {
        leds_1[i] = CRGB(0, l_2, l_2);
    }

    for (int i = 0; i < NUM_LEDS_2; i++)
    {
        leds_2[i] = CRGB(0, l_2, l_2);
    }

    // Start serial
    Serial.begin(115200);
    Serial.setTimeout(1);
}

void loop()
{
    // Read string from serial
    while (Serial.available() > 0)
    {
        com_str += char(Serial.read());
        delay(2);
    }

    if (com_str.length() != 0)
    {
        // Print string to serial
        Serial.println("str:" + com_str);

        // Split string
        char *str_c = (char *)com_str.c_str();
        token = strtok(str_c, s);
        count = 0;
        while (token != 0)
        {
            led_info[count] = atoi(token);
            count += 1;
            token = strtok(0, s);
        }
        com_str = "";

        // Set LEDs by splited data
        int led_data[12] = {0};
        led_data[led_info[0]] = led_info[1];
        if (led_info[0] == 0)
        {
            led_data[11] = led_info[1] / 2;
            led_data[1] = led_info[1] / 2;
        }
        else if (led_info[0] == 11)
        {
            led_data[10] = led_info[1] / 2;
            led_data[0] = led_info[1] / 2;
        }
        else
        {
            led_data[led_info[0] - 1] = led_info[1] / 2;
            led_data[led_info[0] + 1] = led_info[1] / 2;
        }
        for (int i = 0; i < NUM_LEDS_1; i++)
        {
            int x = led_data[i];
            if (x > l_2)
            {
                leds_1[i] = CRGB(0, x, x);
                leds_2[i] = CRGB(0, x, x);
            }
            else
            {
                leds_1[i] = CRGB(0, l_2, l_2);
                leds_2[i] = CRGB(0, l_2, l_2);
            }
        }
    }
}