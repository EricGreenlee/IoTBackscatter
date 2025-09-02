#include <EEPROM.h>

void saveConfig() {

    // //device 0
    // int device_id = 0;
    // int freq_offset = 0;

    // //device 1
    // int device_id = 1;
    // int freq_offset = 901;

    //device 2
    int device_id = 2;
    int freq_offset = 921;

    // //device 3
    // int device_id = 3;
    // int freq_offset = 903;

    // //device 4
    // int device_id = 4;
    // int freq_offset = 904;

    // //device 5
    // int device_id = 5;
    // int freq_offset = 905;

    // //device 6
    // int device_id = 6;
    // int freq_offset = 906;

    // //device 7
    // int device_id = 7;
    // int freq_offset = 907;

    //device 8
    // int device_id = 8;
    // int freq_offset = 908;
      
    EEPROM.put(0, device_id);  // Store at address 1
    EEPROM.put(4, freq_offset);  // Store at address 2
    EEPROM.commit();  // Needed for ESP but not for AVR boards
}

void setup() {
    EEPROM.begin(512); // Required for ESP, not needed for AVR
    saveConfig();
}

void loop() {}