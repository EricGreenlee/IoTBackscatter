#include <EEPROM.h>

void saveConfig() {

    //device 7
    int device_id = 8;
    int freq_offset = 908;
    int device_id_2 = 8;

    Serial.println("Saving config...");
      
    EEPROM.put(0, device_id);  // Store at address 1
    // EEPROM.commit();

    Serial.println("Saved device_id");
    
    EEPROM.put(4, freq_offset);  // Store at address 2
    EEPROM.commit();  // Needed for ESP but not for AVR boards

    Serial.println("Saved freq_offset");
}

#include <EEPROM.h>

void loadConfig() {
    int test;
    int device_id;
    int freq_offset;
    // EEPROM.get(0, test);
    EEPROM.get(0, device_id);
    EEPROM.get(4, freq_offset);
    Serial.print("test: ");
    Serial.println(test);
    Serial.print("device_id: ");
    Serial.println(device_id);
    Serial.print("freq_offset: ");
    Serial.println(freq_offset);
}

void setup() {
    Serial.begin(115200);
    EEPROM.begin(512);
    saveConfig();
    delay(1000);
    loadConfig();
}

void loop() {}
