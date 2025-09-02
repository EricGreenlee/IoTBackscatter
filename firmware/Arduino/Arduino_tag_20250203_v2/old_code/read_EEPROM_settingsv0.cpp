#include <EEPROM.h>

void loadConfig() {
    int device_id;
    int freq_offset;

    EEPROM.get(0, device_id);
    EEPROM.get(4, freq_offset);

    Serial.print("device_id: ");
    Serial.println(device_id);
    Serial.print("freq_offset: ");
    Serial.println(freq_offset);
}

void setup() {
    Serial.begin(115200);
    EEPROM.begin(512);
    loadConfig();
}

void loop() {}