/*
 TempSensorTag_Final.ino
 Eric Greenlee, Jason Cox
 2024-08-19


 Hardware: SparkFun ESP32 Thing Plus C
 This program reads a temperature value over the I2C interface, add two parity bits, and then transmits a BPSK packet with a prepended preamble.
 It is intended to feed into a backscatter tag that modulates the BPSK signal onto a higher frequency (915MHz) carrier wave.
*/

#include <Arduino.h>
#include "driver/dac.h"
#include <Wire.h>
// //#include "SHTSensor.h"
// #include <Adafruit_Sensor.h>
// // #include <Adafruit_BME280.h>
// #include <Adafruit_BMP280.h>




// Adafruit_BME280 bme; // I2C
// Adafruit_BMP280 bmp; //I2N


int switchPin = A3;
int switchVal;




//SHTSensor sht;


// the setup function runs once when you press reset or power the board
int ledPin = 13;
//int DAC_CHANNEL_1 = 25;


//from https://forum.arduino.cc/t/esp32-dac-cosine-generator/993374/4
int clk_8m_div = 7;      // RTC 8M clock divider (division is by clk_8m_div+1, i.e. 0 means 8MHz frequency)
int frequency_step = 8;  // Frequency step for CW generator
int scale = 0;           // 50% of the full scale
int offset;              // leave it default / 0 = no any offset
int invert =2;          // invert MSB to get sine waveform


//register addresses:
int SENS_SAR_DAC_CTRL1_REG = 0x3FF48898;//0x0098;
int SENS_SAR_DAC_CTRL2_REG = 0x3FF4889C;//0x009c;


//register indexes
int SENS_SW_TONE_EN =   0x00010000;
int SENS_DAC_CW_EN1_M = 0x01000000;
int SENS_DAC_CW_EN1_S = 25;
int SENS_DEC_CW_EN1_Mb = 0x1;
int SENS_DAC_CW_EN2_M = 0x02000000;
int SENS_DAC_INV1 =     0x3;
int SENS_DAC_INV1_S =   20;
int SENS_DAC_INV2 =     0x3;
int SENS_DAC_INV2_S =   22;
int SENS_SW_FSTEP =     0xFFFF;
int SENS_SW_FSTEP_S =   0;


int nloop = 0;


/*
* Enable cosine waveform generator on a DAC channel
*/
void dac_cosine_enable(dac_channel_t channel)
{
 // Enable tone generator common to both channels
 SET_PERI_REG_MASK(SENS_SAR_DAC_CTRL1_REG, SENS_SW_TONE_EN);
 switch(channel)
 {
     case DAC_CHANNEL_1:
         // Enable / connect tone tone generator on / to this channel
         SET_PERI_REG_MASK(SENS_SAR_DAC_CTRL2_REG, SENS_DAC_CW_EN1_M);
         // Invert MSB, otherwise part of waveform will have inverted
         SET_PERI_REG_BITS(SENS_SAR_DAC_CTRL2_REG, SENS_DAC_INV1, 2, SENS_DAC_INV1_S);


         //set the entire register manually for now
         //SET_PERI_REG_BITS(SENS_SAR_DAC_CTRL2_REG, 0xFFFFFFFF, 0x03A00000, 0);
         break;
     case DAC_CHANNEL_2:
         Serial.println("DAC_CHANNEL_2 not configured yet*****");
         SET_PERI_REG_MASK(SENS_SAR_DAC_CTRL2_REG, SENS_DAC_CW_EN2_M);
         SET_PERI_REG_BITS(SENS_SAR_DAC_CTRL2_REG, SENS_DAC_INV2, 2, SENS_DAC_INV2_S);
         break;
 }
}




void dac_cosine_disable(dac_channel_t channel)
{
 switch(channel)
 {
   case DAC_CHANNEL_1:
     // disable tone tone generator on / to this channel
     //SET_PERI_REG_MASK(SENS_SAR_DAC_CTRL2_REG, SENS_DAC_CW_EN1_M);
     SET_PERI_REG_BITS(SENS_SAR_DAC_CTRL2_REG, 0x01, 0, 21);
     //SET_PERI_REG(SENS_SAR_DAC_CTRL2_REG,0x03000000);  
     break;
   case DAC_CHANNEL_2:
     Serial.println("DAC_CHANNEL_2 not configured yet*****");
     SET_PERI_REG_MASK(SENS_SAR_DAC_CTRL2_REG, SENS_DAC_CW_EN2_M);
     break;
 }


}




/* Set frequency of internal CW generator common to both DAC channels
* clk_8m_div: 0b000 - 0b111
* frequency_step: range 0x0001 - 0xFFFF
*/
void dac_frequency_set(int clk_8m_div, int frequency_step)
{
 Serial.println("setting freq");
 //REG_SET_FIELD(RTC_CNTL_CLK_CONF_REG, RTC_CNTL_CK8M_DIV_SEL, clk_8m_div); //sets RTC clock rate division?


 SET_PERI_REG_BITS(SENS_SAR_DAC_CTRL1_REG, SENS_SW_FSTEP, frequency_step, SENS_SW_FSTEP_S);
}


/*
* Invert output pattern of a DAC channel
*
* - 00: does not invert any bits,
* - 01: inverts all bits,
* - 10: inverts MSB,
* - 11: inverts all bits except for MSB
*
*/
void dac_invert_set(dac_channel_t channel, int invert)
{
   switch(channel)
   {
       case DAC_CHANNEL_1:
           SET_PERI_REG_BITS(SENS_SAR_DAC_CTRL2_REG, SENS_DAC_INV1, invert, SENS_DAC_INV1_S);
           break;
       case DAC_CHANNEL_2:
           SET_PERI_REG_BITS(SENS_SAR_DAC_CTRL2_REG, SENS_DAC_INV2, invert, SENS_DAC_INV2_S);
           break;
   }
}


//add a two bit checksum to a 6 bit int
// int with_checksum(int raw)
// {
//   int DATA_BITS = 8;
//   int PARITY_BITS = 4;


//   const unsigned int zero = 0;
//   const unsigned int data_mask = ~zero >> (sizeof(zero) * 8 - DATA_BITS);
//   const unsigned int parity_mask = ~zero >> (sizeof(zero) * 8 - PARITY_BITS);


//   int data = raw;
//   int checksum = 0;
//   for(int i = 0; i< DATA_BITS; i += PARITY_BITS){
//     checksum = checksum ^ (data & parity_mask);
//     data = data >>PARITY_BITS;
//   }
//   return ((raw & data_mask) | (checksum << DATA_BITS));


// }


// Array to store the packed bytes
uint8_t packedData[8] = {0};  // 8 bytes (64 bits)
uint8_t dataPacket[10] = {0}; //10 bytes (80 bits)


//function to add the checksum
uint16_t add_checksum(int raw)
{
 int DATA_BITS = 8;
 int PARITY_BITS = 4;


 const unsigned int zero = 0;
 const unsigned int data_mask = ~zero >> (sizeof(zero) * 8 - DATA_BITS);
 const unsigned int parity_mask = ~zero >> (sizeof(zero) * 8 - PARITY_BITS);


 uint8_t data = raw;
 uint16_t checksum = 0;
 for(int i = 0; i< DATA_BITS; i += PARITY_BITS){
   checksum = checksum ^ (data & parity_mask);
   data = data >>PARITY_BITS;
 }
 // return ((raw & data_mask) | (checksum << DATA_BITS));
 return ((raw << PARITY_BITS) | checksum );
}


// Function to pack the 12-bit array into 8 bytes
void packArray(uint16_t input) {
 uint8_t bitPos = 0;  // Current bit position in packedData array
 
 uint16_t value = input & 0x0FFF; //mask to ensure 12 bits
 Serial.println("i,j,bitPos,value & (1 << (11-j)), packedData[bitPos/8]");
 for (int i = 0; i < 5; i++) {
   // Pack the 12-bit value into the byte array
   for (int j = 0; j < 12; j++) {
     // if (value & (1 << j)) {
     //   packedData[bitPos / 8] |= (1 << (bitPos % 8));  // Set the bit
     // }
     if (value & (1 << (11-j))) {
       packedData[bitPos/8] |= (1 << (7-(bitPos % 8)));
     }
     // Serial.print(i);
     // Serial.print("\t");
     // Serial.print(j);
     // Serial.print("\t");
     // Serial.print(bitPos);
     // Serial.print("\t");
     // Serial.print(value & (1 << (11-j)));
     // Serial.print("\t");
     // Serial.println(packedData[bitPos/8]);
   bitPos++;  // Move to the next bit
   }
 }


}






int lcount = 0;


//const int preamble1 = 0b10101010; //0xAA
//const int preamble2 = 0b10101111; //0xAF
const int preamble1 = 0b10101010; //0xAA
const int preamble2 = 0b00001111; //0xAF
const int pre_bytes = 2;


//long int header = 0b00000000000011000000000000001100; //0x000C 000C
//int header_bytes = 4;


// int payload[] = {0x48656C6C, 0x6F20576F, 0x726C6421};//"Hello World!";
// const int payload_bytes = 1;
// const int num_payload_copies = 5;


// const int pack_bytes = pre_bytes+payload_bytes*num_payload_copies;
// const int lpack = pack_bytes*8;


const int pack_bytes = 11;
const int lpack = 76;


// long int packet[pack_bytes];


long int packet[] = {0b10101010, 0b00001111, 0b00010110, 0b01110001, 0b01100111, 0b00010110, 0b01110001, 0b01100111, 0b00010110, 0b01110001, 0b01100111};


int symbol_rate_usec = 40;
// int symbol_rate_usec = 160; //this was working stably
// int symbol_rate_usec = 320;
int usec_delay = symbol_rate_usec-2;


void addPreamble(void){
 dataPacket[0] = preamble1;
 dataPacket[1] = preamble2;
 for (int i = 0; i<8; i++){
   dataPacket[i+2] = packedData[i];


 }


}




void setup() {
   pinMode(ledPin, OUTPUT);
   Serial.begin(115200);
  
   pinMode(25, OUTPUT);


   packet[0] = preamble1;
   packet[1] = preamble2;


   delay(500);


   Serial.println("Initial registers CTRL1 and CTRL2:");
   Serial.println(READ_PERI_REG(SENS_SAR_DAC_CTRL1_REG),HEX);
   Serial.println(READ_PERI_REG(SENS_SAR_DAC_CTRL2_REG),HEX);
   digitalWrite(ledPin, HIGH);


  //Serial.println("Enabling Cosine, register vals:");
   dac_cosine_enable(DAC_CHANNEL_1);
   dac_output_enable(DAC_CHANNEL_1);


   dac_frequency_set(clk_8m_div, 1000);    //1000 ~=132kHz when clk_8m_div is set correctly


   Serial.println("next stage registers CTRL1 and CTRL2:");
   Serial.println(READ_PERI_REG(SENS_SAR_DAC_CTRL1_REG),HEX);
   Serial.println(READ_PERI_REG(SENS_SAR_DAC_CTRL2_REG),HEX);


 //bool status;
 //status = bme.begin(0x76);
 // const unsigned int zero = 0;
 // Serial.println("Unsigned int bytes: ");
 // Serial.println(sizeof(zero));


 // int16_t H, T, L, P, Pcheck;


//  if (!bmp.begin(0x76)) {  // 0x76 is the default I2C address of BME280
//    Serial.println("Could not find a valid BMP280 sensor, check wiring!");
//    // while (1);  // Stay in a loop if sensor initialization fails
//  }


//  Serial.println("BMP280 found");


 // P=22;
 // Pcheck = with_checksum(P);
 // Serial.println("Checksum output:");
 // Serial.println(Pcheck);
 // Serial.println(Pcheck, BIN);
 // Serial.println(Pcheck, HEX);


 Serial.println("Starting to read temperatures...");
}


int npac;
int nind;
bool TXbit;




int t_elapsed;


unsigned long t_start = micros();
unsigned long t_end;


int i = 0;


int packet_count = 0;


// int8_t H, T, L, P, Pcheck;
int16_t H, T, L, P, Pcheck;
float lightLevel;


void loop(){
  // read temperature
//  int T = bmp.readTemperature();
 // T = packet_count/500;
 T = 23; //Values that work: 22, 23,
 //T Values that don't work: 3
//  T = 20;
 Serial.print("recorded temp: ");
 // int T = bme.readTemperature();
 Serial.println(T);
 // T = 22;


 uint8_t T_byte = (uint8_t) T;
 Serial.println(T_byte);
 Serial.println(T_byte, BIN);


 // Encode the data
 uint16_t T_encoded = add_checksum(T_byte);
 Serial.println(T_encoded, BIN);
  // Pack the array
 packArray(T_encoded);
 // for (int i = 0; i < 8; i++) {
 //   Serial.print("Byte ");
 //   Serial.print(i);
 //   Serial.print(": 0b");
 //   Serial.println(packedData[i], BIN);
 // }


 addPreamble();
 for (int i = 0; i < 10; i++) {
   Serial.print("Byte ");
   Serial.print(i);
   Serial.print(": 0b");
   Serial.println(dataPacket[i], BIN);
 }




 //P = bme.readTemperature()*9/5;
 // P=22;
 //Serial.print("\n T: ");
 //Serial.println(P);
 //Serial.println(P,BIN);


 //check if value is out of range
 // if (P>63){
 //   P = 63;
 //   Serial.println("packet data too high, setting to max (63) ...");
 // }
 // if (P<0){
 //   P = 0;
 //   Serial.println("Packet data too low, setting to min (0)...");
 // }




 //add checksum
 // Pcheck = with_checksum(P);


 // Serial.println(pcheck);


 // for(i=0; i<num_payload_copies; i++){
 //   packet[i+2] = Pcheck;
 // }


 // packet[] = {0b10101010, 0b00001111, 0b00010110, 0b01110001, 0b01100111, 0b00010110, 0b01110001, 0b01100111, 0b00010110, 0b01110001, 0b01100111};


 t_start = micros();
 //Serial.println("TX bits:");


 for(lcount=0;lcount<lpack;lcount++){


   npac = lcount / 8; //this is wrong
   nind = lcount % 8;
   // TXbit = (packet[npac] & (1 << (7-nind))) != 0;
   TXbit = (dataPacket[npac] & (1 << (7-nind))) != 0;
 // for()


   // Serial.print(TXbit);
    if(TXbit)
   {
     dac_invert_set(DAC_CHANNEL_1,2); //command takes ~1.5 useconds
     //Serial.println("not inverting");
   }
   else
   {
     dac_invert_set(DAC_CHANNEL_1,3);
     //Serial.println("inverting");
   }
   delayMicroseconds(usec_delay);
  


 }
 // delayMicroseconds(500000);
 packet_count++;


 /*
 //debug values:
 t_end = micros();
 Serial.print("delta_t = ");
 t_elapsed = t_end-t_start;
 Serial.println(t_elapsed);
 Serial.print("Desired Symbol rate (us) = ");
 Serial.println(symbol_rate_usec);
 Serial.print("Calculated Symbol rate (us) = ");
 Serial.println(t_elapsed/lpack);
  t_start = t_end;
 */
 /*
 //Serial.println("Reg values:");
 dac_cosine_disable(DAC_CHANNEL_1);
 //dac_output_enable(DAC_CHANNEL_1);
 //Serial.println(READ_PERI_REG(SENS_SAR_DAC_CTRL2_REG),HEX);
 delay(5);
 dac_cosine_enable(DAC_CHANNEL_1);
 //dac_output_enable(DAC_CHANNEL_1);
 //Serial.println(READ_PERI_REG(SENS_SAR_DAC_CTRL2_REG),HEX);
 */
}

// void setup() {
//   pinMode(LED_BUILTIN, OUTPUT);  // Set the LED_BUILTIN pin as output
// }

// void loop() {
//   digitalWrite(LED_BUILTIN, HIGH);  // Turn the LED on
//   delay(1000);                      // Wait for 1 second
//   digitalWrite(LED_BUILTIN, LOW);   // Turn the LED off
//   delay(1000);                      // Wait for 1 second
// }