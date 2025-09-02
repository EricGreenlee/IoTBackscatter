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
#include <EEPROM.h>

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


int symbol_rate_usec = 40;
// int symbol_rate_usec = 160; //this was working stably
// int symbol_rate_usec = 320;
int usec_delay = symbol_rate_usec-2;

const int n_gcs = 10;
const int gc_len = 127;
// int goldcode[] = {-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, };
// int goldcode[] = {-1, 1, -1, -1, 1, -1, 1, 1 };
int goldcode[n_gcs][gc_len] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1},
    {1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1},
    {1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1},
    {-1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1},
    {-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1},
    {-1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1},
    {-1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1},
    {1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1},
    {-1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1},
    {1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1},
    };

//raw bits
// const int bit_len = 10;
// int bits_bin[bit_len] = {1, 0, 1, 1, 1, 0, 0, 1, 0, 1};

//make array of all 1's for testing
// const int bit_len = 24;
// int bits_bin[bit_len] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
// int bits_bin[bit_len] = {1,1,1,1,0,0,0,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0,0,1};

const int bit_len = 100;
int bits_bin[bit_len] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        0,1,0,1,1,0,0,1,1,1,1,1,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                        1,1,1,1};


//make array of all 1's for testing
// const int bit_len = 8;
// int bits_bin[bit_len] = {1,1,1,1,1,1,1,1};

int bits_bip[bit_len];

//bits with CDMA encoding
const int CDMA_packet_len = bit_len*gc_len;
int CDMA_packet_bin[CDMA_packet_len];

int device_id = -1;
int freq_offset = 0;

void loadConfig() {
    Serial.println("Loading configs from EEPROM...");

    EEPROM.get(0, device_id);
    EEPROM.get(4, freq_offset);

    Serial.print("device_id: ");
    Serial.println(device_id);
    Serial.print("freq_offset: ");
    Serial.println(freq_offset);
}


void setup() {
    pinMode(ledPin, OUTPUT);
    Serial.begin(115200);
  
    pinMode(25, OUTPUT);

    delay(500);
    
    EEPROM.begin(512);
    loadConfig();

    //Serial.println("Enabling Cosine, register vals:");
    dac_cosine_enable(DAC_CHANNEL_1);
    dac_output_enable(DAC_CHANNEL_1);

//    dac_frequency_set(clk_8m_div, 1000);    //1000 ~=132kHz when clk_8m_div is set correctly
    dac_frequency_set(clk_8m_div, freq_offset);    //1000 ~=132kHz when clk_8m_div is set correctly

    Serial.println("Update test 0");

    Serial.println("next stage registers CTRL1 and CTRL2:");
    Serial.println(READ_PERI_REG(SENS_SAR_DAC_CTRL1_REG),HEX);
    Serial.println(READ_PERI_REG(SENS_SAR_DAC_CTRL2_REG),HEX);


        //convert binary to bipolar
        for (int i = 0; i < bit_len; i++){
            if (bits_bin[i] == 0){
                bits_bip[i] = -1;
            }
            else{
                bits_bip[i] = 1;
            }
        }

    //create packet
    for (int i = 0; i < bit_len; i++){
        for (int j = 0; j < gc_len; j++){
            CDMA_packet_bin[i*gc_len+j] = bits_bip[i]*goldcode[device_id][j];
        }
    }

    // Serial.println("CDMA packet:");
    // for (int i = 0; i < CDMA_packet_len; i++){
    //     if (CDMA_packet_bin[i] == 1){
    //         Serial.print("+");
    //     }
    //     Serial.print(CDMA_packet_bin[i]);
    //     Serial.print(",");
    //     if (i % 16 == 15){
    //         Serial.println();
    //     }
    // }
    // Serial.println();

    Serial.println("Starting to transmit...");
}

int count = 0;

void loop(){

    for (int i = 0; i < CDMA_packet_len; i++){
        if (CDMA_packet_bin[i] == 1){
            dac_invert_set(DAC_CHANNEL_1,2);
        }
        else{  
            dac_invert_set(DAC_CHANNEL_1,3);
        }
        delayMicroseconds(usec_delay);
    }

    dac_output_disable(DAC_CHANNEL_1);
    delayMicroseconds(usec_delay*gc_len*10);
    dac_output_enable(DAC_CHANNEL_1);
 
  
}
