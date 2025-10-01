/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2025 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "stm32l0xx_hal_i2c.h"
#include "stm32l0xx_hal_adc.h"
#include "precomputed_samples_GC0.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
// === Streaming parameters ===
// #define SAMPLE_RATE 200000 // 200 ksps DAC update rate
#define SAMPLE_RATE 400000 // 400 ksps DAC update rate
#define CARRIER_SAMPLES 4  // 4 samples per 50 kHz cycle
#define GOLD_LEN 127
#define SYMBOL_SAMPLES (GOLD_LEN * 16) // 2032 samples per encoded bit
#define BUF_SAMPLES 2048               // DMA circular buffer (2 halves)
#define HALF_SAMPLES (BUF_SAMPLES / 2) // 1024 per half
// #define IDLE_BITS 80
#define IDLE_BITS 80

const int preamble[64] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0};
// const int preamble[64] = {
//     1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
//     1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
//     1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
//     1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
// const int preamble[64] = {
//     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
const int payload[16] = {0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1};
// const int payload[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

// int goldcode[GOLD_LEN] = {
//     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1,
//     -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1,
//     -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1,
//     1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1,
//     1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1,
//     1, 1, -1, -1, 1, -1, 1, 1, -1};
// uint16_t gold_pos[SYMBOL_SAMPLES];
// uint16_t gold_neg[SYMBOL_SAMPLES];

static int packet_bits[80];

static uint16_t dma_buf[BUF_SAMPLES]; // 4 KB

// Streaming state
static int cur_bit = 0;                  // 0..79
static int sym_offset = 0;               // 0..2032
static int in_idle = 0;                  // 0=packet streaming, 1=idle (2048)
static int idle_sent = 0;                // how many idle "bits" have been sent
static int idle_remain = SYMBOL_SAMPLES; // samples remaining in the current idle "bit"

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
DAC_HandleTypeDef hdac;
TIM_HandleTypeDef htim6;

UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
// Sine wave data buffer
// uint16_t sine_wave[SAMPLES];
DMA_HandleTypeDef hdma_dac_ch1; // Moved here to protect from CubeMX regeneration

// I2C_HandleTypeDef hi2c1;
// ADC_HandleTypeDef hadc;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
// void SystemClock_Config(void);
void MX_GPIO_Init(void);
void MX_USART2_UART_Init(void);
void MX_DAC_Init(void);
void MX_TIM6_Init(void);
/* USER CODE BEGIN PFP */
void MX_DMA_Init(void);
// void Generate_Sine_Table(void);

// void MX_TIM6_Init(void);
// void DMA1_Channel2_IRQHandler(void);
void HAL_DAC_ConvHalfCpltCallbackCh1(DAC_HandleTypeDef *hdac);
void HAL_DAC_ConvCpltCallbackCh1(DAC_HandleTypeDef *hdac);
// void GenerateGoldTables(void);
static void BuildPacket(void);
static void FillHalf(int half_index);
static void restart_packet_stream(void);
static void enter_idle(void);

// void MX_I2C1_Init(void); // === NEW
// void MX_ADC_Init(void);  // === NEW

// // === NEW: helpers
// int I2C_ReadRegs(uint16_t dev7b, uint8_t reg, uint8_t *data, uint16_t len);
// int I2C_WriteRegs(uint16_t dev7b, uint8_t reg, const uint8_t *data, uint16_t len);
// uint16_t ADC_ReadRaw(uint32_t channel);
// uint32_t Read_Vref_mV(void);
// uint32_t ADC_Channel_mV(uint32_t channel, uint32_t vref_mV);
// float Divider_InputVoltage_V(float vout_mV, float Rhigh_ohm, float Rlow_ohm);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();  // Add this line - DMA must be initialized before DAC
  MX_TIM6_Init(); // Initialize Timer 6
  MX_USART2_UART_Init();
  MX_DAC_Init();
  /* USER CODE BEGIN 2 */
  // MX_I2C1_Init();
  // MX_ADC_Init();

  // GenerateGoldTables();
  BuildPacket();
  restart_packet_stream(); // <— add this line

  FillHalf(0);
  FillHalf(1);

  HAL_DAC_Start_DMA(&hdac, DAC_CHANNEL_1, (uint32_t *)dma_buf, BUF_SAMPLES, DAC_ALIGN_12B_R);
  // HAL_TIM_Base_Start(&htim6);

  // // Generate sine wave lookup table
  // Generate_Sine_Table();

  // // Debug: Print first 10 samples
  // char debug_msg[100];
  // HAL_UART_Transmit(&huart2, (uint8_t *)"Sine wave samples (1kHz test):\r\n", 32, HAL_MAX_DELAY);
  // for (int i = 0; i < 10; i++)
  // {
  //   sprintf(debug_msg, "Sample %d: %d (%.2fV)\r\n", i, sine_wave[i], (float)sine_wave[i] * 3.3f / 4095.0f);
  //   HAL_UART_Transmit(&huart2, (uint8_t *)debug_msg, strlen(debug_msg), HAL_MAX_DELAY);
  // }

  // // Send startup message
  // char msg[] = "1kHz Sinusoid Test Started!\r\n";
  // HAL_UART_Transmit(&huart2, (uint8_t *)msg, sizeof(msg) - 1, HAL_MAX_DELAY);

  // // Start DAC with DMA in circular mode
  // if (HAL_DAC_Start_DMA(&hdac, DAC_CHANNEL_1, (uint32_t *)sine_wave, SAMPLES, DAC_ALIGN_12B_R) != HAL_OK)
  // {
  //   char error_msg[] = "ERROR: DAC DMA start failed!\r\n";
  //   HAL_UART_Transmit(&huart2, (uint8_t *)error_msg, sizeof(error_msg) - 1, HAL_MAX_DELAY);
  // }
  // else
  // {
  //   char dac_msg[] = "DAC DMA started successfully\r\n";
  //   HAL_UART_Transmit(&huart2, (uint8_t *)dac_msg, sizeof(dac_msg) - 1, HAL_MAX_DELAY);
  // }

  // Start Timer 6 to trigger DAC
  if (HAL_TIM_Base_Start(&htim6) != HAL_OK)
  {
    char timer_error[] = "ERROR: Timer 6 start failed!\r\n";
    HAL_UART_Transmit(&huart2, (uint8_t *)timer_error, sizeof(timer_error) - 1, HAL_MAX_DELAY);
  }
  else
  {
    char timer_msg[] = "Timer 6 started - 1kHz sine wave on PA4- test\r\n";
    HAL_UART_Transmit(&huart2, (uint8_t *)timer_msg, sizeof(timer_msg) - 1, HAL_MAX_DELAY);
  }

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);

    // // Test: Manual DAC update to verify DAC is working
    // static int manual_test = 0;
    // if (manual_test < 10)
    // {
    //   // Try setting DAC manually to different values
    //   HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, manual_test < 5 ? 1000 : 3000);
    //   char test_msg[50];
    //   sprintf(test_msg, "Manual DAC test: %d\r\n", manual_test < 5 ? 1000 : 3000);
    //   HAL_UART_Transmit(&huart2, (uint8_t *)test_msg, strlen(test_msg), HAL_MAX_DELAY);
    //   manual_test++;
    // }
    // else
    // {
    //   // Status message - sine wave should be running via DMA
    //   char status_msg[] = "DMA sine wave should be running...\r\n";
    //   HAL_UART_Transmit(&huart2, (uint8_t *)status_msg, sizeof(status_msg) - 1, HAL_MAX_DELAY);
    // }

    // // === NEW: I2C example (WHO_AM_I @ 0x0F)
    // const uint8_t WHOAMI_REG = 0x0F;
    // const uint16_t DEV_ADDR_7B = 0x6A; // change to your sensor's 7-bit address
    // uint8_t who = 0x00;
    // if (I2C_ReadRegs(DEV_ADDR_7B, WHOAMI_REG, &who, 1) == 0)
    // {
    //   char msg[64];
    //   int n = snprintf(msg, sizeof(msg), "I2C 0x%02X WHO_AM_I=0x%02X\r\n", DEV_ADDR_7B, who);
    //   HAL_UART_Transmit(&huart2, (uint8_t *)msg, n, HAL_MAX_DELAY);
    // }
    // else
    // {
    //   char msg[] = "I2C read failed\r\n";
    //   HAL_UART_Transmit(&huart2, (uint8_t *)msg, sizeof(msg) - 1, HAL_MAX_DELAY);
    // }

    // // === NEW: ADC example on PA0 (ADC_CHANNEL_0)
    // uint32_t vref_mV = Read_Vref_mV();
    // uint32_t vout_mV = ADC_Channel_mV(ADC_CHANNEL_0, vref_mV);

    // // Replace with your actual divider values:
    // const float Rhigh = 100000.0f; // ohms (top resistor to Vin)
    // const float Rlow = 100000.0f;  // ohms (bottom to GND)
    // float vin_V = Divider_InputVoltage_V((float)vout_mV, Rhigh, Rlow);

    // char vmsg[96];
    // int vn = snprintf(vmsg, sizeof(vmsg),
    //                   "ADC: Vref=%lumV, Vout=%lumV on PA0 → Vin=%.3f V\r\n",
    //                   (unsigned long)vref_mV, (unsigned long)vout_mV, vin_V);
    // HAL_UART_Transmit(&huart2, (uint8_t *)vmsg, vn, HAL_MAX_DELAY);

    HAL_Delay(1000); // Slower status updates
  }
  /* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
// void SystemClock_Config(void)
// {
//   RCC_OscInitTypeDef RCC_OscInitStruct = {0};
//   RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
//   RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

//   /** Configure the main internal regulator output voltage
//    */
//   __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

//   /** Initializes the RCC Oscillators according to the specified parameters
//    * in the RCC_OscInitTypeDef structure.
//    */
//   RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_MSI;
//   RCC_OscInitStruct.MSIState = RCC_MSI_ON;
//   RCC_OscInitStruct.MSICalibrationValue = 0;
//   RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_5;
//   RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
//   if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
//   {
//     Error_Handler();
//   }

//   /** Initializes the CPU, AHB and APB buses clocks
//    */
//   RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
//   RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_MSI;
//   RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
//   RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
//   RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

//   if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
//   {
//     Error_Handler();
//   }
//   PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USART2;
//   PeriphClkInit.Usart2ClockSelection = RCC_USART2CLKSOURCE_PCLK1;
//   if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
//   {
//     Error_Handler();
//   }
// }

// custom clock config to increase the maximum stable DAC rate
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  // Enable HSI16 as system clock
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_OFF; // keep it simple (16 MHz)
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
    Error_Handler();

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
    Error_Handler();

  // Keep USART2 on PCLK1
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USART2;
  PeriphClkInit.Usart2ClockSelection = RCC_USART2CLKSOURCE_PCLK1;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
    Error_Handler();
}

/**
 * @brief DAC Initialization Function
 * @param None
 * @retval None
 */
void MX_DAC_Init(void)
{

  /* USER CODE BEGIN DAC_Init 0 */

  /* USER CODE END DAC_Init 0 */

  DAC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN DAC_Init 1 */

  /* USER CODE END DAC_Init 1 */

  /** DAC Initialization
   */
  hdac.Instance = DAC;
  if (HAL_DAC_Init(&hdac) != HAL_OK)
  {
    Error_Handler();
  }

  /** DAC channel OUT1 config
   */
  sConfig.DAC_Trigger = DAC_TRIGGER_T6_TRGO; // Changed: Use Timer 6 trigger
  sConfig.DAC_OutputBuffer = DAC_OUTPUTBUFFER_ENABLE;
  if (HAL_DAC_ConfigChannel(&hdac, &sConfig, DAC_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN DAC_Init 2 */

  /* USER CODE END DAC_Init 2 */
}

/**
 * @brief USART2 Initialization Function
 * @param None
 * @retval None
 */
void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart2.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */
}

/**
 * @brief GPIO Initialization Function
 * @param None
 * @retval None
 */
void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
void MX_TIM6_Init(void)
{
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  __HAL_RCC_TIM6_CLK_ENABLE();

  htim6.Instance = TIM6;
  htim6.Init.Prescaler = 0;
  htim6.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim6.Init.Period = (SystemCoreClock / SAMPLE_RATE) - 1; // drive DAC at SAMPLE_RATE
  htim6.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  HAL_TIM_Base_Init(&htim6);

  sMasterConfig.MasterOutputTrigger = TIM_TRGO_UPDATE;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  HAL_TIMEx_MasterConfigSynchronization(&htim6, &sMasterConfig);
}

void MX_DMA_Init(void)
{
  __HAL_RCC_DMA1_CLK_ENABLE();

  hdma_dac_ch1.Instance = DMA1_Channel2;
  hdma_dac_ch1.Init.Request = DMA_REQUEST_9; // DAC_CH1 request
  hdma_dac_ch1.Init.Direction = DMA_MEMORY_TO_PERIPH;
  hdma_dac_ch1.Init.PeriphInc = DMA_PINC_DISABLE;
  hdma_dac_ch1.Init.MemInc = DMA_MINC_ENABLE;
  hdma_dac_ch1.Init.PeriphDataAlignment = DMA_PDATAALIGN_HALFWORD;
  hdma_dac_ch1.Init.MemDataAlignment = DMA_MDATAALIGN_HALFWORD;
  hdma_dac_ch1.Init.Mode = DMA_CIRCULAR;
  hdma_dac_ch1.Init.Priority = DMA_PRIORITY_HIGH;
  HAL_DMA_Init(&hdma_dac_ch1);

  __HAL_LINKDMA(&hdac, DMA_Handle1, hdma_dac_ch1);

  // === NEW: Enable DMA interrupts in NVIC ===
  HAL_NVIC_SetPriority(DMA1_Channel2_3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel2_3_IRQn);
}

// void DMA1_Channel2_IRQHandler(void)
// {
//   HAL_DMA_IRQHandler(&hdma_dac_ch1);
// }

// Called when first half of buffer is done
void HAL_DAC_ConvHalfCpltCallbackCh1(DAC_HandleTypeDef *hdac)
{
  // HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin); // blink LED
  // const char *msg = "DMA half-transfer interrupt!\r\n";
  // HAL_UART_Transmit(&huart2, (uint8_t *)msg, strlen(msg), HAL_MAX_DELAY);
  FillHalf(0);
}

// Called when second half is done
void HAL_DAC_ConvCpltCallbackCh1(DAC_HandleTypeDef *hdac)
{
  // HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin); // blink LED
  // const char *msg = "DMA full-transfer interrupt!\r\n";
  // HAL_UART_Transmit(&huart2, (uint8_t *)msg, strlen(msg), HAL_MAX_DELAY);
  FillHalf(1);
}

static void BuildPacket(void)
{
  for (int i = 0; i < 64; i++)
    packet_bits[i] = preamble[i];
  for (int i = 0; i < 16; i++)
    packet_bits[64 + i] = payload[i];
}
// Returns pointer to the precomputed table for the current bit
static inline const uint16_t *table_for_bit(int bit_value)
{
  // 1 => normal (in-phase), 0 => inverted (anti-phase)
  return bit_value ? normal_goldcode_samples : inverted_goldcode_samples;
}

static void restart_packet_stream(void)
{
  cur_bit = 0;
  sym_offset = 0;
  in_idle = 0;
  idle_sent = 0;
  idle_remain = SYMBOL_SAMPLES;
}

static void enter_idle(void)
{
  in_idle = 1;
  idle_sent = 0;
  idle_remain = SYMBOL_SAMPLES;
}

static void FillHalf(int half_index)
{
  uint16_t *dst = &dma_buf[half_index * HALF_SAMPLES];
  int remaining_in_half = HALF_SAMPLES;

  while (remaining_in_half > 0)
  {
    if (!in_idle)
    {
      // === Streaming packet bits ===
      const uint16_t *tab = table_for_bit(packet_bits[cur_bit]);

      int remain_in_symbol = SYMBOL_SAMPLES - sym_offset;
      int to_copy = (remain_in_symbol < remaining_in_half) ? remain_in_symbol : remaining_in_half;

      memcpy(dst, &tab[sym_offset], to_copy * sizeof(uint16_t));

      dst += to_copy;
      remaining_in_half -= to_copy;
      sym_offset += to_copy;

      if (sym_offset >= SYMBOL_SAMPLES)
      {
        sym_offset = 0;
        cur_bit++;

        if (cur_bit >= 80)
        {
          // Finished the whole packet → go idle
          enter_idle();
        }
      }
    }
    else
    {
      // === Idle at mid-level (2048) for IDLE_BITS symbols ===
      int to_fill = (idle_remain < remaining_in_half) ? idle_remain : remaining_in_half;

      for (int i = 0; i < to_fill; i++)
      {
        dst[i] = 2048;
      }

      dst += to_fill;
      remaining_in_half -= to_fill;
      idle_remain -= to_fill;

      if (idle_remain == 0)
      {
        idle_sent++;
        if (idle_sent >= IDLE_BITS)
        {
          // Done with idle — restart sequence from the first bit
          restart_packet_stream();
        }
        else
        {
          // Next idle "bit"
          idle_remain = SYMBOL_SAMPLES;
        }
      }
    }
  }
}

// // === NEW: I2C1 on PB8 (SCL), PB9 (SDA), 100 kHz
// void MX_I2C1_Init(void)
// {
//   __HAL_RCC_GPIOB_CLK_ENABLE();
//   __HAL_RCC_I2C1_CLK_ENABLE();

//   GPIO_InitTypeDef GPIO_InitStruct = {0};
//   GPIO_InitStruct.Pin = GPIO_PIN_8 | GPIO_PIN_9;
//   GPIO_InitStruct.Mode = GPIO_MODE_AF_OD; // open-drain
//   GPIO_InitStruct.Pull = GPIO_PULLUP;     // ext pullups OK too
//   GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
//   GPIO_InitStruct.Alternate = GPIO_AF4_I2C1;
//   HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

//   hi2c1.Instance = I2C1;
//   hi2c1.Init.Timing = 0x00303D5B; // ~100 kHz on L0 @ 16 MHz HSI (Cube default timing)
//   hi2c1.Init.OwnAddress1 = 0;
//   hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
//   hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
//   hi2c1.Init.OwnAddress2 = 0;
//   hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
//   hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
//   hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
//   if (HAL_I2C_Init(&hi2c1) != HAL_OK)
//     Error_Handler();

//   // Enable analog filter, leave digital filter off
//   if (HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
//     Error_Handler();
//   if (HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0) != HAL_OK)
//     Error_Handler();
// }

// // === NEW: ADC1 single-ended, software trigger; example channel: PA0 (ADC_IN0)
// void MX_ADC_Init(void)
// {
//   __HAL_RCC_ADC1_CLK_ENABLE();
//   __HAL_RCC_GPIOA_CLK_ENABLE();

//   // PA0 as analog for divider sense
//   GPIO_InitTypeDef GPIO_InitStruct = {0};
//   GPIO_InitStruct.Pin = GPIO_PIN_0;
//   GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
//   GPIO_InitStruct.Pull = GPIO_NOPULL;
//   HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

//   hadc.Instance = ADC1;
//   hadc.Init.OversamplingMode = DISABLE;
//   hadc.Init.ClockPrescaler = ADC_CLOCK_ASYNC_DIV1; // L0: async clock
//   hadc.Init.Resolution = ADC_RESOLUTION_12B;
//   hadc.Init.SamplingTime = ADC_SAMPLETIME_160CYCLES_5; // stable for high source impedance
//   hadc.Init.ScanConvMode = ADC_SCAN_DIRECTION_FORWARD;
//   hadc.Init.DataAlign = ADC_DATAALIGN_RIGHT;
//   hadc.Init.ContinuousConvMode = DISABLE;
//   hadc.Init.DiscontinuousConvMode = DISABLE;
//   hadc.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
//   hadc.Init.ExternalTrigConv = ADC_SOFTWARE_START;
//   hadc.Init.DMAContinuousRequests = DISABLE;
//   hadc.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
//   hadc.Init.Overrun = ADC_OVR_DATA_PRESERVED;
//   hadc.Init.LowPowerAutoWait = DISABLE;
//   hadc.Init.LowPowerFrequencyMode = DISABLE;
//   hadc.Init.LowPowerAutoPowerOff = DISABLE;

//   if (HAL_ADC_Init(&hadc) != HAL_OK)
//     Error_Handler();

//   // Enable Vrefint channel
//   __HAL_ADC_ENABLE(&hadc);
// }

// // === NEW: I2C register helpers (7-bit addr, mem read/write)
// int I2C_ReadRegs(uint16_t dev7b, uint8_t reg, uint8_t *data, uint16_t len)
// {
//   // dev7b is 7-bit (e.g., 0x6A), HAL expects 7-bit << 1 internally
//   if (HAL_I2C_Mem_Read(&hi2c1, (dev7b << 1), reg, I2C_MEMADD_SIZE_8BIT, data, len, 100) == HAL_OK)
//     return 0;
//   return -1;
// }

// int I2C_WriteRegs(uint16_t dev7b, uint8_t reg, const uint8_t *data, uint16_t len)
// {
//   if (HAL_I2C_Mem_Write(&hi2c1, (dev7b << 1), reg, I2C_MEMADD_SIZE_8BIT, (uint8_t *)data, len, 100) == HAL_OK)
//     return 0;
//   return -1;
// }

// // === NEW: Raw ADC read from a single channel
// uint16_t ADC_ReadRaw(uint32_t channel)
// {
//   ADC_ChannelConfTypeDef sConfig = {0};
//   sConfig.Channel = channel;
//   sConfig.Rank = ADC_RANK_CHANNEL_NUMBER;
//   // sConfig.SamplingTime = ADC_SAMPLETIME_160CYCLES_5;
//   if (HAL_ADC_ConfigChannel(&hadc, &sConfig) != HAL_OK)
//     Error_Handler();

//   if (HAL_ADC_Start(&hadc) != HAL_OK)
//     Error_Handler();
//   if (HAL_ADC_PollForConversion(&hadc, 5) != HAL_OK)
//     Error_Handler();
//   uint16_t val = (uint16_t)HAL_ADC_GetValue(&hadc);
//   HAL_ADC_Stop(&hadc);
//   return val;
// }

// // === NEW: Vref compensation (uses factory calibration @ 3.0 V on L0)
// // L0 VREFINT calibration address: 0x1FF80078 (16-bit)
// #ifndef VREFINT_CAL_ADDR
// #define VREFINT_CAL_ADDR ((uint16_t *)(0x1FF80078U))
// #endif

// uint32_t Read_Vref_mV(void)
// {
//   // Measure internal Vrefint channel
//   // On L0, Vrefint channel is ADC_CHANNEL_17
//   // Factory calibration is at Vdda = 3.0V
//   const uint16_t vrefint_cal = *VREFINT_CAL_ADDR; // at 3.0 V
//   uint16_t vrefint_adc = 0;

//   // Enable Vrefint
//   __HAL_ADC_ENABLE(&hadc);
//   vrefint_adc = ADC_ReadRaw(ADC_CHANNEL_VREFINT);

//   // Vdda (mV) = 3000 * VREFINT_CAL / VREFINT_DATA
//   if (vrefint_adc == 0)
//     return 3300;
//   uint32_t vdda_mV = (3000UL * (uint32_t)vrefint_cal) / (uint32_t)vrefint_adc;
//   return vdda_mV;
// }

// // === NEW: Convert an ADC channel reading to mV with Vref compensation
// uint32_t ADC_Channel_mV(uint32_t channel, uint32_t vref_mV)
// {
//   uint16_t raw = ADC_ReadRaw(channel);
//   return (uint32_t)((uint64_t)raw * vref_mV / 4095ULL);
// }

// // === NEW: Solve resistor divider input voltage (Vin) from measured Vout
// // Vin = Vout * (Rhigh + Rlow) / Rlow
// float Divider_InputVoltage_V(float vout_mV, float Rhigh_ohm, float Rlow_ohm)
// {
//   return (vout_mV * (Rhigh_ohm + Rlow_ohm) / Rlow_ohm) / 1000.0f;
// }

/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
 * @brief  Reports the name of the source file and the source line number
 *         where the assert_param error has occurred.
 * @param  file: pointer to the source file name
 * @param  line: assert_param error line source number
 * @retval None
 */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */