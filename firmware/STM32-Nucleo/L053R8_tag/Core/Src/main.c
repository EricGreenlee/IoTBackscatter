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
#include "precomputed_samples.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
// === Streaming parameters ===
#define SAMPLE_RATE 100000 // 200 ksps DAC update rate
#define CARRIER_SAMPLES 4  // 4 samples per 50 kHz cycle
#define GOLD_LEN 127
#define SYMBOL_SAMPLES (GOLD_LEN * 16) // 2032 samples per encoded bit
#define BUF_SAMPLES 2048               // DMA circular buffer (2 halves)
#define HALF_SAMPLES (BUF_SAMPLES / 2) // 1024 per half
#define IDLE_BITS 4

// const int preamble[64] = {
//     1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,
//     1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,
//     1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,
//     1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0};
const int preamble[64] = {
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
// const int payload[16] = {0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1};
const int payload[16] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};

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
    char timer_msg[] = "Timer 6 started - 1kHz sine wave on PA4\r\n";
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

    HAL_Delay(1000); // Slower status updates
  }
  /* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Configure the main internal regulator output voltage
   */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
   * in the RCC_OscInitTypeDef structure.
   */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_MSI;
  RCC_OscInitStruct.MSIState = RCC_MSI_ON;
  RCC_OscInitStruct.MSICalibrationValue = 0;
  RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_5;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
   */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_MSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USART2;
  PeriphClkInit.Usart2ClockSelection = RCC_USART2CLKSOURCE_PCLK1;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
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

// void Generate_Sine_Table(void)
// {
//   for (int i = 0; i < SAMPLES; i++)
//   {
//     float angle = (2.0f * M_PI * i) / SAMPLES;
//     sine_wave[i] = (uint16_t)((sinf(angle) + 1.0f) * (DAC_MAX / 2));
//   }
// }

// void GenerateGoldTables(void)
// {
//   // 4-sample 50kHz sinusoid @ 200ksps: [mid, max, mid, min]
//   int16_t base_wave[4] = {2048, 4095, 2048, 0};

//   int idx = 0;
//   for (int i = 0; i < GOLD_LEN; i++)
//   {
//     for (int j = 0; j < 16; j++)
//     {
//       int carrier_idx = (j % 4);
//       int16_t sample = base_wave[carrier_idx];

//       // Multiply by goldcode chip (+1/-1)
//       if (goldcode[i] == -1)
//       {
//         sample = 4095 - sample; // phase inversion
//       }

//       gold_pos[idx] = sample;        // for data bit = 1
//       gold_neg[idx] = 4095 - sample; // for data bit = 0
//       idx++;
//     }
//   }
// }

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