#include "main.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
// #include "stm32l0xx_hal_i2c.h"
// #include "stm32l0xx_hal_adc.h"
// #include "precomputed_samples_GC0.h"

// ===== Waveform synthesis at 800 kS/s with 100 kHz subcarrier =====
#define FS_DAC 800000u                 // DAC sample rate
#define F_SC 100000u                   // subcarrier (kept coherent with cycles/chip)
#define SPP 8                          // samples per subcarrier period (power of 2)
#define CYCLES_PER_CHIP 4              // integer cycles/chip for continuous phase
#define SPCHIP (SPP * CYCLES_PER_CHIP) // samples per chip (32)
#define IDLE_SAMPLES_PER_BIT (SPCHIP * GOLD_LEN)

// Symbol/chip structure (unchanged Gold length)
#define GOLD_LEN 127

// DAC scaling
#define DAC_MID 2048
// #define DAC_AMP 1800 // leave headroom; adjust to taste
#define DAC_AMP 800

// Optional envelope pulse shaping (RRC later)
#define USE_RRC 0 // 0=rectangular envelope (fast memcpy), 1=RRC (envelope FIR)

// RRC parameters (only used if USE_RRC=1)
#define RRC_TAPS 33                    // example length (span ~2 chips on each side)
extern const float rrc_taps[RRC_TAPS]; // put your taps in a separate .c later
const float rrc_taps[RRC_TAPS] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

// #define SYMBOL_SAMPLES (GOLD_LEN * 16)
#define BUF_SAMPLES 2048
#define HALF_SAMPLES (BUF_SAMPLES / 2)
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

static int packet_bits[80];
static uint16_t dma_buf[BUF_SAMPLES]; // 4 KB

// ±1 Gold sequence (use your existing contents)
// extern const int8_t gold[GOLD_LEN];  // fill in precomputed ±1 values
const int8_t gold[GOLD_LEN] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1,
    -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1,
    -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1,
    1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1,
    1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1,
    1, 1, -1, -1, 1, -1, 1, 1, -1};

// Fast path: precomputed per-chip waveforms (rectangular envelope)
static uint16_t CHIP_POS[SPCHIP];
static uint16_t CHIP_NEG[SPCHIP];

// Optional: tiny 8-pt sine for constructing CHIP_POS/NEG
static const int16_t SIN8[SPP] = {0, 707, 1000, 707, 0, -707, -1000, -707};

// Streaming state
static int bit_idx = 0;   // 0..79
static int chip_idx = 0;  // 0..126
static int s_in_chip = 0; // 0..SPCHIP-1

// Idle handling (reuse your logic)
static int in_idle = 0;
static int idle_sent = 0;
static int idle_remain = SPCHIP * GOLD_LEN; // one "idle bit" reuses symbol span

// static int cur_bit = 0;
// static int sym_offset = 0;

// Private Variables
DAC_HandleTypeDef hdac;
TIM_HandleTypeDef htim6;

UART_HandleTypeDef huart2;

DMA_HandleTypeDef hdma_dac_ch1;

// Function prototypes
void SystemClock_Config(void);
void MX_GPIO_Init(void);
void MX_USART2_UART_Init(void);
void MX_DAC_Init(void);
void MX_TIM6_Init(void);
/* USER CODE BEGIN PFP */
void MX_DMA_Init(void);
void HAL_DAC_ConvHalfCpltCallbackCh1(DAC_HandleTypeDef *hdac);
void HAL_DAC_ConvCpltCallbackCh1(DAC_HandleTypeDef *hdac);
static void BuildPacket(void);
static void FillHalf(int half_index);
static void restart_packet_stream(void);
static void enter_idle(void);
static void BuildChipWave(void);

int main(void)
{
    HAL_Init();

    SystemClock_Config();

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
    BuildChipWave();
    restart_packet_stream(); // <— add this line

    FillHalf(0);
    FillHalf(1);

    HAL_DAC_Start_DMA(&hdac, DAC_CHANNEL_1, (uint32_t *)dma_buf, BUF_SAMPLES, DAC_ALIGN_12B_R);

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

    while (1)
    {
        /* USER CODE END WHILE */

        /* USER CODE BEGIN 3 */
        HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);

        HAL_Delay(1000); // Slower status updates
    }
    /* USER CODE END 3 */
}

// Functions
//  custom clock config to increase the maximum stable DAC rate
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

void MX_TIM6_Init(void)
{
    TIM_MasterConfigTypeDef sMasterConfig = {0};
    __HAL_RCC_TIM6_CLK_ENABLE();

    htim6.Instance = TIM6;
    htim6.Init.Prescaler = 0;
    htim6.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim6.Init.Period = (SystemCoreClock / FS_DAC) - 1; // FS_DAC controls DAC update
    htim6.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    HAL_TIM_Base_Init(&htim6);

    sMasterConfig.MasterOutputTrigger = TIM_TRGO_UPDATE;
    sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
    HAL_TIMEx_MasterConfigSynchronization(&htim6, &sMasterConfig);
}

static void BuildChipWave(void)
{
    // Stitch 4 periods of 8 samples → 32-sample sinusoid centered at DAC_MID
    int k = 0;
    for (int p = 0; p < CYCLES_PER_CHIP; ++p)
    {
        for (int i = 0; i < SPP; ++i, ++k)
        {
            int32_t v = DAC_MID + (int32_t)DAC_AMP * SIN8[i] / 1000;
            if (v < 0)
                v = 0;
            else if (v > 4095)
                v = 4095;
            CHIP_POS[k] = (uint16_t)v;
        }
    }
    // Inverted (anti-phase)
    for (int i = 0; i < SPCHIP; ++i)
    {
        int32_t v = 2 * DAC_MID - CHIP_POS[i];
        if (v < 0)
            v = 0;
        else if (v > 4095)
            v = 4095;
        CHIP_NEG[i] = (uint16_t)v;
    }
}

// Inline Helpers
static inline int chip_sign_for_current_bit(void)
{
    // bit 1 → +1, bit 0 → −1 (your convention); Gold is ±1
    int b = packet_bits[bit_idx] ? +1 : -1;
    return gold[chip_idx] * b;
}

static inline const uint16_t *chip_table_for_current_bit(void)
{
    return (chip_sign_for_current_bit() > 0) ? CHIP_POS : CHIP_NEG;
}

static void advance_chip_bit_state(void)
{
    if (++chip_idx == GOLD_LEN)
    {
        chip_idx = 0;
        if (++bit_idx == 80)
        {                 // packet done → idle or restart
            enter_idle(); // or bit_idx=0; if you prefer continuous loop
        }
    }
}

static void restart_packet_stream(void)
{
    // cur_bit = 0;
    // sym_offset = 0;
    bit_idx = 0;
    chip_idx = 0;
    s_in_chip = 0;

    // leave idle mode
    in_idle = 0;
    idle_sent = 0;
    idle_remain = IDLE_SAMPLES_PER_BIT;
}

static void enter_idle(void)
{
    in_idle = 1;
    idle_sent = 0;
    idle_remain = IDLE_SAMPLES_PER_BIT;
}

static void FillHalf_Rect(uint16_t *dst, int count)
{
    int remain = count;

    while (remain > 0)
    {
        const uint16_t *tab = chip_table_for_current_bit();
        int n_chip = SPCHIP - s_in_chip;
        int n_copy = (n_chip < remain) ? n_chip : remain;

        memcpy(dst, &tab[s_in_chip], (size_t)n_copy * sizeof(uint16_t));
        dst += n_copy;
        remain -= n_copy;
        s_in_chip += n_copy;

        if (s_in_chip == SPCHIP)
        {
            s_in_chip = 0;
            advance_chip_bit_state();
        }
    }
}

// Prototype for using RRC filtering if enabled
#if USE_RRC
// Keep a short envelope history for FIR across chip boundaries
static float env_hist[SPCHIP * 2]; // enough headroom for overlap
static int env_hist_len = 0;

// Produce 'count' envelope samples (float), shaped by FIR over chip edges
// Then multiply by sinusoid and write to dst.
static void FillHalf_RRC(uint16_t *dst, int count)
{
    // Step 1: ensure env_hist has at least 'count' samples appended
    while (env_hist_len < count + RRC_TAPS)
    {
        // Append one chip worth of envelope (+1 or −1)
        float env = (float)chip_sign_for_current_bit();
        for (int i = 0; i < SPCHIP; ++i)
        {
            if (env_hist_len < (int)(sizeof(env_hist) / sizeof(env_hist[0])))
                env_hist[env_hist_len++] = env;
        }
        // next chip
        if (++chip_idx == GOLD_LEN)
        {
            chip_idx = 0;
            if (++bit_idx == 80)
                enter_idle(); // or restart_packet_stream();
        }
    }

    // Step 2: convolve env_hist with rrc_taps (scalar FIR) and multiply by sinusoid
    // To keep phase continuous we still use the same 8-sample sine progression
    static int phase = 0; // 0..SPP-1, continuous across calls
    for (int n = 0; n < count; ++n)
    {
        // FIR on envelope
        float acc = 0.f;
        for (int k = 0; k < RRC_TAPS; ++k)
            acc += rrc_taps[k] * env_hist[n + (RRC_TAPS - 1 - k)];

        // Multiply by sine at this sample
        int16_t s = SIN8[phase];
        int32_t v = DAC_MID + (int32_t)((float)DAC_AMP * acc * ((float)s / 1000.f));
        if (v < 0)
            v = 0;
        else if (v > 4095)
            v = 4095;
        dst[n] = (uint16_t)v;

        // advance sine phase
        phase = (phase + 1) & (SPP - 1);
    }

    // Step 3: slide the env_hist by 'count'
    memmove(env_hist, &env_hist[count], (env_hist_len - count) * sizeof(float));
    env_hist_len -= count;
}
#endif

static void FillHalf(int half_index)
{
    uint16_t *dst = &dma_buf[half_index * HALF_SAMPLES];

    if (!in_idle)
    {
#if USE_RRC
        FillHalf_RRC(dst, HALF_SAMPLES);
#else
        FillHalf_Rect(dst, HALF_SAMPLES);
#endif
    }
    else
    {
        // char idle_msg[] = "In Idle\r\n";
        // HAL_UART_Transmit(&huart2, (uint8_t *)idle_msg, sizeof(idle_msg) - 1, HAL_MAX_DELAY);
        // Idle at mid-level for one symbol span per "idle bit"
        int remain = HALF_SAMPLES;
        while (remain > 0)
        {
            int to_fill = (idle_remain < remain) ? idle_remain : remain;
            for (int i = 0; i < to_fill; ++i)
                dst[i] = DAC_MID;
            dst += to_fill;
            remain -= to_fill;
            idle_remain -= to_fill;
            if (idle_remain == 0)
            {
                idle_sent++;
                if (idle_sent >= IDLE_BITS)
                    restart_packet_stream();
                else
                    idle_remain = IDLE_SAMPLES_PER_BIT;
            }
        }
    }
}

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