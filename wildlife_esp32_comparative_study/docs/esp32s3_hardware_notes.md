# ESP32-S3 target hardware notes

Pick one specific module/dev board and pin it here before firmware work starts — "ESP32-S3" alone
is not a single hardware spec (RAM/flash/PSRAM vary by SKU).

- CPU: dual-core Xtensa LX7 @ up to 240 MHz, INT8/INT16 SIMD instructions (used by ESP-DL)
- SRAM: ~512 KB total (shared with stack, WiFi/BT stack if enabled, OS)
- PSRAM: 0 / 2 / 8 MB depending on module (octal SPI on N16R8 etc.) — slower than SRAM, budget
  explicitly rather than assuming it's free extra RAM
- Flash: 4-16 MB typical dev boards
- NN runtimes with official support: ESP-DL (Espressif, INT8-native) and TFLite-Micro via
  `esp-tflite-micro`. Neither has GNN message-passing or KAN spline ops — see PLAN.md §6/§7 for
  the custom-C workaround this study commits to.

## Study budgets (targets to report against, not hard cutoffs)
- Peak RAM: <=200 KB (no PSRAM) / <=1.5 MB (PSRAM variant, stretch)
- Latency: <=50 ms/inference
- Flash: <=512 KB/model

## Measurement tools needed
- Timing: `esp_timer_get_time()` around inference call
- Memory: `idf.py size`, `heap_caps_get_info()`
- Energy: external power profiler (e.g. INA219 board or Nordic PPK2) — confirm availability early
  (see PLAN.md §11 risk 4); without it, report latency x rated current as an estimated bound only
