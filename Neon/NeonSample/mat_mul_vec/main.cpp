/*
 * @Description:
 * @Author: kkchen
 * @Email: kkchen.lg@qq.com
 * @Date: 2023-02-07 18:15:45
 * @LastEditTime: 2023-02-07 19:18:47
 * @LastEditors: kkchen
 */
#include <chrono>
#include <cstring>
#include <ctime>
#include <iostream>
#include <random>

#include "arm_neon.h"

void genrandom(float* src, int data_len) {
  std::uniform_real_distribution<double> u(-1, 1);
  std::default_random_engine e(std::time(nullptr));
  for (int i = 0; i < data_len; i++) {
    src[i] = u(e);
  }
}

void c_mmulv(float* dsti, float* dstr, float* ai, float* ar, float* bi,
             float* br, int rows, int cols) {
  for (int j = 0; j < rows; j++) {
    for (int i = 0; i < cols; i++) {
      dstr[j] += ar[j * cols + i] * br[j * cols + i];
      dstr[j] += ai[j * cols + i] * bi[j * cols + i];
      dsti[j] += ai[j * cols + i] * (-1.0f) * br[j * cols + i];
      dsti[j] += ar[j * cols + i] * bi[j * cols + i];
    }
  }
}

void complex_real_conv16_version(float* dsti, float* dstr, float* ai, float* ar,
                                 float* br, float* bi, int rows, int cols) {
  for (int j = 0; j < rows; j++) {
    float32x4_t resultr_vec = vdupq_n_f32(0.0f);
    float32x4_t resulti_vec = vdupq_n_f32(0.0f);
    for (int i = 0; i < cols; i += 4) {
      float32x4_t ai_vec = vld1q_f32(ai + j * cols + i);
      float32x4_t ar_vec = vld1q_f32(ar + j * cols + i);
      float32x4_t bi_vec = vld1q_f32(bi + j * cols + i);
      float32x4_t br_vec = vld1q_f32(br + j * cols + i);

      resultr_vec = vmlaq_f32(resultr_vec, ar_vec, br_vec);
      resultr_vec = vmlaq_f32(resultr_vec, ai_vec, bi_vec);

      resulti_vec = vmlaq_f32(resulti_vec, vnegq_f32(ai_vec), br_vec);
      resulti_vec = vmlaq_f32(resulti_vec, ar_vec, bi_vec);
    }
    resultr_vec = vpaddq_f32(resultr_vec, resultr_vec);
    dstr[j] = vgetq_lane_f32(resultr_vec, 0) + vgetq_lane_f32(resultr_vec, 1);

    resulti_vec = vpaddq_f32(resulti_vec, resulti_vec);
    dsti[j] = vgetq_lane_f32(resulti_vec, 0) + vgetq_lane_f32(resulti_vec, 1);
  }
}

void calmae(float* a, float* b, int data_len) {
  float mae = 0.0f;
  for (int i = 0; i < data_len; i++) {
    mae += std::fabs(a[i] - b[i]);
  }

  std::cout << "mae : " << mae << " >>>>> " << a[0] << " " << b[0] << std::endl;
}

int main(int argc, char** argv) {
  int rows = atoi(argv[1]);
  int cols = atoi(argv[2]);
  int data_len = rows * cols;
  std::cout << "rows " << rows << " cols " << cols << std::endl;

  float* ai = new float[rows * cols];
  float* ar = new float[rows * cols];
  float* bi = new float[rows * cols];
  float* br = new float[rows * cols];
  float* dstr = new float[rows];
  float* dsti = new float[rows];
  float* dsttmpr = new float[rows];
  float* dsttmpi = new float[rows];

  memset(dstr, 0, rows * sizeof(float));
  memset(dsti, 0, rows * sizeof(float));
  memset(dsttmpi, 0, rows * sizeof(float));
  memset(dsttmpr, 0, rows * sizeof(float));

  genrandom(ai, data_len);
  genrandom(ar, data_len);
  genrandom(bi, data_len);
  genrandom(br, data_len);

  auto start = std::chrono::steady_clock::now();
  c_mmulv(dsti, dstr, ai, ar, bi, br, rows, cols);
  auto end = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "c cost " << duration.count() / 1000.0 << std::endl;

  start = std::chrono::steady_clock::now();
  complex_real_conv16_version(dsttmpi, dsttmpr, ai, ar, bi, br, rows, cols);
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "neon cost " << duration.count() / 1000.0 << " ms" << std::endl;

  calmae(dsti, dsttmpi, rows);
  calmae(dstr, dsttmpr, rows);

  return 0;
}