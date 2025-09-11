#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <hip/hip_runtime.h>
#include <fstream>
#include <climits>
#include <algorithm>

// 定义无穷大值（题目要求）
constexpr int INF = 1073741823; // 2^30 - 1

// 声明 solve 函数（由 kernel.hip 实现）
extern "C" void solve(int* d_dist, int V);

#endif