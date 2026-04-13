/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <iostream>
#include <cstdio>

#ifdef _WIN32
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

#define ZE_CHECK(cmd) do {                         \
  ze_result_t e = cmd;                             \
  if( e != ZE_RESULT_SUCCESS ) {                   \
    printf("Level-Zero error at %s:%d code=%d\n",  \
            __FILE__,__LINE__, e);                 \
  }                                                \
} while(0)

void print_cerr() {
}

// Helper variadic template function to print messages to std::cerr
template <typename T>
void print_cerr(const T& t) {
    std::cerr << t;
}

template <typename T, typename... Rest>
void print_cerr(const T& t, const Rest&... rest) {
    std::cerr << t;
    print_cerr(rest...);
}

// The main macro
#define COND_CHECK(condition, ...) \
    do { \
        if (!(condition)) { \
            std::cerr << "ERROR: Condition '" << #condition << "' failed." \
                      << " File: " << __FILE__ << ", Line: " << __LINE__ << std::endl; \
            std::cerr << "Messages: "; \
            print_cerr(__VA_ARGS__); \
            std::cerr << std::endl; \
            throw std::runtime_error("Condition failed"); \
        } \
    } while (false)