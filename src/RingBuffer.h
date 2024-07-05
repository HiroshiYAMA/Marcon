/*
 * Copyright (c) 2024, Backcasters. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <iostream>
#include <vector>
#include <list>
#include <queue>
#include <mutex>
#include <cstring>

#if defined(USE_JETSON_UTILS) || defined(USE_VPI)
#include <jetson-utils/cudaMappedMemory.h>
#endif

template<typename T, bool pinned = true>
class RingBufferWithPool
{
private:
    int max_size;

    std::queue<T *> buffers;

    int idx;
    size_t buf_size;
    std::vector<T *> pool;

    std::mutex mtx;

public:
    RingBufferWithPool() = delete;
    RingBufferWithPool(int count)
    {
        max_size = count;
        idx = 0;
        buf_size = 0;
    }

    virtual ~RingBufferWithPool()
    {
        for (auto &e : pool) {
#if defined(USE_JETSON_UTILS) || defined(USE_VPI)
            if (e) CUDA_FREE_MAPPED(e, true);
#else
            if (e) free(e);
#endif
        }
    }

    bool alloc(size_t size)
    {
        if (size <= buf_size) return true;

        for (auto &e : pool) {
#if defined(USE_JETSON_UTILS) || defined(USE_VPI)
            if (e) CUDA_FREE_MAPPED(e, true);
#else
            if (e) free(e);
#endif
        }
        pool.clear();

        idx = 0;
        buf_size = size;

        while (pool.size() < max_size) {
#if defined(USE_JETSON_UTILS) || defined(USE_VPI)
            T *ptr = nullptr;
            cudaAllocMapped(&ptr, size, 1, true);
#else
            auto ptr = (T *)malloc(size);
#endif
            if (!ptr) {
                std::cout << "[RingBufferWithPool] Error : memory allocation." << std::endl;
                return false;
            } else {
                pool.push_back(ptr);
            }
        }

        return true;
    }

    void Read(T *dst, size_t size, bool latest = false)
    {
        std::lock_guard<std::mutex> lg(mtx);

#ifndef NDEBUG
        std::cout << "[RingBufferWithPool] queue size = " << buffers.size() << ", " << pool.size() << std::endl;
#endif

        T *src;

        if (buffers.empty()) {
            return;
        } else {
            src = latest ? buffers.back() : buffers.front();
        }

#ifdef USE_JETSON_UTILS_RING_BUFFER
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault);
#else
        memcpy(dst, src, size);
#endif
        if (buffers.size() > 1) {
            buffers.pop();
        }
    }
    void ReadLatest(T *dst, size_t size)
    {
        Read(dst, size, true);
    }

    T *get_read_buf(bool latest = false)
    {
        std::lock_guard<std::mutex> lg(mtx);

        if (buffers.empty()) {
            return nullptr;
        }

        return latest ? buffers.back() : buffers.front();
    }
    T *get_read_buf_latest()
    {
        return get_read_buf(true);
    }

    void pop_read_buf()
    {
        std::lock_guard<std::mutex> lg(mtx);

#ifndef NDEBUG
        std::cout << "[RingBufferWithPool] queue size = " << buffers.size() << ", " << pool.size() << std::endl;
#endif

        // if (!buffers.empty()) {
        if (buffers.size() > 1) {
            buffers.pop();
        }
    }

    T *Write(T *src, size_t size)
    {
        std::lock_guard<std::mutex> lg(mtx);

        if (pool.size() < max_size || pool.empty() || !src || size == 0) return nullptr;

        T *dst = nullptr;
        if (buffers.size() < max_size) {
            dst = pool[idx];
            idx++;
            if (idx >= max_size) idx = 0;
            buffers.push(dst);
        } else {
            auto idx_ow = idx - 1;
            if (idx_ow < 0) idx_ow = max_size - 1;
            dst = pool[idx_ow];    // overwrite latest.
        }
#ifdef USE_JETSON_UTILS_RING_BUFFER
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault);
#else
        memcpy(dst, src, size);
#endif

        return dst;
    }
};

template<typename T>
class RingBufferAsync
{
private:
    int max_size;

    std::queue<T> buffers;

    std::mutex mtx;

public:
    RingBufferAsync() = delete;
    RingBufferAsync(int count)
    {
        max_size = count;
    }

    virtual ~RingBufferAsync()
    {
        ;
    }

    T Read(bool latest = false)
    {
        std::lock_guard<std::mutex> lg(mtx);

#ifndef NDEBUG
        std::cout << "[RingBufferAsync] queue size = " << buffers.size() << std::endl;
#endif

        if (buffers.empty()) {
            return T{};
        } else if (buffers.size() == 1) {
            return buffers.front();
        }

        T val = latest ? buffers.back() : buffers.front();
        buffers.pop();

        return val;
    }
    T ReadLatest()
    {
        return Read(true);
    }

    T Peek(bool latest = false)
    {
        std::lock_guard<std::mutex> lg(mtx);

        if (buffers.empty()) {
            return T{};
        }

        return latest ? buffers.back() : buffers.front();
    }
    T PeekLatest()
    {
        return Peek(true);
    }

    void Write(T val)
    {
        std::lock_guard<std::mutex> lg(mtx);

        if (buffers.size() < max_size) {
            buffers.push(val);
        } else {
            buffers.back() = val;   // overwrite latest.
        }
    }
};
