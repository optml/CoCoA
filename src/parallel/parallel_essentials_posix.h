/*
 This file is based on GraphLab.

 GraphLab is free software: you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as
 published by the Free Software Foundation, either version 3 of
 the License, or (at your option) any later version.
 */

#ifndef PARALLEL_ESSENTIALS_POSIX
#define PARALLEL_ESSENTIALS_POSIX

#include <stdint.h>

namespace parallel {


/******************************************************************
 atomic_compare_and_swap, the primitive we use in atomic_add.
 this is equivalent to the following:
 \code
 if (a==oldval) { a = newval; return true; }
 else { return false; }
 \endcode
 */
template<typename T>
bool atomic_compare_and_swap(T& a, const T &oldval, const T &newval) {
    return __sync_bool_compare_and_swap(&a, oldval, newval);
}
;

template<typename T>
bool atomic_compare_and_swap(volatile T& a, const T &oldval, const T &newval) {
    return __sync_bool_compare_and_swap(&a, oldval, newval);
}
;

template<>
inline bool atomic_compare_and_swap(volatile double &a, const double &oldval,
        const double &newval) {
    return __sync_bool_compare_and_swap(
            reinterpret_cast<volatile uint64_t*> (&a),
            *reinterpret_cast<const uint64_t*> (&oldval),
            *reinterpret_cast<const uint64_t*> (&newval));
}
;

inline bool atomic_compare_and_swap(volatile float& a, const float &oldval,
        const float &newval) {
    return __sync_bool_compare_and_swap(
            reinterpret_cast<volatile uint32_t*> (&a),
            *reinterpret_cast<const uint32_t*> (&oldval),
            *reinterpret_cast<const uint32_t*> (&newval));
}
;


/*******************************************************************/
// atomic_add, which is what we use, really


inline void atomic_add(volatile double &variable, const double valueToAdd) {
    double v, n;
    do {
        v = variable;
        n = v + valueToAdd;
    } while (!atomic_compare_and_swap(variable, v, n));
}


inline void atomic_add(volatile float &variable, const float valueToAdd) {
    float v, n;
    do {
        v = variable;
        n = v + valueToAdd;
    } while (!atomic_compare_and_swap(variable, v, n));

}

/*******************************************************************/
// Further bits

template<typename T>
void atomic_exchange(T& a, T& b) {
    b = __sync_lock_test_and_set(&a, b);
}
;

template<typename T>
T fetch_and_store(T& a, const T& newval) {
    return __sync_lock_test_and_set(&a, newval);
}
;


inline void atomic_add_without_load(volatile double *variable,
        const double valueToAdd, const double originalValue) {
    double v, n;

    v = originalValue;
    n = v + valueToAdd;
    bool doneInFirstStep = atomic_compare_and_swap(*variable, v, n);
    if (!doneInFirstStep) {
        do {
            v = variable[0];
            n = v + valueToAdd;
        } while (!atomic_compare_and_swap(*variable, v, n));
    }
}

inline void atomic_add_without_load(volatile float *variable,
        const float valueToAdd, const float originalValue) {
    float v, n;

    v = originalValue;
    n = v + valueToAdd;
    bool doneInFirstStep = atomic_compare_and_swap(*variable, v, n);
    if (!doneInFirstStep) {
        do {
            v = variable[0];
            n = v + valueToAdd;
        } while (!atomic_compare_and_swap(*variable, v, n));
    }
}


template<typename T>
class atomic {
public:
    volatile T value;
    atomic(const T& value = 0) :
        value(value) {
    }
    T inc() {
        return __sync_add_and_fetch(&value, 1);
    }
    T dec() {
        return __sync_sub_and_fetch(&value, 1);
    }
    T inc(T val) {
        return __sync_add_and_fetch(&value, val);
    }
    T dec(T val) {
        return __sync_sub_and_fetch(&value, val);
    }
    T inc_ret_last() {
        return __sync_fetch_and_add(&value, 1);
    }
    T dec_ret_last() {
        return __sync_fetch_and_sub(&value, 1);
    }
    T inc_ret_last(T val) {
        return __sync_fetch_and_add(&value, val);
    }
    T dec_ret_last(T val) {
        return __sync_fetch_and_sub(&value, val);
    }
    void add(float val) {
        atomic_add(value, val);
    }
};

}
#endif // PARALLEL_ESSENTIALS_POSIX
