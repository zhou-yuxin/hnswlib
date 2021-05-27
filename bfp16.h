#pragma once

#include <math.h>
#include <stdint.h>

class Bfp16 {

private:
    union {
        struct {
            uint16_t mantissa : 7;
            uint16_t exponent : 8;
            uint16_t sign : 1;
        };
        uint16_t u16;
    }
    storage;

    typedef union {
        struct {
            uint32_t mantissa : 23;
            uint32_t exponent : 8;
            uint32_t sign : 1;
        };
        float value;
        uint32_t u32;
        uint16_t u16s[2];
    }
    Fp32;

public:
    inline Bfp16() {
    }

    template <typename T>
    inline Bfp16(const T& x) {
        *this = x;
    }

    template <typename T>
    inline operator T() const {
        Fp32 a;
        a.u16s[0] = 0;
        a.u16s[1] = storage.u16;
        return T(a.value);
    }

    template <typename T>
    inline void operator =(const T& x) {
        Fp32 a;
        a.value = float(x);
        if(isnan(a.value)) {
            a.u16s[1] |= 0x7f;
        }
        else if(issubnormal(a.value)) {
            Fp32 b;
            b.u32 = 0x8000;
            b.sign = a.sign;
            a.value += b.value;
        }
        else if(isnormal(a.value)) {
            Fp32 b = a;
            b.mantissa = 0;
            a.value += b.value / 256.0f;
        }
        storage.u16 = a.u16s[1];
    }

};

typedef Bfp16 bfp16_t;