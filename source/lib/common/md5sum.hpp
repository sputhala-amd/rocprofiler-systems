// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <string_view>
#include <type_traits>

#include "traits.hpp"

namespace rocprofsys
{
inline namespace common
{

class md5sum
{
public:
    using size_type                = uint32_t;  // must be 32bit
    using raw_digest_t             = std::array<uint8_t, 16>;
    static constexpr int blocksize = 64;

    template <typename Tp, typename... Args>
    explicit md5sum(Tp&& arg, Args&&... args);

    /**
 * @brief Default-constructs an MD5 accumulator.
 *
 * Constructs an md5sum object in its initial state, ready to accept data via update()
 * and later produce a digest with finalize()/hexdigest().
 */
md5sum()              = default;
    /**
 * @brief Default destructor.
 *
 * Destroys the md5sum instance and its members.
 */
~md5sum()             = default;
    /**
 * @brief Default copy constructor.
 *
 * Creates a new md5sum by performing a memberwise copy of the source object.
 * The new instance begins with the same internal state, buffer, counters and
 * finalized/digest values as the original; subsequent modifications to one
 * object do not affect the other.
 */
md5sum(const md5sum&) = default;
    /**
 * @brief Move constructor.
 *
 * Constructs a new md5sum by moving the state from another instance.
 * Performs member-wise move of internal buffers and state; the source object
 * is left in a valid but unspecified state.
 */
md5sum(md5sum&&)      = default;

    /**
 * @brief Copy-assigns the MD5 computation state.
 *
 * Performs a memberwise copy of all internal state (finalized flag, counters,
 * buffer, state words, and digest). After assignment, the target object
 * represents the same MD5 computation state as the source.
 *
 * @return Reference to this md5sum after assignment.
 */
md5sum& operator=(const md5sum&) = default;
    /**
 * @brief Move-assigns the contents of another md5sum into this one.
 *
 * Performs a move assignment of the internal MD5 state, buffer, and digest.
 * The source object is left in a valid but unspecified state. Returns a
 * reference to *this.
 */
md5sum& operator=(md5sum&&)      = default;

    md5sum&      update(std::string_view inp);
    md5sum&      update(const unsigned char* buf, size_type length);
    md5sum&      update(const char* buf, size_type length);
    md5sum&      finalize();
    std::string  hexdigest() const;
    std::string  hexliteral() const;
    /**
 * @brief Return the raw 16-byte MD5 digest.
 *
 * @return raw_digest_t The 16-byte digest produced by the MD5 computation.
 */
raw_digest_t rawdigest() const { return digest; }

    template <typename Tp,
              typename Up = std::enable_if_t<std::is_arithmetic<Tp>::value, int>>
    md5sum& update(Tp inp);

    friend std::ostream& operator<<(std::ostream&, md5sum md5);

private:
    void transform(const uint8_t block[blocksize]);

    bool finalized = false;
    // 64bit counter for number of bits (lo, hi)
    std::array<uint32_t, 2>        count = { 0, 0 };
    std::array<uint8_t, blocksize> buffer{};  // overflow bytes from last 64 byte chunk
    // digest so far, initialized to magic initialization constants.
    std::array<uint32_t, 4> state = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 };
    std::array<uint8_t, 16> digest{};  // result
};

template <typename Tp, typename... Args>
/**
 * @brief Construct an md5sum by consuming one or more input values.
 *
 * This templated constructor feeds each provided argument into the MD5
 * update pipeline (via md5sum::update) and then finalizes the digest.
 *
 * Each argument is perfectly-forwarded to update; pointer types are rejected
 * at compile time (static_assert). After all inputs are processed, finalize()
 * is invoked so the object holds a completed digest (hexdigest(), rawdigest(),
 * and hexliteral() become available).
 */
md5sum::md5sum(Tp&& arg, Args&&... args)
{
    auto _update = [&](auto&& _val) {
        using value_type =
            std::remove_reference_t<std::remove_cv_t<std::decay_t<decltype(_val)>>>;
        static_assert(!std::is_pointer<value_type>::value,
                      "constructor cannot be called with pointer argument");
        update(std::forward<decltype(_val)>(_val));
    };

    _update(std::forward<Tp>(arg));
    (_update(std::forward<Args>(args)), ...);
    finalize();
}

template <typename Tp, typename Up>
/**
 * @brief Update the MD5 state with the raw byte representation of an arithmetic value.
 *
 * The template is constrained to arithmetic types; the object is treated as a POD
 * byte sequence and its in-memory representation (sizeof(Tp) bytes) is fed into
 * the hash. This performs no textual or numeric serialization — use explicit
 * conversions beforehand if a platform-independent encoding is required.
 *
 * @tparam Tp An arithmetic type (integral or floating point). A static assert
 *            enforces this constraint.
 * @param inp Value whose bytes will be appended to the MD5 input stream.
 * @return Reference to this md5sum instance to allow chaining.
 */
md5sum&
md5sum::update(Tp inp)
{
    static_assert(std::is_arithmetic<Tp>::value, "expected arithmetic type");
    return update(reinterpret_cast<const char*>(&inp), sizeof(Tp));
}

template <template <typename, typename...> class ContainerT, typename Tp,
          typename... TailT>
/**
 * @brief Compute the MD5 hex digest for a container of string-literal elements.
 *
 * Computes the MD5 message digest by feeding each element of the provided
 * container into an MD5 accumulator in iteration order and returning the final
 * digest as a lowercase hexadecimal string.
 *
 * Enabled only when the container's element type is a string literal (via
 * SFINAE). The function treats each element as a string view and updates the
 * digest with the element's bytes sequentially.
 *
 * @param inp Container whose element type is a string literal (e.g., array of
 *            string literals). Each element is hashed in iteration order.
 * @return std::string Lowercase hexadecimal MD5 digest of the data fed into
 *                     the accumulator.
 */
std::string
compute_md5sum(const ContainerT<Tp, TailT...>& inp,
               std::enable_if_t<traits::is_string_literal<Tp>(), int>)
{
    auto _val = md5sum{};
    for(const auto& itr : inp)
        _val.update(std::string_view{ inp });
    _val.finalize();
    return _val.hexdigest();
}

namespace
{

using size_type = typename md5sum::size_type;

// Constants for md5sumTransform routine.
constexpr uint32_t S11 = 7;
constexpr uint32_t S12 = 12;
constexpr uint32_t S13 = 17;
constexpr uint32_t S14 = 22;
constexpr uint32_t S21 = 5;
constexpr uint32_t S22 = 9;
constexpr uint32_t S23 = 14;
constexpr uint32_t S24 = 20;
constexpr uint32_t S31 = 4;
constexpr uint32_t S32 = 11;
constexpr uint32_t S33 = 16;
constexpr uint32_t S34 = 23;
constexpr uint32_t S41 = 6;
constexpr uint32_t S42 = 10;
constexpr uint32_t S43 = 15;
constexpr uint32_t S44 = 21;

// low level logic operations
static inline uint32_t
F(uint32_t x, uint32_t y, uint32_t z);

static inline uint32_t
G(uint32_t x, uint32_t y, uint32_t z);

static inline uint32_t
H(uint32_t x, uint32_t y, uint32_t z);

static inline uint32_t
I(uint32_t x, uint32_t y, uint32_t z);

static inline uint32_t
rotate_left(uint32_t x, int n);

static inline void
FF(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac);

static inline void
GG(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac);

static inline void
HH(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac);

static inline void
II(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac);

/**
 * @brief MD5 auxiliary function F.
 *
 * Computes the MD5 non-linear function F(x, y, z) = (x & y) | (~x & z),
 * which selects bits from y or z according to mask x.
 *
 * @param x Selector mask.
 * @param y Value selected when mask bit is 1.
 * @param z Value selected when mask bit is 0.
 * @return uint32_t Result of the F function.
 */
inline uint32_t
F(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) | (~x & z);
}

/**
 * @brief MD5 auxiliary function G.
 *
 * Computes the MD5 round-2 nonlinear function G(x, y, z) = (x & z) | (y & ~z).
 *
 * @param x First 32-bit word input.
 * @param y Second 32-bit word input.
 * @param z Third 32-bit word input.
 * @return uint32_t Resulting 32-bit word.
 */
inline uint32_t
G(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & z) | (y & ~z);
}

/**
 * @brief MD5 auxiliary function H.
 *
 * Computes the MD5 round-3 nonlinear function H(x, y, z) = x XOR y XOR z.
 *
 * @param x First 32-bit input word.
 * @param y Second 32-bit input word.
 * @param z Third 32-bit input word.
 * @return uint32_t Result of the bitwise XOR of the three inputs.
 */
inline uint32_t
H(uint32_t x, uint32_t y, uint32_t z)
{
    return x ^ y ^ z;
}

/**
 * @brief MD5 auxiliary function I.
 *
 * Computes the MD5 round-4 non-linear function I(x, y, z) = y XOR (x OR NOT z) on 32-bit words.
 *
 * @param x First 32-bit word operand.
 * @param y Second 32-bit word operand.
 * @param z Third 32-bit word operand.
 * @return uint32_t Result of the bitwise operation (32-bit).
 */
inline uint32_t
I(uint32_t x, uint32_t y, uint32_t z)
{
    return y ^ (x | ~z);
}

/**
 * @brief Rotate a 32-bit unsigned integer left by a specified number of bits.
 *
 * Performs a circular left rotation of the 32-bit value `x` by `n` bit positions.
 *
 * @param x Value to rotate.
 * @param n Number of bit positions to rotate left (expected in range [0, 31]).
 * @return uint32_t Result of rotating `x` left by `n` bits.
 *
 * @note Behavior is undefined if `n` is not within 0..31 due to C++ shift semantics.
 */
inline uint32_t
rotate_left(uint32_t x, int n)
{
    return (x << n) | (x >> (32 - n));
}

// FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4.
/**
 * @brief Perform the MD5 first-round transformation step (FF).
 *
 * Updates the accumulator word `a` with the result of the MD5 "FF" operation:
 * a := rotate_left(a + F(b,c,d) + x + ac, s) + b.
 *
 * @param a Reference to the accumulator word that will be updated.
 * @param b First source word used by the nonlinear function and added back after rotation.
 * @param c Second source word for the nonlinear function.
 * @param d Third source word for the nonlinear function.
 * @param x 32-bit word from the current message block.
 * @param s Number of bits to rotate left.
 * @param ac Additive constant for this step.
 */
inline void
FF(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac)
{
    a = rotate_left(a + F(b, c, d) + x + ac, s) + b;
}

/**
 * @brief Performs the MD5 "GG" (round 2) transformation step.
 *
 * Applies the second-round MD5 operation to update `a`:
 * a := a + G(b,c,d) + x + ac; then left-rotate by `s` bits and add `b`.
 *
 * @param a[in,out] Accumulator word updated in place with the result.
 * @param b First working word.
 * @param c Second working word.
 * @param d Third working word.
 * @param x Message word for this operation (32-bit).
 * @param s Rotation amount (number of bits to rotate left).
 * @param ac Additive constant for this step.
 */
inline void
GG(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac)
{
    a = rotate_left(a + G(b, c, d) + x + ac, s) + b;
}

/**
 * @brief MD5 round-3 transformation macro implemented as a function.
 *
 * Performs the "HH" operation from the MD5 specification: computes
 * a = ROTATE_LEFT(a + H(b, c, d) + x + ac, s) + b, where H is the
 * MD5 nonlinear function (b ^ c ^ d) and ROTATE_LEFT is a left bit-rotate.
 *
 * @param a In/out accumulator updated with the result.
 * @param b First input word.
 * @param c Second input word.
 * @param d Third input word.
 * @param x Message word for this operation.
 * @param s Rotation amount (number of bits to rotate left).
 * @param ac Additive constant for this step.
 */
inline void
HH(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac)
{
    a = rotate_left(a + H(b, c, d) + x + ac, s) + b;
}

/**
 * @brief MD5 round transformation helper for the fourth round (operation II).
 *
 * Applies the MD5-specific non-linear function I, adds the message word and
 * constant, performs a left rotation by s bits, and accumulates the result
 * into `a` as specified by the MD5 algorithm.
 *
 * @param a (in/out) Accumulated state word updated with the transformation.
 * @param b State word used as part of the computation and added to the result.
 * @param c State word input to the non-linear function I.
 * @param d State word input to the non-linear function I.
 * @param x 32-bit message word for this operation.
 * @param s Number of bits to rotate left.
 * @param ac Additive constant for this step (round constant).
 */
inline void
II(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac)
{
    a = rotate_left(a + I(b, c, d) + x + ac, s) + b;
}

/**
 * @brief Decode a byte array into 32-bit words (little-endian).
 *
 * Converts `len` bytes from `input` into `len/4` 32-bit words written to `output`.
 * Each group of four input bytes is interpreted in little-endian order:
 * output[i] = input[j] | input[j+1]<<8 | input[j+2]<<16 | input[j+3]<<24.
 *
 * @param output Destination array for decoded 32-bit words. Must have at least `len/4` elements.
 * @param input  Source byte array of length `len`.
 * @param len    Number of bytes to decode; must be a multiple of 4.
 */
void
decode(uint32_t output[], const uint8_t input[], size_type len)
{
    for(unsigned int i = 0, j = 0; j < len; i++, j += 4)
        output[i] = ((uint32_t) input[j]) | (((uint32_t) input[j + 1]) << 8) |
                    (((uint32_t) input[j + 2]) << 16) | (((uint32_t) input[j + 3]) << 24);
}

// encodes input (uint32_t) into output (unsigned char). Assumes len is
/**
 * @brief Encode 32-bit words into a byte array in little-endian order.
 *
 * Converts `len/4` 32-bit words from `input` into `len` bytes written to
 * `output`, storing each word as four bytes in little-endian order
 * (least-significant byte first).
 *
 * @param output Destination byte buffer; must be at least `len` bytes.
 * @param input  Source array of 32-bit words.
 * @param len    Number of bytes to produce; must be a multiple of 4.
 *
 * @note No bounds checks are performed; callers must ensure buffers are large enough.
 */
void
encode(uint8_t output[], const uint32_t input[], size_type len)
{
    for(size_type i = 0, j = 0; j < len; i++, j += 4)
    {
        output[j]     = input[i] & 0xff;
        output[j + 1] = (input[i] >> 8) & 0xff;
        output[j + 2] = (input[i] >> 16) & 0xff;
        output[j + 3] = (input[i] >> 24) & 0xff;
    }
}

}  // namespace

/**
 * @brief Process a single 64-byte MD5 message block and update the internal state.
 *
 * Decodes the given 64-byte input block into 32-bit words, performs the four
 * MD5 transformation rounds, and adds the results into the instance's state
 * words. Temporary working storage is cleared before returning.
 *
 * @param block Pointer to exactly md5sum::blocksize (64) bytes to process.
 */
void
md5sum::transform(const uint8_t block[blocksize])
{
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3], x[16];
    decode(x, block, blocksize);

    /* Round 1 */
    FF(a, b, c, d, x[0], S11, 0xd76aa478);  /* 1 */
    FF(d, a, b, c, x[1], S12, 0xe8c7b756);  /* 2 */
    FF(c, d, a, b, x[2], S13, 0x242070db);  /* 3 */
    FF(b, c, d, a, x[3], S14, 0xc1bdceee);  /* 4 */
    FF(a, b, c, d, x[4], S11, 0xf57c0faf);  /* 5 */
    FF(d, a, b, c, x[5], S12, 0x4787c62a);  /* 6 */
    FF(c, d, a, b, x[6], S13, 0xa8304613);  /* 7 */
    FF(b, c, d, a, x[7], S14, 0xfd469501);  /* 8 */
    FF(a, b, c, d, x[8], S11, 0x698098d8);  /* 9 */
    FF(d, a, b, c, x[9], S12, 0x8b44f7af);  /* 10 */
    FF(c, d, a, b, x[10], S13, 0xffff5bb1); /* 11 */
    FF(b, c, d, a, x[11], S14, 0x895cd7be); /* 12 */
    FF(a, b, c, d, x[12], S11, 0x6b901122); /* 13 */
    FF(d, a, b, c, x[13], S12, 0xfd987193); /* 14 */
    FF(c, d, a, b, x[14], S13, 0xa679438e); /* 15 */
    FF(b, c, d, a, x[15], S14, 0x49b40821); /* 16 */

    /* Round 2 */
    GG(a, b, c, d, x[1], S21, 0xf61e2562);  /* 17 */
    GG(d, a, b, c, x[6], S22, 0xc040b340);  /* 18 */
    GG(c, d, a, b, x[11], S23, 0x265e5a51); /* 19 */
    GG(b, c, d, a, x[0], S24, 0xe9b6c7aa);  /* 20 */
    GG(a, b, c, d, x[5], S21, 0xd62f105d);  /* 21 */
    GG(d, a, b, c, x[10], S22, 0x2441453);  /* 22 */
    GG(c, d, a, b, x[15], S23, 0xd8a1e681); /* 23 */
    GG(b, c, d, a, x[4], S24, 0xe7d3fbc8);  /* 24 */
    GG(a, b, c, d, x[9], S21, 0x21e1cde6);  /* 25 */
    GG(d, a, b, c, x[14], S22, 0xc33707d6); /* 26 */
    GG(c, d, a, b, x[3], S23, 0xf4d50d87);  /* 27 */
    GG(b, c, d, a, x[8], S24, 0x455a14ed);  /* 28 */
    GG(a, b, c, d, x[13], S21, 0xa9e3e905); /* 29 */
    GG(d, a, b, c, x[2], S22, 0xfcefa3f8);  /* 30 */
    GG(c, d, a, b, x[7], S23, 0x676f02d9);  /* 31 */
    GG(b, c, d, a, x[12], S24, 0x8d2a4c8a); /* 32 */

    /* Round 3 */
    HH(a, b, c, d, x[5], S31, 0xfffa3942);  /* 33 */
    HH(d, a, b, c, x[8], S32, 0x8771f681);  /* 34 */
    HH(c, d, a, b, x[11], S33, 0x6d9d6122); /* 35 */
    HH(b, c, d, a, x[14], S34, 0xfde5380c); /* 36 */
    HH(a, b, c, d, x[1], S31, 0xa4beea44);  /* 37 */
    HH(d, a, b, c, x[4], S32, 0x4bdecfa9);  /* 38 */
    HH(c, d, a, b, x[7], S33, 0xf6bb4b60);  /* 39 */
    HH(b, c, d, a, x[10], S34, 0xbebfbc70); /* 40 */
    HH(a, b, c, d, x[13], S31, 0x289b7ec6); /* 41 */
    HH(d, a, b, c, x[0], S32, 0xeaa127fa);  /* 42 */
    HH(c, d, a, b, x[3], S33, 0xd4ef3085);  /* 43 */
    HH(b, c, d, a, x[6], S34, 0x4881d05);   /* 44 */
    HH(a, b, c, d, x[9], S31, 0xd9d4d039);  /* 45 */
    HH(d, a, b, c, x[12], S32, 0xe6db99e5); /* 46 */
    HH(c, d, a, b, x[15], S33, 0x1fa27cf8); /* 47 */
    HH(b, c, d, a, x[2], S34, 0xc4ac5665);  /* 48 */

    /* Round 4 */
    II(a, b, c, d, x[0], S41, 0xf4292244);  /* 49 */
    II(d, a, b, c, x[7], S42, 0x432aff97);  /* 50 */
    II(c, d, a, b, x[14], S43, 0xab9423a7); /* 51 */
    II(b, c, d, a, x[5], S44, 0xfc93a039);  /* 52 */
    II(a, b, c, d, x[12], S41, 0x655b59c3); /* 53 */
    II(d, a, b, c, x[3], S42, 0x8f0ccc92);  /* 54 */
    II(c, d, a, b, x[10], S43, 0xffeff47d); /* 55 */
    II(b, c, d, a, x[1], S44, 0x85845dd1);  /* 56 */
    II(a, b, c, d, x[8], S41, 0x6fa87e4f);  /* 57 */
    II(d, a, b, c, x[15], S42, 0xfe2ce6e0); /* 58 */
    II(c, d, a, b, x[6], S43, 0xa3014314);  /* 59 */
    II(b, c, d, a, x[13], S44, 0x4e0811a1); /* 60 */
    II(a, b, c, d, x[4], S41, 0xf7537e82);  /* 61 */
    II(d, a, b, c, x[11], S42, 0xbd3af235); /* 62 */
    II(c, d, a, b, x[2], S43, 0x2ad7d2bb);  /* 63 */
    II(b, c, d, a, x[9], S44, 0xeb86d391);  /* 64 */

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;

    // Zeroize sensitive information.
    memset(x, 0, sizeof x);
}

/**
 * @brief Update the MD5 context with the bytes from a string view.
 *
 * Appends the bytes of @p inp to the running MD5 computation.
 *
 * @param inp Input bytes to incorporate into the digest.
 * @return md5sum& Reference to this object to allow call chaining.
 */
md5sum&
md5sum::update(std::string_view inp)
{
    return update(inp.data(), inp.length());
}

// md5sum block update operation. Continues an md5sum message-digest
/**
 * @brief Incorporates a sequence of bytes into the ongoing MD5 computation.
 *
 * Updates the internal bit counters and buffers, processes any complete 64-byte
 * blocks (calling the internal transform() for each), and stores any remaining
 * partial block for later updates or finalization. This does not finalize the
 * digest; call finalize() when all data has been provided.
 *
 * @param input Pointer to the input byte sequence to be hashed.
 * @param length Number of bytes available at `input`.
 * @return md5sum& Reference to this object to allow call chaining.
 */
md5sum&
md5sum::update(const unsigned char input[], size_type length)
{
    // compute number of bytes mod 64
    size_type index = count[0] / 8 % blocksize;

    // Update number of bits
    if((count[0] += (length << 3)) < (length << 3)) count[1]++;
    count[1] += (length >> 29);

    // number of bytes we need to fill in buffer
    size_type firstpart = 64 - index;
    size_type i         = 0;

    // transform as many times as possible.
    if(length >= firstpart)
    {
        // fill buffer first, transform
        memcpy(&buffer[index], input, firstpart);
        transform(buffer.data());

        // transform chunks of blocksize (64 bytes)
        for(i = firstpart; i + blocksize <= length; i += blocksize)
            transform(&input[i]);

        index = 0;
    }

    // buffer remaining input
    memcpy(&buffer[index], &input[i], length - i);

    return *this;
}

/**
 * @brief Update the MD5 state with a signed-char buffer.
 *
 * Convenience overload that accepts a signed `char` buffer and forwards its
 * bytes to the unsigned-byte `update` overload.
 *
 * @param input Pointer to the input buffer (may contain binary data).
 * @param length Number of bytes to read from `input`.
 * @return md5sum& Reference to this object to allow call chaining.
 */
md5sum&
md5sum::update(const char input[], size_type length)
{
    return update((const unsigned char*) input, length);
}

// md5sum finalization. Ends an md5sum message-digest operation, writing the
/**
 * @brief Finalize the MD5 computation and produce the digest.
 *
 * Converts any buffered input into the final 16-byte MD5 digest by padding
 * the message, appending the 64-bit message length (in bits), encoding the
 * internal state into the digest buffer, and clearing sensitive internal
 * state. Calling finalize() more than once has no additional effect.
 *
 * After successful finalization:
 * - rawdigest() returns the 16-byte binary digest.
 * - hexdigest() and hexliteral() return string representations of the digest.
 *
 * @return md5sum& Reference to this instance (finalized).
 */
md5sum&
md5sum::finalize()
{
    static unsigned char padding[64] = { 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    if(!finalized)
    {
        // Save number of bits
        unsigned char bits[8];
        encode(bits, count.data(), 8);

        // pad out to 56 mod 64.
        size_type index  = count[0] / 8 % 64;
        size_type padLen = (index < 56) ? (56 - index) : (120 - index);
        update(padding, padLen);

        // Append length (before padding)
        update(bits, 8);

        // Store state in digest
        encode(digest.data(), state.data(), 16);

        // Zeroize sensitive information.
        memset(buffer.data(), 0, sizeof buffer);
        memset(count.data(), 0, sizeof count);

        finalized = true;
    }

    return *this;
}

/**
 * @brief Returns the finalized MD5 digest as a lowercase hexadecimal string.
 *
 * If the md5sum has not been finalized (finalize() not called), returns an
 * empty string. When finalized, returns a 32-character lowercase hex string
 * representing the 16-byte MD5 digest.
 *
 * @return std::string 32-char lowercase hex representation of the digest, or
 * an empty string if not finalized.
 */
std::string
md5sum::hexdigest() const
{
    if(!finalized) return std::string{};

    char buf[33];
    for(int i = 0; i < 16; i++)
        snprintf(buf + i * 2, 3, "%02x", digest[i]);
    buf[32] = '\0';

    return std::string(buf);
}

/**
 * @brief Return the finalized digest as a SQL-style hex literal.
 *
 * If the digest has been finalized, returns a string of the form `X'...'`
 * where each byte of the 16-byte MD5 digest is emitted as two lowercase
 * hexadecimal characters. If the md5sum is not finalized, returns an empty
 * string.
 *
 * @return std::string Hex-literal representation of the finalized digest, or
 * an empty string if not finalized.
 */
std::string
md5sum::hexliteral() const
{
    if(!finalized) return std::string{};

    auto _oss = std::ostringstream{};
    _oss << "X'";
    for(auto itr : rawdigest())
        _oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(itr);
    _oss << "'";
    return _oss.str();
}

/**
 * @brief Inserts the MD5 hex digest into an output stream.
 *
 * Writes the object's hex digest (as returned by md5sum::hexdigest()) to the provided
 * output stream and returns the stream reference for chaining.
 *
 * If the md5sum object has not been finalized, its hexdigest() is an empty string,
 * so nothing will be written.
 *
 * @param out Stream to write the hex digest into.
 * @param md5 The md5sum whose hex digest will be written (taken by value).
 * @return std::ostream& Reference to the same output stream.
 */
std::ostream&
operator<<(std::ostream& out, md5sum md5)
{
    return out << md5.hexdigest();
}

/**
 * @brief Compute the MD5 hex digest of a string view.
 *
 * Computes the MD5 hash of the provided input and returns the finalized
 * digest as a lowercase hexadecimal string.
 *
 * @param inp Input data to hash.
 * @return std::string Lowercase hexadecimal representation of the MD5 digest (32 hex characters).
 */
std::string
compute_md5sum(std::string_view inp)
{
    return md5sum{ inp }.finalize().hexdigest();
}

}  // namespace common
}  // namespace rocprofsys
