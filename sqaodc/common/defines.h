#ifndef SQAOD_COMMON_DEFINES_H__
#define SQAOD_COMMON_DEFINES_H__

#include <stdarg.h>


enum {
    SQAODC_SIMD_ALIGNMENT = 32, /* FIXME: assuming AVX(2) */
};


#ifdef __GNUC__
#define FORMATATTR(stringIdx, firstToCheck) __attribute__((format(printf, stringIdx, firstToCheck)))
#else
#define FORMATATTR(stringIdx, firstToCheck)
#endif


namespace sqaod {

void __abort(const char *file, unsigned long line);
void __abort(const char *file, unsigned long line, const char *format, ...) FORMATATTR(3, 4);

template <class... Args> inline
void _abort(const char *file, unsigned long line, Args... args) {
    __abort(file, line, args...);
}

void __throwError(const char *file, unsigned long line);
void __throwError(const char *file, unsigned long line, const char *format, ...) FORMATATTR(3, 4);

template <class... Args> inline
void _throwError(const char *file, unsigned long line, Args... args) {
    __throwError(file, line, args...);
}


void log(const char *format, ...) FORMATATTR(1, 2);

}

/* FIXME: undef somewhere */
#define abort_(...) sqaod::_abort(__FILE__, __LINE__, __VA_ARGS__)
#define abortIf(cond, ...) if (cond) sqaod::_abort(__FILE__, __LINE__, __VA_ARGS__)
#define throwError(...) sqaod::_throwError(__FILE__, __LINE__, __VA_ARGS__)
#define throwErrorIf(cond, ...) if (cond) sqaod::_throwError(__FILE__, __LINE__, __VA_ARGS__)


#ifndef _DEBUG
#  ifndef NDEBUG
#    define NDEBUG
#  endif
#endif

#include <assert.h> /* Here's the only place to include assert.h */


#ifdef __linux__
/* macros generated by autoconf */
#include <sqaodc/common/sqaod_config.h>
#include <malloc.h>

inline void aligned_free(void *pv) {
    ::free(pv);
}

#endif

#ifdef _WIN32
#define SQAODC_CUDA_ENABLED
/* #define SQAOD_WITH_BLAS */

#include <malloc.h>

inline void *aligned_alloc(int alignment, size_t size) {
    return ::_aligned_malloc(size, alignment);
}

inline void aligned_free(void *pv) {
    ::_aligned_free(pv);
}

#endif


/* define this macro to enable range coverage tests in BF searchers. */
// #define SQAODC_ENABLE_RANGE_COVERAGE_TEST

#endif
