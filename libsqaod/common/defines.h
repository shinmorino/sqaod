#ifndef SQAOD_COMMON_DEFINES_H__
#define SQAOD_COMMON_DEFINES_H__

#include <stdarg.h>
#include <assert.h>

namespace sqaod {

typedef unsigned int SizeType;
typedef int IdxType;

/* FIXME: add format warning on gcc. */

void _abort(const char *file, unsigned long line, const char *msg);
void _abort(const char *file, unsigned long line, const char *format, va_list va);
void _throwError(const char *file, unsigned long line, const char *msg);
void _throwError(const char *file, unsigned long line, const char *format, va_list va);

}

/* FIXME: undef somewhere */
#define abort(msg) sqaod::_abort(__FILE__, __LINE__, msg)
#define abort_(msg, ...) sqaod::_abort(__FILE__, __LINE__, msg, __VA_ARGS__)
#define abortIf(cond, msg) if (cond) sqaod::_abort(__FILE__, __LINE__, msg);
#define abortIf_(cond, msg, ...) if (cond) sqaod::abort_(__FILE__, __LINE__, msg, __VA_ARGS__);
#define throwError(msg) throw sqaod::_throwError(__FILE__, __LINE__, msg);
#define throwError_(msg, ...) throw sqaod::_throwError(sqaod::_format(msg, __VA_ARGS__);
#define throwErrorIf(cond, msg) if (cond) sqaod::_throwError(__FILE__, __LINE__, msg)
#define throwErrorIf_(cond, msg, ...) if (cond) sqaod::_throwError(__FILE__, __LINE__, msg, __VA_ARGS__);

#endif
