#include "defines.h"
#include "undef.h"

#include <stdarg.h>
#include <stdlib.h>
#include <stdexcept>

using namespace sqaod;

void sqaod::__abort(const char *file, unsigned long line) {
    fprintf(stderr, "%s:%d %s\n", file, (int)line, "internal error, aborted.");
    abort();
}

void sqaod::__abort(const char *file, unsigned long line, const char *format, ...) {
    char msg[512];

    va_list va;
    va_start(va, format);
    vsnprintf(msg, sizeof(msg), format, va);
    va_end(va);
    fprintf(stderr, "%s:%d %s\n", file, (int)line, msg);
    abort();
}

void sqaod::__throwError(const char *file, unsigned long line) {
    __throwError(file, line, "abort");
}

void sqaod::__throwError(const char *file, unsigned long line, const char *format, ...) {
    char msg[512];

    va_list va;
    va_start(va, format);
    vsnprintf(msg, sizeof(msg), format, va);
    va_end(va);
    
    char buffer[512];
    snprintf(buffer, sizeof(buffer), "%s:%d %s\n", file, (int)line, msg);
    throw std::runtime_error(buffer);
}
