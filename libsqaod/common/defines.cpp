#include "defines.h"
#include "undef.h"

#include <stdlib.h>
#include <stdexcept>

using namespace sqaod;

void sqaod::__abort(const char *file, unsigned long line) {
    fprintf(stderr, "%s:%d %s\n", file, (int)line, "internal error, aborted.");
    abort();
}

void sqaod::__abort(const char *file, unsigned long line, const char *msg) {
    fprintf(stderr, "%s:%d %s\n", file, (int)line, msg);
    abort();
}

void sqaod::__abort(const char *file, unsigned long line, const char *format, va_list va) {
    char msg[512];
    vsnprintf(msg, sizeof(msg), format, va);
    fprintf(stderr, "%s:%d %s\n", file, (int)line, msg);
    abort();
}

void sqaod::__throwError(const char *file, unsigned long line) {
    __throwError(file, line, "abort");
}

void sqaod::__throwError(const char *file, unsigned long line, const char *msg) {
    char buffer[512];
    snprintf(buffer, sizeof(buffer), "%s:%d %s\n", file, (int)line, msg);
    throw std::runtime_error(buffer);
}

void sqaod::__throwError(const char *file, unsigned long line, const char *format, va_list va) {
    char msg[512];
    vsnprintf(msg, sizeof(msg), format, va);
    __throwError(file, line, msg);
}
