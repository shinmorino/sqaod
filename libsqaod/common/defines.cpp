#include "defines.h"
#undef abort

#include <stdlib.h>
#include <stdexcept>

using namespace sqaod;

void sqaod::_abort(const char *file, unsigned long line, const char *msg) {
    fprintf(stderr, "%s:%d %s\n", file, (int)line, msg);
    abort();
}

void sqaod::_abort(const char *file, unsigned long line, const char *format, va_list va) {
    char msg[512];
    vsnprintf(msg, sizeof(msg), format, va);
    fprintf(stderr, "%s:%d %s\n", file, (int)line, msg);
    abort();
}

void sqaod::_throwError(const char *file, unsigned long line, const char *msg) {
    char buffer[512];
    snprintf(buffer, sizeof(buffer), "%s:%d %s\n", file, (int)line, msg);
    throw std::runtime_error(buffer);
}

void sqaod::_throwError(const char *file, unsigned long line, const char *format, va_list va) {
    char msg[512], buffer[512];
    vsnprintf(msg, sizeof(msg), format, va);
    snprintf(buffer, sizeof(msg), "%s:%d %s\n", file, (int)line, msg);
    throw std::runtime_error(buffer);
}
