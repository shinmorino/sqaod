#pragma once

#include <sqaodc/common/Array.h>

namespace sqaod {


enum Algorithm {
    algoUnknown,
    algoDefault,
    algoNaive,
    algoColoring,
    algoBruteForceSearch,
};


const char *algorithmToString(Algorithm algo);

Algorithm algorithmFromString(const char *algoStr);


enum PreferenceName {
    pnUnknown = 0,
    pnAlgorithm = 1,
    pnNumTrotters = 2, /* for annealers */
    pnTileSize = 3,    /* tileSize for brute force searchers */
    pnTileSize0 = 4,   /* tileSize0 for bipartite graph searchers */
    pnTileSize1 = 5,   /* tileSize1 for bipartite graph searchers */
    pnPrecision = 6,
    pnDevice = 7,
    pnMax = 8,
};

enum PreferenceName preferenceNameFromString(const char *name);

const char *preferenceNameToString(enum PreferenceName pn);

struct Preference {
    Preference(PreferenceName _name, SizeType _size) : name(_name), size(_size) { }
    Preference(PreferenceName _name, Algorithm _algo) : name(_name), algo(_algo) { }
    Preference(PreferenceName _name, const char *_str) : name(_name), str(_str) { }
    Preference() : name(pnUnknown) { }
    Preference(const Preference &) = default;

    PreferenceName name; /* must be the first member. */
    union {
        SizeType size;
        const char *str;

        Algorithm algo;
        SizeType tileSize;
        SizeType nTrotters;
        const char *precision;
        const char *device;
    };
};

typedef ArrayType<Preference> Preferences;

}
