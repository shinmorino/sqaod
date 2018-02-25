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
    pnMax = 6,
};

enum PreferenceName preferenceNameFromString(const char *name);

const char *preferenceNameToString(enum PreferenceName sp);

struct Preference {
    Preference(PreferenceName _name, SizeType _size) : name(_name), size(_size) { }
    Preference(PreferenceName _name, Algorithm _algo) : name(_name), algo(_algo) { }
    Preference() : name(pnUnknown) { }
    Preference(const Preference &) = default;

    PreferenceName name; /* must be the first member. */
    union {
        Algorithm algo;
        SizeType size;
        SizeType tileSize;
        SizeType nTrotters;
    };
};

typedef ArrayType<Preference> Preferences;

}
