#include "Preference.h"
#include <string.h>

#ifdef _WIN32
#define strcasecmp _stricmp
#endif

using namespace sqaod;

const char *sqaod::algorithmToString(Algorithm algo) {
    switch (algo) {
    case algoNaive:
        return "naive";
    case algoColoring:
        return "coloring";
    case algoBruteForceSearch:
        return "brute_force_search";
    case algoDefault:
        return "default";
    case algoUnknown:
    default:
        return "unknown";
    }
}

Algorithm sqaod::algorithmFromString(const char *algoStr) {
    if (strcasecmp("naive", algoStr) == 0)
        return algoNaive;
    if (strcasecmp("coloring", algoStr) == 0)
        return algoColoring;
    if (strcasecmp("brute_force_search", algoStr) == 0)
        return algoBruteForceSearch;
    if (strcasecmp("default", algoStr) == 0)
        return algoDefault;
    return algoUnknown;
}


enum PreferenceName sqaod::preferenceNameFromString(const char *name) {
    if (strcasecmp("algorithm", name) == 0)
        return pnAlgorithm;
    if (strcasecmp("n_trotters", name) == 0)
        return pnNumTrotters;
    if (strcasecmp("tile_size", name) == 0)
        return pnTileSize;
    if (strcasecmp("tile_size_0", name) == 0)
        return pnTileSize0;
    if (strcasecmp("tile_size_1", name) == 0)
        return pnTileSize1;
    if (strcasecmp("precision", name) == 0)
        return pnPrecision;
    if (strcasecmp("experiment", name) == 0)
        return pnExperiment;
    return pnUnknown;
}

const char *sqaod::preferenceNameToString(enum PreferenceName pn) {
    switch (pn) {
    case pnAlgorithm:
        return "algorithm";
    case pnNumTrotters:
        return "n_trotters";
    case pnTileSize:
        return "tile_size";
    case pnTileSize0:
        return "tile_size_0";
    case pnTileSize1:
        return "tile_size_1";
    case pnPrecision:
        return "precision";
    case pnDevice:
        return "device";
    case pnExperiment:
        return "experiment";
    default:
        return "unknown";
    }
}
