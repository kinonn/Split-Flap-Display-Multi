#pragma once

#include <Arduino.h>

// Single CSV tokenizer shared by the .ino, SplitFlapEspNow and
// SplitFlapWebServer. Trimmed tokens; "" for an out-of-range index. This
// unifies three drifted copies — the old extractFromCSV in the .ino did not
// trim and returned the WHOLE string for an out-of-range index, which the
// other two callers already handled as "".
inline String getCsvToken(const String &csv, int index) {
    int tokenStart = 0;
    int tokenIndex = 0;

    for (int i = 0; i <= (int) csv.length(); i++) {
        if (i == (int) csv.length() || csv[i] == ',') {
            if (tokenIndex == index) {
                String token = csv.substring(tokenStart, i);
                token.trim();
                return token;
            }
            tokenStart = i + 1;
            tokenIndex++;
        }
    }

    return "";
}
