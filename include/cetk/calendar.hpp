#pragma once
// Contract calendar: delivery months, NYMEX expiry rules, maturities in years, seasonal phase.
//   WTI (CL): trading terminates 3 business days before the 25th calendar day of the month preceding delivery
//             (if the 25th is not a business day, 3 business days before the last business day before the 25th).
//   Henry Hub (NG): 3 business days before the first calendar day of the delivery month.
// Business days are weekdays; exchange holidays are not modelled (stated in the README; the EIA contract-1
// series is used to check the mapping empirically).
#include <string>
#include <cstdio>
#include <cmath>
#include <stdexcept>

namespace cetk {

struct Error : std::runtime_error { using std::runtime_error::runtime_error; };

struct Ymd { int y = 1970, m = 1, d = 1; };

inline long long days_from_civil(int y, unsigned m, unsigned d) {   // Howard Hinnant
    y -= m <= 2; const long long era = (y >= 0 ? y : y - 399) / 400; const unsigned yoe = (unsigned)(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1; const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + (long long)doe - 719468;
}
inline Ymd civil_from_days(long long z) {
    z += 719468; const long long era = (z >= 0 ? z : z - 146096) / 146097; const unsigned doe = (unsigned)(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; const long long y = (long long)yoe + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100); const unsigned mp = (5 * doy + 2) / 153; const unsigned d = doy - (153 * mp + 2) / 5 + 1; const unsigned m = mp < 10 ? mp + 3 : mp - 9;
    return {(int)(y + (m <= 2)), (int)m, (int)d};
}
inline long long to_days(const Ymd& x) { return days_from_civil(x.y, (unsigned)x.m, (unsigned)x.d); }
inline Ymd parse_date(const std::string& s) { Ymd x; if (std::sscanf(s.c_str(), "%d-%d-%d", &x.y, &x.m, &x.d) != 3) throw Error("bad date " + s); return x; }
inline std::string fmt_date(const Ymd& x) { char b[16]; std::snprintf(b, sizeof b, "%04d-%02d-%02d", x.y, x.m, x.d); return b; }
inline std::string fmt_month(int y, int m) { char b[16]; std::snprintf(b, sizeof b, "%04d-%02d", y, m); return b; }
inline int weekday(long long days) { long long w = (days + 4) % 7; if (w < 0) w += 7; return (int)w; }   // 0 = Sunday
inline bool is_business_day(long long days) { int w = weekday(days); return w != 0 && w != 6; }
inline long long add_business_days(long long days, int n) { int step = n < 0 ? -1 : 1; int left = std::abs(n); while (left > 0) { days += step; if (is_business_day(days)) --left; } return days; }
inline long long prev_business_day_or_self(long long days) { while (!is_business_day(days)) --days; return days; }

// Expiry (last trading day) of the contract for delivery month (y, m) as days since epoch.
inline long long expiry_days(const std::string& root, int y, int m) {
    if (root == "NG") return add_business_days(days_from_civil(y, (unsigned)m, 1), -3);
    // CL: month before delivery
    int py = y, pm = m - 1; if (pm == 0) { pm = 12; --py; }
    long long d25 = prev_business_day_or_self(days_from_civil(py, (unsigned)pm, 25));
    return add_business_days(d25, -3);
}
// Years from date to the contract's expiry (ACT/365)
inline double years_to_expiry(const std::string& root, const Ymd& date, int y, int m) { return (double)(expiry_days(root, y, m) - to_days(date)) / 365.0; }
// Seasonal phase of a delivery month in [0, 1): January = 0
inline double season_phase(int m) { return (m - 1) / 12.0; }

// The n-th nearest unexpired delivery month on a date (n = 1 is the prompt contract), as (y, m).
inline std::pair<int, int> nth_delivery(const std::string& root, const Ymd& date, int n) {
    long long today = to_days(date); int y = date.y, m = date.m; int count = 0;
    for (int k = 0; k < 400; ++k) {
        if (expiry_days(root, y, m) > today) { if (++count == n) return {y, m}; }
        if (++m == 13) { m = 1; ++y; }
    }
    throw Error("nth_delivery");
}
inline int months_between(int y0, int m0, int y1, int m1) { return (y1 - y0) * 12 + (m1 - m0); }

} // namespace cetk
