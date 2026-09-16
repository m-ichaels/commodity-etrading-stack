#pragma once
// CSV loaders for the derived data: daily futures curves, bar panels, EIA front contracts, feature tables.
#include "calendar.hpp"
#include "json.hpp"
#include <vector>
#include <string>
#include <map>
#include <cstdlib>
#include <cmath>
#include <algorithm>

namespace cetk {

struct Table {   // a CSV with a header row; numeric access by column name, NaN when empty
    std::vector<std::string> cols; std::vector<std::vector<std::string>> rows;
    int col(const std::string& c) const { for (size_t i = 0; i < cols.size(); ++i) if (cols[i] == c) return (int)i; return -1; }
    double num(size_t r, int c) const { if (c < 0 || c >= (int)rows[r].size() || rows[r][c].empty()) return NAN; return std::atof(rows[r][c].c_str()); }
    const std::string& str(size_t r, int c) const { static std::string e; return (c < 0 || c >= (int)rows[r].size()) ? e : rows[r][c]; }
};
inline Table read_table(const std::string& path) {
    Table t; auto lines = split_lines(read_file(path)); if (lines.empty()) throw Error("empty " + path);
    t.cols = split_csv(lines[0]); for (auto& c : t.cols) while (!c.empty() && (c.back() == '\r' || c.back() == ' ')) c.pop_back();
    for (size_t i = 1; i < lines.size(); ++i) t.rows.push_back(split_csv(lines[i]));
    return t;
}

// ---- futures curves ----------------------------------------------------------------------------------------
struct CurvePoint { std::string delivery; int dy = 0, dm = 0; int months = 0; double T = 0, phase = 0, price = 0, volume = 0; std::string source; };
struct CurveDay { std::string date; Ymd ymd; std::vector<CurvePoint> pts; int gap_months = 0;   // number of missing months inside the observed range
    const CurvePoint* at_months(int m) const { for (auto& p : pts) if (p.months == m) return &p; return nullptr; } };

inline std::vector<CurveDay> read_curves(const std::string& path, const std::string& root) {
    Table t = read_table(path); int cd = t.col("date"), cdel = t.col("delivery"), cp = t.col("price"), cs = t.col("source"), cv = t.col("volume");
    std::map<std::string, CurveDay> days;
    for (size_t r = 0; r < t.rows.size(); ++r) {
        const std::string& d = t.str(r, cd); if (d.size() < 10) continue;
        CurveDay& day = days[d]; if (day.date.empty()) { day.date = d; day.ymd = parse_date(d); }
        CurvePoint p; p.delivery = t.str(r, cdel); if (std::sscanf(p.delivery.c_str(), "%d-%d", &p.dy, &p.dm) != 2) continue;
        p.price = t.num(r, cp); if (!(p.price > 0)) continue;
        p.T = years_to_expiry(root, day.ymd, p.dy, p.dm); if (p.T <= 0) continue;
        p.phase = season_phase(p.dm); p.source = t.str(r, cs); p.volume = cv >= 0 ? t.num(r, cv) : 0; if (std::isnan(p.volume)) p.volume = 0;
        auto first = nth_delivery(root, day.ymd, 1); p.months = months_between(first.first, first.second, p.dy, p.dm) + 1;
        day.pts.push_back(p);
    }
    std::vector<CurveDay> out; for (auto& kv : days) { CurveDay d = kv.second; std::sort(d.pts.begin(), d.pts.end(), [](const CurvePoint& a, const CurvePoint& b) { return a.months < b.months; });
        // de-duplicate delivery months (EIA and Yahoo can both report one): keep the first (EIA listed first by the builder)
        std::vector<CurvePoint> u; for (auto& p : d.pts) if (u.empty() || u.back().months != p.months) u.push_back(p); d.pts = u;
        d.gap_months = 0; for (size_t i = 1; i < d.pts.size() && d.pts[i - 1].months < 24; ++i) d.gap_months += std::min(24, d.pts[i].months) - d.pts[i - 1].months - 1;   // missing months inside the first two years; beyond that the exchange lists only some months
        if (d.pts.size() >= 4) out.push_back(d); }
    return out;
}

// ---- bar panels ----------------------------------------------------------------------------------------------
struct Bar { long long ts = 0; int months = 0; std::string delivery; double open = 0, high = 0, low = 0, close = 0, volume = 0; };
inline std::vector<Bar> read_bars(const std::string& path) {
    Table t = read_table(path); int ct = t.col("ts"), cdel = t.col("delivery"), cm = t.col("months"), co = t.col("open"), ch = t.col("high"), cl = t.col("low"), cc = t.col("close"), cv = t.col("volume");
    std::vector<Bar> v; v.reserve(t.rows.size());
    for (size_t r = 0; r < t.rows.size(); ++r) { Bar b; b.ts = (long long)t.num(r, ct); b.delivery = t.str(r, cdel); b.months = (int)t.num(r, cm); b.open = t.num(r, co); b.high = t.num(r, ch); b.low = t.num(r, cl); b.close = t.num(r, cc); b.volume = t.num(r, cv); if (std::isnan(b.volume)) b.volume = 0; if (b.close > 0) v.push_back(b); }
    std::sort(v.begin(), v.end(), [](const Bar& a, const Bar& b) { return a.ts != b.ts ? a.ts < b.ts : a.months < b.months; });
    return v;
}

// ---- EIA front contracts (date, c1..c4) ------------------------------------------------------------------------
struct FrontRow { std::string date; double c[4]; };
inline std::vector<FrontRow> read_front(const std::string& path) {
    Table t = read_table(path); int cd = t.col("date"); std::vector<FrontRow> v;
    for (size_t r = 0; r < t.rows.size(); ++r) { FrontRow f; f.date = t.str(r, cd); bool ok = true; for (int k = 0; k < 4; ++k) { f.c[k] = t.num(r, t.col("c" + std::to_string(k + 1))); if (!(f.c[k] > 0)) ok = false; } if (ok && f.date.size() == 10) v.push_back(f); }
    return v;
}

// ---- generic feature table keyed by date ------------------------------------------------------------------------
struct Features { std::vector<std::string> cols; std::map<std::string, std::vector<double>> by_date;
    double get(const std::string& date, const std::string& col, double d = 0) const { auto it = by_date.find(date); if (it == by_date.end()) return d; for (size_t i = 0; i < cols.size(); ++i) if (cols[i] == col) { double v = it->second[i]; return std::isnan(v) ? d : v; } return d; } };
inline Features read_features(const std::string& path) {
    Table t = read_table(path); Features f; f.cols = t.cols; int cd = t.col("date");
    for (size_t r = 0; r < t.rows.size(); ++r) { std::vector<double> v; for (size_t c = 0; c < t.cols.size(); ++c) v.push_back(t.num(r, (int)c)); f.by_date[t.str(r, cd)] = v; }
    return f;
}

} // namespace cetk
