#pragma once
// Minimal JSON reader/writer. Enough for exchange messages and our own config/result files;
// no external dependency so the engine builds with a bare compiler.
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <variant>

namespace cetk {

class Json {
public:
    using Array  = std::vector<Json>;
    using Object = std::vector<std::pair<std::string, Json>>;   // insertion order kept

    enum Type { Null, Bool, Number, String, ArrayT, ObjectT };

    Json() : t_(Null) {}
    Json(bool b) : t_(Bool), b_(b) {}
    Json(int v) : t_(Number), n_(v) {}
    Json(long v) : t_(Number), n_((double)v) {}             // int64_t is `long` on LP64 Linux
    Json(unsigned long v) : t_(Number), n_((double)v) {}
    Json(long long v) : t_(Number), n_((double)v) {}
    Json(unsigned long long v) : t_(Number), n_((double)v) {}
    Json(double v) : t_(Number), n_(v) {}
    Json(const char* s) : t_(String), s_(s) {}
    Json(std::string s) : t_(String), s_(std::move(s)) {}
    Json(Array a) : t_(ArrayT), a_(std::make_shared<Array>(std::move(a))) {}
    Json(Object o) : t_(ObjectT), o_(std::make_shared<Object>(std::move(o))) {}
    template <class T> Json(const std::vector<T>& v) : t_(ArrayT), a_(std::make_shared<Array>()) {
        for (auto& x : v) a_->push_back(Json(x));
    }

    static Json object() { return Json(Object{}); }
    static Json array()  { return Json(Array{}); }

    Type type() const { return t_; }
    bool is_null() const { return t_ == Null; }
    bool is_number() const { return t_ == Number; }
    bool is_string() const { return t_ == String; }
    bool is_array() const { return t_ == ArrayT; }
    bool is_object() const { return t_ == ObjectT; }

    double num() const { if (t_ != Number) throw std::runtime_error("json: not a number"); return n_; }
    long long i64() const { return (long long)std::llround(num()); }
    bool boolean() const { if (t_ != Bool) throw std::runtime_error("json: not a bool"); return b_; }
    const std::string& str() const { if (t_ != String) throw std::runtime_error("json: not a string"); return s_; }
    const Array& arr() const { if (t_ != ArrayT) throw std::runtime_error("json: not an array"); return *a_; }
    Array& arr() { if (t_ != ArrayT) throw std::runtime_error("json: not an array"); return *a_; }
    const Object& obj() const { if (t_ != ObjectT) throw std::runtime_error("json: not an object"); return *o_; }
    Object& obj() { if (t_ != ObjectT) throw std::runtime_error("json: not an object"); return *o_; }

    size_t size() const { return t_ == ArrayT ? a_->size() : t_ == ObjectT ? o_->size() : 0; }
    const Json& operator[](size_t i) const { return arr().at(i); }
    const Json& operator[](int i) const { return arr().at((size_t)i); }
    Json& operator[](size_t i) { return arr().at(i); }
    Json& operator[](int i) { return arr().at((size_t)i); }

    bool has(const std::string& k) const {
        if (t_ != ObjectT) return false;
        for (auto& kv : *o_) if (kv.first == k) return true;
        return false;
    }
    const Json& operator[](const std::string& k) const {
        for (auto& kv : obj()) if (kv.first == k) return kv.second;
        static Json null; return null;
    }
    const Json& operator[](const char* k) const { return (*this)[std::string(k)]; }
    Json& operator[](const std::string& k) {
        if (t_ == Null) { t_ = ObjectT; o_ = std::make_shared<Object>(); }
        for (auto& kv : obj()) if (kv.first == k) return kv.second;
        o_->emplace_back(k, Json());
        return o_->back().second;
    }
    Json& operator[](const char* k) { return (*this)[std::string(k)]; }
    Json& set(const std::string& k, Json v) { (*this)[k] = std::move(v); return *this; }
    void push(Json v) { if (t_ == Null) { t_ = ArrayT; a_ = std::make_shared<Array>(); } arr().push_back(std::move(v)); }

    // Typed getters with defaults (for config files)
    double get(const std::string& k, double d) const { return has(k) && (*this)[k].is_number() ? (*this)[k].num() : d; }
    long long get(const std::string& k, long long d) const { return has(k) && (*this)[k].is_number() ? (*this)[k].i64() : d; }
    int get(const std::string& k, int d) const { return (int)get(k, (long long)d); }
    std::string get(const std::string& k, const std::string& d) const { return has(k) && (*this)[k].is_string() ? (*this)[k].str() : d; }
    bool get(const std::string& k, bool d) const { return has(k) && (*this)[k].t_ == Bool ? (*this)[k].b_ : d; }

    // ---- parsing -------------------------------------------------------------------------
    static Json parse(const std::string& s) { const char* p = s.c_str(); return parse(p, p + s.size()); }
    static Json parse(const char* p, const char* end) {
        Parser ps{p, end};
        Json j = ps.value();
        ps.ws();
        return j;
    }

    // ---- serialisation -------------------------------------------------------------------
    std::string dump(int indent = 0) const { std::string out; dump_to(out, indent, 0); return out; }

private:
    struct Parser {
        const char* p; const char* end;
        void ws() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }
        [[noreturn]] void fail(const char* m) { throw std::runtime_error(std::string("json parse: ") + m); }
        Json value() {
            ws();
            if (p >= end) fail("unexpected end");
            switch (*p) {
                case '{': return object();
                case '[': return array();
                case '"': return Json(string());
                case 't': expect("true"); return Json(true);
                case 'f': expect("false"); return Json(false);
                case 'n': expect("null"); return Json();
                default: return number();
            }
        }
        void expect(const char* lit) {
            size_t n = std::strlen(lit);
            if ((size_t)(end - p) < n || std::strncmp(p, lit, n) != 0) fail("bad literal");
            p += n;
        }
        Json number() {
            const char* s = p;
            if (p < end && (*p == '-' || *p == '+')) ++p;
            while (p < end && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '-' || *p == '+')) ++p;
            if (s == p) fail("bad number");
            return Json(std::strtod(std::string(s, p).c_str(), nullptr));
        }
        std::string string() {
            ++p; // opening quote
            std::string out;
            while (p < end && *p != '"') {
                if (*p == '\\') {
                    ++p; if (p >= end) fail("bad escape");
                    switch (*p) {
                        case '"': out += '"'; break; case '\\': out += '\\'; break; case '/': out += '/'; break;
                        case 'b': out += '\b'; break; case 'f': out += '\f'; break; case 'n': out += '\n'; break;
                        case 'r': out += '\r'; break; case 't': out += '\t'; break;
                        case 'u': { // keep it simple: encode BMP code point as UTF-8
                            if (end - p < 5) fail("bad \\u");
                            unsigned cp = (unsigned)std::strtoul(std::string(p + 1, p + 5).c_str(), nullptr, 16);
                            p += 4;
                            if (cp < 0x80) out += (char)cp;
                            else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
                            else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
                            break;
                        }
                        default: fail("bad escape");
                    }
                    ++p;
                } else out += *p++;
            }
            if (p >= end) fail("unterminated string");
            ++p;
            return out;
        }
        Json array() {
            ++p; Array a; ws();
            if (p < end && *p == ']') { ++p; return Json(std::move(a)); }
            while (true) {
                a.push_back(value()); ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == ']') { ++p; break; }
                fail("expected , or ]");
            }
            return Json(std::move(a));
        }
        Json object() {
            ++p; Object o; ws();
            if (p < end && *p == '}') { ++p; return Json(std::move(o)); }
            while (true) {
                ws(); if (p >= end || *p != '"') fail("expected key");
                std::string k = string(); ws();
                if (p >= end || *p != ':') fail("expected :");
                ++p;
                o.emplace_back(std::move(k), value()); ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == '}') { ++p; break; }
                fail("expected , or }");
            }
            return Json(std::move(o));
        }
    };

    static void dump_str(std::string& out, const std::string& s) {
        out += '"';
        for (char c : s) {
            switch (c) {
                case '"': out += "\\\""; break; case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break; case '\r': out += "\\r"; break; case '\t': out += "\\t"; break;
                default: out += c;
            }
        }
        out += '"';
    }
    void dump_to(std::string& out, int indent, int depth) const {
        auto nl = [&](int d) { if (indent) { out += '\n'; out.append((size_t)(indent * d), ' '); } };
        switch (t_) {
            case Null: out += "null"; break;
            case Bool: out += b_ ? "true" : "false"; break;
            case Number: {
                if (std::isfinite(n_) && n_ == std::floor(n_) && std::fabs(n_) < 1e15) {
                    char b[32]; std::snprintf(b, sizeof b, "%lld", (long long)n_); out += b;
                } else if (!std::isfinite(n_)) out += "null";
                else { char b[32]; std::snprintf(b, sizeof b, "%.10g", n_); out += b; }
                break;
            }
            case String: dump_str(out, s_); break;
            case ArrayT: {
                out += '[';
                bool first = true;
                for (auto& v : *a_) { if (!first) out += ','; first = false; nl(depth + 1); v.dump_to(out, indent, depth + 1); }
                if (!a_->empty()) nl(depth);
                out += ']'; break;
            }
            case ObjectT: {
                out += '{';
                bool first = true;
                for (auto& kv : *o_) {
                    if (!first) out += ','; first = false; nl(depth + 1);
                    dump_str(out, kv.first); out += indent ? ": " : ":"; kv.second.dump_to(out, indent, depth + 1);
                }
                if (!o_->empty()) nl(depth);
                out += '}'; break;
            }
        }
    }

    Type t_;
    bool b_ = false;
    double n_ = 0;
    std::string s_;
    std::shared_ptr<Array> a_;
    std::shared_ptr<Object> o_;
};

inline bool file_exists(const std::string& path) { FILE* f = std::fopen(path.c_str(), "rb"); if (!f) return false; std::fclose(f); return true; }
inline std::string read_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string s;
    char buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    std::fclose(f);
    return s;
}
inline void write_file(const std::string& path, const std::string& content) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    std::fwrite(content.data(), 1, content.size(), f);
    std::fclose(f);
}

inline std::vector<std::string> split_lines(const std::string& s) { std::vector<std::string> v; std::string cur; for (char c : s) { if (c == '\n') { if (!cur.empty() && cur.back() == '\r') cur.pop_back(); if (!cur.empty()) v.push_back(cur); cur.clear(); } else cur += c; } if (!cur.empty()) v.push_back(cur); return v; }
inline std::vector<std::string> split_csv(const std::string& line, char sep = ',') { std::vector<std::string> v; std::string cur; for (char c : line) { if (c == sep) { v.push_back(cur); cur.clear(); } else cur += c; } v.push_back(cur); return v; }

} // namespace cetk
