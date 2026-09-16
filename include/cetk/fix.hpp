#pragma once
// FIX 4.4 RFQ gateway around the RfqEngine, plus the client that drives it.  The session layer (tag=value, SOH,
// BodyLength, CheckSum, sequence numbers, Logon / Heartbeat / TestRequest / Logout) is standard FIX 4.4, so a
// QuickFIX initiator can connect.  Application messages:
//   R  QuoteRequest      131 QuoteReqID, 55 Symbol, 54 Side (1 client buys / 2 sells), 38 OrderQty (contracts),
//                        20101 ladder tenor index, 20102 client tier index, 20103 request time (seconds in the day)
//   S  Quote             117 QuoteID, 131, 132 BidPx, 133 OfferPx, 134/135 sizes, 20110 half-spread, 20111 valid, 20112 mid
//   D  NewOrderSingle    117 QuoteID (hit the quote), 11 ClOrdID, 54, 38, 44 Price
//   8  ExecutionReport   150=F 39=2 filled: 31 LastPx, 32 LastQty, 20112 mid at fill, 20110 half-spread; 150=8 rejected
//   0  Heartbeat 112=SOD:<day index> starts a day (the reply carries 20120 = hidden flow direction so the client can draw
//      the same RFQ script the in-process run draws); 112=EOD ends it.
// Lockstep and deterministic: the gateway advances the engine to the request time before quoting, so an RFQ script
// produces identical fills in-process and over FIX, which `cetk_fixgw --selftest` asserts.
#include "engine.hpp"
#include <string>
#include <map>
#include <vector>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <deque>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET sock_t;
#define CLOSESOCK closesocket
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
typedef int sock_t;
#define INVALID_SOCKET (-1)
#define CLOSESOCK close
#endif

namespace cetk {
namespace fix {

constexpr char SOH = '\x01';

struct Message {
    std::vector<std::pair<int, std::string>> fields;   // in order
    void set(int tag, const std::string& v) { for (auto& f : fields) if (f.first == tag) { f.second = v; return; } fields.emplace_back(tag, v); }
    void set(int tag, long long v) { set(tag, std::to_string(v)); }
    const std::string* get(int tag) const { for (auto& f : fields) if (f.first == tag) return &f.second; return nullptr; }
    std::string str(int tag) const { auto* p = get(tag); return p ? *p : ""; }
    long long i64(int tag) const { auto* p = get(tag); return p ? std::atoll(p->c_str()) : 0; }
    double num(int tag) const { auto* p = get(tag); return p ? std::atof(p->c_str()) : 0; }
    std::string type() const { return str(35); }
};

inline std::string utc_timestamp(long long ns) {
    long long s = ns / 1000000000LL; int ms = (int)((ns / 1000000LL) % 1000);
    long long days = s / 86400, rem = s % 86400;
    // civil from days (Howard Hinnant)
    long long z = days + 719468; long long era = (z >= 0 ? z : z - 146096) / 146097; unsigned doe = (unsigned)(z - era * 146097);
    unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; long long y = (long long)yoe + era * 400;
    unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100); unsigned mp = (5 * doy + 2) / 153; unsigned d = doy - (153 * mp + 2) / 5 + 1; unsigned m = mp < 10 ? mp + 3 : mp - 9;
    if (m <= 2) ++y;
    char buf[40]; std::snprintf(buf, sizeof buf, "%04lld%02u%02u-%02lld:%02lld:%02lld.%03d", y, m, d, rem / 3600, (rem % 3600) / 60, rem % 60, ms);
    return buf;
}

// Serialise with header (8, 9), body, trailer (10).
inline std::string encode(const Message& body, const std::string& sender, const std::string& target, long long seq, long long now_ns) {
    std::string b;
    b += "35=" + body.type() + SOH;
    b += "49=" + sender + SOH + "56=" + target + SOH + "34=" + std::to_string(seq) + SOH + "52=" + utc_timestamp(now_ns) + SOH;
    for (auto& f : body.fields) if (f.first != 35) b += std::to_string(f.first) + "=" + f.second + SOH;
    std::string m = std::string("8=FIX.4.4") + SOH + "9=" + std::to_string(b.size()) + SOH + b;
    unsigned sum = 0; for (unsigned char c : m) sum += c;
    char cs[8]; std::snprintf(cs, sizeof cs, "%03u", sum % 256);
    m += std::string("10=") + cs + SOH;
    return m;
}

// Parse one complete message (validates BodyLength and CheckSum).
inline bool decode(const std::string& raw, Message& out) {
    out.fields.clear();
    size_t i = 0;
    while (i < raw.size()) {
        size_t eq = raw.find('=', i); if (eq == std::string::npos) return false;
        size_t soh = raw.find(SOH, eq); if (soh == std::string::npos) return false;
        out.fields.emplace_back(std::atoi(raw.substr(i, eq - i).c_str()), raw.substr(eq + 1, soh - eq - 1));
        i = soh + 1;
    }
    auto* cs = out.get(10); if (!cs) return false;
    size_t tail = raw.rfind("10="); unsigned sum = 0; for (size_t k = 0; k < tail; ++k) sum += (unsigned char)raw[k];
    return std::atoi(cs->c_str()) == (int)(sum % 256);
}

// Blocking TCP framing of FIX messages.
class Conn {
public:
    explicit Conn(sock_t s) : s_(s) {}
    ~Conn() { if (s_ != INVALID_SOCKET) CLOSESOCK(s_); }
    bool send_raw(const std::string& m) { size_t off = 0; while (off < m.size()) { int n = ::send(s_, m.data() + off, (int)(m.size() - off), 0); if (n <= 0) return false; off += (size_t)n; } return true; }
    // Returns false on disconnect.
    bool recv_msg(std::string& out) {
        while (true) {
            // a message ends with "10=xxx<SOH>"
            size_t p = buf_.find("\x01" "10=");
            if (p != std::string::npos) { size_t e = buf_.find(SOH, p + 1); if (e != std::string::npos) { out = buf_.substr(0, e + 1); buf_.erase(0, e + 1); return true; } }
            char tmp[65536]; int n = ::recv(s_, tmp, sizeof tmp, 0);
            if (n <= 0) return false;
            buf_.append(tmp, (size_t)n);
        }
    }
    sock_t raw() const { return s_; }
private:
    sock_t s_; std::string buf_;
};

inline void net_init() {
#ifdef _WIN32
    static bool done = false; if (!done) { WSADATA w; WSAStartup(MAKEWORD(2, 2), &w); done = true; }
#endif
}
inline sock_t listen_on(int port) {
    net_init();
    sock_t s = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof one);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons((unsigned short)port);
    if (::bind(s, (sockaddr*)&a, sizeof a) != 0) throw Error("bind failed on port " + std::to_string(port));
    if (::listen(s, 1) != 0) throw Error("listen failed");
    return s;
}
inline sock_t connect_to(int port) {
    net_init();
    sock_t s = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons((unsigned short)port);
    for (int tries = 0; tries < 50; ++tries) { if (::connect(s, (sockaddr*)&a, sizeof a) == 0) return s;
#ifdef _WIN32
        Sleep(100);
#else
        usleep(100000);
#endif
    }
    throw Error("connect failed");
}

// Session bookkeeping shared by both ends.
struct Session {
    std::string sender, target; long long out_seq = 0, in_seq = 0;
    Conn* conn = nullptr;
    bool send(const Message& m, long long now_ns) { return conn->send_raw(encode(m, sender, target, ++out_seq, now_ns)); }
    bool recv(Message& m) {
        std::string raw; if (!conn->recv_msg(raw)) return false;
        if (!decode(raw, m)) throw Error("bad FIX message");
        long long seq = m.i64(34); if (seq != in_seq + 1) throw Error("FIX sequence gap: expected " + std::to_string(in_seq + 1) + " got " + std::to_string(seq));
        in_seq = seq; return true;
    }
};


// ---- gateway ---------------------------------------------------------------------------------------------------
class RfqGateway {
public:
    RfqGateway(RfqEngine& eng, const std::vector<DaySpec>& days) : eng_(eng), days_(days) {}
    void serve(Conn& conn) {
        Session s; s.sender = "CETK"; s.target = "CLIENT"; s.conn = &conn;
        Message m; if (!s.recv(m) || m.type() != "A") throw Error("expected Logon");
        Message logon; logon.set(35, "A"); logon.set(98, 0LL); logon.set(108, 30LL); s.send(logon, now_ns());
        while (s.recv(m)) {
            const std::string t = m.type();
            if (t == "5") { Message lo; lo.set(35, "5"); s.send(lo, now_ns()); return; }
            if (t == "1") { Message hb; hb.set(35, "0"); hb.set(112, m.str(112)); s.send(hb, now_ns()); continue; }
            if (t == "0") {
                std::string id = m.str(112);
                if (id.rfind("SOD:", 0) == 0) { size_t d = (size_t)std::atoll(id.c_str() + 4); if (d >= days_.size()) throw Error("SOD beyond the day list"); eng_.start_day(days_[d]); Message hb; hb.set(35, "0"); hb.set(112, id); hb.set(20120, eng_.flow_dir() > 0 ? "1" : "-1"); s.send(hb, now_ns()); }
                else if (id == "EOD") { eng_.end_day(); Message hb; hb.set(35, "0"); hb.set(112, id); s.send(hb, now_ns()); }
                continue;
            }
            if (t == "R") {
                eng_.advance(m.num(20103));
                auto q = eng_.quote((int)m.i64(20102), (int)m.i64(20101), m.i64(54) == 1 ? +1 : -1, m.num(38));
                long long qid = ++quote_id_; if (q.valid) pending_[qid] = q;
                Message qm; qm.set(35, "S"); qm.set(117, qid); qm.set(131, m.str(131)); qm.set(55, m.str(55)); qm.set(20111, q.valid ? 1LL : 0LL);
                if (q.valid) { qm.set(132, fmt(q.bid)); qm.set(133, fmt(q.ask)); qm.set(134, fmt(q.z)); qm.set(135, fmt(q.z)); qm.set(20110, fmt(q.delta)); qm.set(20112, fmt(q.mid)); }
                s.send(qm, now_ns()); continue;
            }
            if (t == "D") {
                long long qid = m.i64(117); auto it = pending_.find(qid); Message er; er.set(35, "8"); er.set(11, m.str(11)); er.set(117, qid); er.set(17, (long long)++exec_id_);
                if (it == pending_.end()) { er.set(150, "8"); er.set(39, "8"); er.set(58, "unknown or expired quote"); }
                else { Fill f = eng_.execute(it->second); pending_.erase(it); er.set(150, "F"); er.set(39, "2"); er.set(54, f.side > 0 ? "1" : "2"); er.set(32, fmt(f.z)); er.set(31, fmt(f.mid + f.side * f.delta)); er.set(20112, fmt(f.mid)); er.set(20110, fmt(f.delta)); er.set(20113, (long long)eng_.stats().fills.size()); }
                s.send(er, now_ns()); continue;
            }
        }
    }
private:
    static std::string fmt(double x) { char b[32]; std::snprintf(b, sizeof b, "%.17g", x); return b; }
    long long now_ns() const { return (long long)(eng_.t() * 1e9); }
    RfqEngine& eng_; const std::vector<DaySpec>& days_; std::map<long long, RfqEngine::QuoteOut> pending_; long long quote_id_ = 0, exec_id_ = 0;
};

// ---- client: replays the RFQ script over FIX and records the fills ---------------------------------------------
struct ClientFill { double t; int tier, tenor, side; double z, price, delta; };
class RfqClient {
public:
    RfqClient(Conn& conn) { s_.sender = "CLIENT"; s_.target = "CETK"; s_.conn = &conn; Message logon; logon.set(35, "A"); logon.set(98, 0LL); logon.set(108, 30LL); s_.send(logon, 0); Message m; if (!s_.recv(m) || m.type() != "A") throw Error("logon rejected"); }
    double start_day(size_t d) { Message hb; hb.set(35, "0"); hb.set(112, "SOD:" + std::to_string(d)); s_.send(hb, 0); Message m; while (s_.recv(m)) if (m.type() == "0" && m.str(112) == "SOD:" + std::to_string(d)) return m.num(20120); throw Error("no SOD ack"); }
    void end_day() { Message hb; hb.set(35, "0"); hb.set(112, "EOD"); s_.send(hb, 0); Message m; while (s_.recv(m)) if (m.type() == "0" && m.str(112) == "EOD") return; throw Error("no EOD ack"); }
    // one RFQ: returns true if filled
    bool rfq(const Rfq& r, const ClientTier& tier, const std::string& symbol) {
        Message q; q.set(35, "R"); q.set(131, std::to_string(++req_)); q.set(55, symbol); q.set(54, r.side > 0 ? "1" : "2"); q.set(38, fmt(r.z)); q.set(20101, (long long)r.tenor); q.set(20102, (long long)r.tier); q.set(20103, fmt(r.t)); s_.send(q, (long long)(r.t * 1e9));
        Message m; while (s_.recv(m)) if (m.type() == "S") break; if (m.type() != "S") throw Error("no quote");
        n_quotes_++; if (m.i64(20111) != 1) return false; double delta = m.num(20110);
        if (r.u >= tier.f(delta)) return false;
        Message d; d.set(35, "D"); d.set(11, "O" + std::to_string(req_)); d.set(117, m.str(117)); d.set(55, symbol); d.set(54, r.side > 0 ? "1" : "2"); d.set(38, fmt(r.z)); d.set(44, r.side > 0 ? m.str(133) : m.str(132)); d.set(40, "2"); s_.send(d, (long long)(r.t * 1e9));
        Message er; while (s_.recv(er)) if (er.type() == "8") break; if (er.str(150) != "F") return false;
        fills.push_back({r.t, r.tier, r.tenor, r.side, er.num(32), er.num(31), er.num(20110)}); return true;
    }
    void logout() { Message lo; lo.set(35, "5"); s_.send(lo, 0); Message m; s_.recv(m); }
    std::vector<ClientFill> fills; size_t n_quotes_ = 0;
private:
    static std::string fmt(double x) { char b[32]; std::snprintf(b, sizeof b, "%.17g", x); return b; }
    Session s_; long long req_ = 0;
};

} // namespace fix
} // namespace cetk
